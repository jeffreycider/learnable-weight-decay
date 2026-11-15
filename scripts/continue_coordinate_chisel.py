"""Pinned seen-only continuation of the closed coordinate pilot.

No heldout support/query evaluation, no policy deployment, and no automatic
panel expansion. Every arm starts from exactly the same preserved numeric state.
"""
from __future__ import annotations
import argparse,copy,json,sys,time,math
from types import SimpleNamespace
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baseline.execution import enable_deterministic_execution
from rewriter.gpt_precision import configure_precision
from rewriter.coordinate_chisel import CoordinateChisel
from rewriter.coordinate_optimization import configure_shared_continuation,clip_shared_continuation,clip_private_gradients,effective_bfloat16_motion
from rewriter.meta_optimizer import build_meta_adam
from rewriter.multibank import file_sha
from rewriter.tokenwise_reference import TokenwiseConfig,TokenwiseDual
from rewriter.edit_scope import editable_names
from scripts.qualify_coordinate_chisel import FixedContexts,parameter_motion
from scripts.train_multistate_chisel import save_atomic


def stage_origin(metadata):
    """Resolve immutable data/architecture provenance through explicit stages."""
    seen=set()
    while 'protocol' not in metadata:
        if id(metadata) in seen:raise ValueError('cyclic stage provenance')
        seen.add(id(metadata))
        metadata=metadata.get('parent_metadata')
        if not isinstance(metadata,dict):raise ValueError('missing original data/architecture provenance')
    for key in ('base_recipe_sha256','mixed_index_sha256'):
        if not metadata.get(key):raise ValueError('missing original data/architecture provenance')
    return metadata


def validate_stage(recipe,initial,resumed=None):
    """A new bounded stage may start at any closed, hash-pinned update."""
    if recipe['schema'] not in ('coordinate_seen_continuation_v1','coordinate_seen_continuation_v2'):
        raise ValueError('explicit continuation recipe required')
    if recipe.get('actor_backend','eager') not in ('eager','eager_cuda_graph_v1'):
        raise ValueError('unknown coordinate actor backend')
    start=recipe.get('start_update',32 if recipe['schema'].endswith('_v1') else None)
    end=recipe.get('end_update')
    if type(start) is not int or type(end) is not int or start<0 or end<=start:
        raise ValueError('explicit increasing start/end update budget required')
    if initial['update']!=start:raise ValueError('declared parent update differs')
    if type(recipe['checkpoint_every']) is not int or recipe['checkpoint_every']<=0:
        raise ValueError('positive checkpoint cadence required')
    if any(type(u) is not int or not start<=u<=end for u in recipe['readout_updates']):
        raise ValueError('readout outside declared stage budget')
    if resumed is not None:
        if not start<=resumed['update']<=end:raise ValueError('resume outside declared stage budget')
        if set(resumed['private'])!=set(initial['private']):raise ValueError('resume private identities changed')
    return start,stage_origin(initial['metadata'])


def prepare_actor_graphs(contexts,actor,private):
    """Four resident captures for the bounded fixture, not a sharded store."""
    from rewriter.coordinate_actor_graph import CoordinateActorGraph
    from rewriter.private_probe_slot import PrivateProbeSlot
    graphs={}
    for key,p in private.items():
        actor.zero_grad(set_to_none=True)
        for q in private.values():q.zero_grad(set_to_none=True)
        context=contexts.load(*key)
        editable=editable_names(context['engine'].parameters,'grouped')-context['grouped'].frozen_parameter_names
        graphs[key]=CoordinateActorGraph(actor,PrivateProbeSlot(actor,p),context['snapshot'],editable)
    actor.zero_grad(set_to_none=True)
    for p in private.values():p.zero_grad(set_to_none=True)
    return graphs


def learning_actor(graphs,key,context,actor,private):
    if not graphs:return actor
    graph=graphs[key]
    editable=editable_names(context['engine'].parameters,'grouped')-context['grouped'].frozen_parameter_names
    graph.bind_scope(private,editable)
    return graph


def resolve_stage_panels(recipe,parent_metadata):
    origin=stage_origin(parent_metadata)['protocol']['joint_pilot']
    parent=dict(parent_metadata.get('fitting_panel_config',
        {'gpt_sequences':origin['gpt_sequences'],'cifar_images':origin['cifar_images']}))
    requested=recipe.get('fitting_panels',{})
    if set(requested)-set(parent):raise ValueError('unknown fitting panel override')
    effective={**parent,**requested}
    if effective['gpt_sequences'] not in (16,64,128) or effective['gpt_sequences']<parent['gpt_sequences']:
        raise ValueError('GPT fitting panel must preserve or expand to64/128 training sequences')
    if effective['cifar_images']!=parent['cifar_images']:raise ValueError('CIFAR panel change is not part of this stage protocol')
    changed=effective!=parent
    if changed:
        pin=recipe.get('parent_readout',{})
        if not pin.get('path') or not isinstance(pin.get('sha256'),str) or len(pin['sha256'])!=64:
            raise ValueError('panel expansion requires hash-pinned parent readout')
    return parent,effective,changed


def resolve_actor_architecture(recipe,parent_metadata):
    """Only an explicit write-frame transition may change the actor program."""
    parent=copy.deepcopy(parent_metadata.get('actor_architecture',stage_origin(parent_metadata)['protocol']['architecture']))
    requested=recipe.get('actor_architecture_transition',{})
    if set(requested)-{'write_observation_frame'}:raise ValueError('unrelated actor architecture transition')
    effective={**parent,**requested}
    mode=effective.get('write_observation_frame','encoder')
    if mode not in ('encoder','decoder'):raise ValueError('unknown write observation frame')
    effective['name']='checkpoint_coordinate_ed_writeframe_v2' if mode=='decoder' else 'checkpoint_coordinate_ed_nanogpt_v1'
    if effective!=parent and recipe.get('actor_backend','eager')!='eager':
        raise ValueError('new write-frame transition requires eager qualification')
    return parent,effective


def read_parent_readout(pin,update):
    path=Path(pin['path'])
    if file_sha(path)!=pin['sha256']:raise ValueError('parent readout SHA differs')
    rows=[]
    with path.open() as stream:
        for line in stream:
            row=json.loads(line)
            if row.get('event')=='fixed_fitting_readout' and row.get('update')==update:rows.append(row)
    if file_sha(path)!=pin['sha256'] or len(rows)!=1:raise ValueError('parent readout changed or exact update is not unique')
    return rows[0]['rows']


def parent_readout_differences(actual,expected):
    key=lambda row:(row['domain'],row['local_id'])
    a={key(r):r for r in actual};b={key(r):r for r in expected}
    if len(a)!=len(actual) or len(b)!=len(expected) or a.keys()!=b.keys():raise ValueError('parent readout state coverage differs')
    fields=('context_ids','mean_kl','strict_topk_changes','topk_order_changes','topk_membership_changes',
            'mean_group_l2_percentage_reduction','group_l2_percentage_reduction','realized_edit_norm')
    return [dict(domain=k[0],local_id=k[1],field=f,actual=a[k][f],expected=b[k][f]) for k in a for f in fields if a[k][f]!=b[k][f]]


def configure_stage_optimizers(initial,optimizer,private_optimizers,recipe,*,resuming=False):
    """Derive absolute group settings from the pinned parent, never compound LR."""
    factor=recipe.get('primal_lr_factor',1.)
    if not isinstance(factor,(int,float)) or not math.isfinite(factor) or factor<=0:raise ValueError('primal LR factor must be positive and finite')
    before=copy.deepcopy(initial['shared_optimizer']['param_groups'])
    desired=copy.deepcopy(before)
    transition=recipe.get('shared_optimizer_transition','apply_mode')
    if transition=='apply_mode':configure_shared_continuation(SimpleNamespace(param_groups=desired),recipe['shared_optimizer_mode'])
    elif transition!='preserve':raise ValueError('unknown shared optimizer transition')
    for group in desired:group['lr']*=factor
    def install(opt,groups):
        if len(opt.param_groups)!=len(groups):raise ValueError('optimizer group count changed')
        for actual,target in zip(opt.param_groups,groups):
            if len(actual['params'])!=len(target['params']):raise ValueError('optimizer group parameter count changed')
            settings={k:v for k,v in target.items() if k!='params'}
            if resuming and {k:v for k,v in actual.items() if k!='params'}!=settings:
                raise ValueError('resumed optimizer settings differ from pinned parent transition')
            actual.update(settings)
    install(optimizer,desired)
    private={}
    for key,opt in private_optimizers.items():
        old=copy.deepcopy(initial['private'][key]['optimizer']['param_groups']);new=copy.deepcopy(old)
        for group in new:group['lr']*=factor
        install(opt,new)
        private[str(key)]={'before':old,'after':new}
    return dict(shared={'before':before,'after':desired},private=private,primal_lr_factor=factor,transition=transition)


def restore_numeric_state(saved,actor,private,shared_optimizer,private_optimizers,duals):
    actor.load_state_dict(saved['actor'],strict=True);shared_optimizer.load_state_dict(saved['shared_optimizer'])
    if set(private)!=set(saved['private']):raise ValueError('private checkpoint identities changed')
    for key,module in private.items():
        module.load_state_dict(saved['private'][key]['parameters'],strict=True);private_optimizers[key].load_state_dict(saved['private'][key]['optimizer'])
    # Objective transitions retain numeric multipliers and the exact visit clock;
    # the new loss configuration is recorded explicitly in the new stage header.
    for domain,dual in duals.items():
        old=saved['duals'][domain]
        if set(old['multipliers'])!=set(dual.multipliers) or old['dual_lr']!=dual.dual_lr:raise ValueError('dual identities or clock units changed')
        allowed={'ranking_loss_mode','ranking_weight','loss_backend'}
        old_config={k:v for k,v in old['config'].items() if k not in allowed}
        new_config={k:v for k,v in dual.state_dict()['config'].items() if k not in allowed}
        if old_config!=new_config:raise ValueError('unrelated objective or dual clock units changed')
        dual.multipliers=dict(old['multipliers']);dual.updates=old['updates']
    torch.set_rng_state(saved['rng'].cpu());torch.cuda.set_rng_state(saved['cuda_rng'].cpu())


def run(recipe_path,output,resume=None,stop_after_update=None):
    recipe=json.loads(Path(recipe_path).read_text())
    source=Path(recipe['checkpoint']);sha=file_sha(source)
    if sha!=recipe['checkpoint_sha256']:raise ValueError('input checkpoint hash differs')
    saved=torch.load(source,map_location='cpu',weights_only=False)
    if file_sha(source)!=sha:raise ValueError('immutable checkpoint changed while loading')
    initial=saved
    if resume:saved=torch.load(resume,map_location='cpu',weights_only=False)
    start_update,origin=validate_stage(recipe,initial,saved if resume else None)
    parent_panels,panels,panels_changed=resolve_stage_panels(recipe,initial['metadata'])
    expected_parent=read_parent_readout(recipe['parent_readout'],start_update) if panels_changed else None
    if stop_after_update is not None and not saved['update']<stop_after_update<=recipe['end_update']:
        raise ValueError('requested boundary must follow loaded update within stage budget')
    output=Path(output);output.mkdir(parents=True,exist_ok=bool(resume))
    enable_deterministic_execution();torch.cuda.set_device(0);torch.set_num_threads(4);configure_precision('fp32');device=torch.device('cuda',0)
    old_protocol=origin['protocol'];stage=old_protocol['joint_pilot']
    parent_architecture,actor_architecture=resolve_actor_architecture(recipe,initial['metadata'])
    architecture=dict(actor_architecture);architecture.pop('name');architecture.pop('decoder_scale')
    actor=CoordinateChisel(**architecture).to(device);base=json.loads(Path(recipe['base_recipe']).read_text())
    if file_sha(recipe['base_recipe'])!=origin['base_recipe_sha256']:raise ValueError('base data recipe differs')
    contexts=FixedContexts(base,device,gpt_sequences=panels['gpt_sequences'],cifar_images=panels['cifar_images'])
    if file_sha(base['mixed_index'])!=origin['mixed_index_sha256']:raise ValueError('data index differs')
    keys=list(saved['private']);private={};local_optim={}
    for key in keys:
        domain,uid=key;ctx=contexts.load(domain,uid);extra=saved['private'][key]['parameters']['_extra_state']
        p=actor.make_private(ctx['snapshot'],state_key=extra['state_key'],seed=0);private[key]=p
        local_optim[key]=torch.optim.Adam(p.parameters(),lr=stage['private_lr'],betas=tuple(stage['adam_betas']),eps=stage['private_adam_epsilon'])
    optimizer=build_meta_adam(actor,lr=stage['shared_head_lr'],representation_lr_multiplier=stage['shared_representation_multiplier'])
    configs={d:TokenwiseConfig(**v) for d,v in recipe['objectives'].items()}
    duals={d:TokenwiseDual([uid for domain,uid in keys if domain==d],config=configs[d],dual_lr=old_protocol['objective']['dual_lr']) for d in configs}
    restore_numeric_state(saved,actor,private,optimizer,local_optim,duals)
    graph_start=time.perf_counter();graphs={}
    if recipe.get('actor_backend','eager')=='eager_cuda_graph_v1':
        graphs=prepare_actor_graphs(contexts,actor,private)
        # Capture warms only the actor but may consume RNG or touch gradients.
        # Restore every numerical owner before the first real visit.
        restore_numeric_state(saved,actor,private,optimizer,local_optim,duals)
    graph_setup_seconds=time.perf_counter()-graph_start if graphs else 0.
    optimization=configure_stage_optimizers(initial,optimizer,local_optim,recipe,resuming=bool(resume))
    shared_initial={n:p.detach().cpu().clone() for n,p in initial['actor'].items()};private_initial={key:{n:p.detach().cpu().clone() for n,p in initial['private'][key]['parameters'].items() if isinstance(p,torch.Tensor)} for key in private}
    root=Path(__file__).resolve().parents[1]
    sources={str(p.relative_to(root)):file_sha(p) for folder in ('rewriter','baseline') for p in sorted((root/folder).rglob('*.py'))}
    for file in ('scripts/continue_coordinate_chisel.py','scripts/qualify_coordinate_chisel.py','scripts/train_multistate_chisel.py'):sources[file]=file_sha(root/file)
    metadata=dict(recipe=recipe,parent_metadata=initial['metadata'],input_sha256=sha,sources=sources,shared_optimizer_groups=optimization,
        numeric_initialization=f'exact update{start_update} actor, private maps, both Adam states/clocks, dual values/clocks and RNG; only explicitly recorded optimizer, objective, panel, or write-frame transitions change the computation',
        start_update=start_update,compute_scope='this stage only; parent counts retained in input checkpoint',
        actor_backend=recipe.get('actor_backend','eager'),actor_graph_scope='one resident capture per private checkpoint; eager fixed readouts; student/loss/Adam outside graph',
        fitting_panel_config=panels,parent_fitting_panel_config=parent_panels,
        actor_architecture=actor_architecture,parent_actor_architecture=parent_architecture,
        fitting_panels=f'first {panels["gpt_sequences"]} cached TRAIN sequences per GPT state, micro16; first {panels["cifar_images"]} fitting images per CNN; no query evaluation')
    if resume:
        if saved['metadata']!=metadata:raise ValueError('continuation resume source/config/data differs')
        contexts.compute=dict(saved['compute'])
    else:(output/'metadata.json').write_text(json.dumps(metadata,indent=2))
    logfile=(output/'metrics.jsonl').open('a' if resume else 'w');start=time.perf_counter();update=saved['update']
    import wandb
    wb=wandb.init(project='chisel',mode='offline',dir=str(output),name=recipe['name']+('-resume'+str(update) if resume else ''),config=metadata)
    (output/f'wandb-path-{update:06d}.json').write_text(json.dumps(dict(path=str(Path(wb.dir).parent),id=wb.id,name=wb.name)))
    (output/f'actor-setup-{update:06d}.json').write_text(json.dumps(dict(backend=recipe.get('actor_backend','eager'),capture_count=len(graphs),seconds=graph_setup_seconds)))
    def log(row):
        row={**row,'elapsed_seconds':time.perf_counter()-start};line=json.dumps(row,allow_nan=False);print(line,flush=True);logfile.write(line+'\n');logfile.flush()
        measured={k:v for k,v in row.items() if isinstance(v,(int,float))}
        entries=row.get('rows',[])
        if entries:measured['objective']=sum(r['objective'] for r in entries)/len(entries)
        for item in entries:
            prefix=f"{item['domain']}/{item['local_id']}/"
            measured.update({prefix+k:v for k,v in item.items() if isinstance(v,(int,float))})
            measured[prefix+'strict_rank_change_fraction']=item['strict_topk_changes']/item['token_count']
        for family,values in row.get('shared_motion',{}).items():
            measured.update({'shared_motion/'+family+'/'+k:v for k,v in values.items() if isinstance(v,(int,float))})
        effective=row.get('effective_bfloat16_body_motion',{})
        measured.update({'effective_bfloat16/'+k:v for k,v in effective.items() if isinstance(v,(int,float))})
        for key,families in row.get('private_motion',{}).items():
            for family,values in families.items():measured.update({'private_motion/'+key+'/'+family+'/'+k:v for k,v in values.items() if isinstance(v,(int,float))})
        wb.log(measured)
    def checkpoint():
        payload=dict(metadata=metadata,update=update,actor=actor.state_dict(),shared_optimizer=optimizer.state_dict(),
            private={key:dict(parameters=p.state_dict(),optimizer=local_optim[key].state_dict()) for key,p in private.items()},
            duals={d:v.state_dict() for d,v in duals.items()},rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),compute=dict(contexts.compute))
        target=output/f'checkpoint_{update:06d}.pt'
        if target.exists():raise FileExistsError('immutable checkpoint already exists')
        save_atomic(target,payload);(output/'latest_checkpoint.json').write_text(json.dumps(dict(update=update,path=str(target))))
    def readout():
        rows=[]
        for domain,uid in keys:
            ctx=contexts.load(domain,uid);rows.append(contexts.measure(ctx,actor,private[domain,uid],configs[domain],duals[domain],learning=False))
        log(dict(event='fixed_fitting_readout',update=update,rows=rows,shared_motion=parameter_motion(shared_initial,actor),effective_bfloat16_body_motion=effective_bfloat16_motion(shared_initial,actor),
            private_motion={str(key):parameter_motion(private_initial[key],p) for key,p in private.items()},compute=dict(contexts.compute),
            all_seen_strict_zero_and_positive=all(r['strict_topk_changes']==0 and r['mean_group_l2_percentage_reduction']>0 for r in rows)))
    if not resume:
        if panels_changed:
            rows=[]
            for domain,uid in keys:
                ctx=contexts.load(domain,uid,gpt_sequences=parent_panels['gpt_sequences'])
                rows.append(contexts.measure(ctx,actor,private[domain,uid],configs[domain],duals[domain],learning=False))
            differences=parent_readout_differences(rows,expected_parent)
            log(dict(event='parent_panel_identity_check',update=update,rows=rows,differences=differences,bitexact=not differences,compute=dict(contexts.compute)))
            if differences:raise RuntimeError('original parent fitting panel readout differs before expansion')
        readout();checkpoint()
    while update<recipe['end_update']:
        begin=time.perf_counter();optimizer.zero_grad(set_to_none=True);rows=[]
        for domain,uid in keys:
            key=(domain,uid);p=private[key];local_optim[key].zero_grad(set_to_none=True);ctx=contexts.load(domain,uid)
            row=contexts.measure(ctx,learning_actor(graphs,key,ctx,actor,p),p,configs[domain],duals[domain],learning=True,weight=stage['state_weight'])
            for parameter in p.parameters():
                if parameter.grad is not None:parameter.grad.div_(stage['state_weight'])
            row['private_preclip_gradient_norm']=float(clip_private_gradients(p.parameters(),stage['private_clip']))
            row['private_gradient_norms']={n:float(v.grad.norm()) for n,v in p.named_parameters() if v.grad is not None}
            local_optim[key].step();rows.append(row)
        gradient_norms={n:float(p.grad.norm()) for n,p in actor.named_parameters() if p.grad is not None}
        clipping=clip_shared_continuation(optimizer,recipe['shared_optimizer_mode'],max_norm=1.);optimizer.step()
        for domain,dual in duals.items():
            measurements=[(r['local_id'],r['mean_kl']) for r in rows if r['domain']==domain];dual.step(measurements,expected_ids=[i for i,k in measurements])
        update+=1;torch.cuda.synchronize();log(dict(event='joint_update',update=update,wall_seconds=time.perf_counter()-begin,
            shared_preclip_gradient_norms=clipping,shared_gradient_norms=gradient_norms,rows=rows,compute=dict(contexts.compute)))
        if update in recipe['readout_updates']:readout()
        if update%recipe['checkpoint_every']==0 or update in recipe['readout_updates'] or update==recipe['end_update'] or update==stop_after_update:checkpoint()
        if update==stop_after_update:break
    (output/'summary.json').write_text(json.dumps(dict(status='bounded seen-only continuation complete; convergence not implied' if update==recipe['end_update'] else 'stopped at explicitly requested immutable boundary; stage unfinished',start_update=start_update,end_update=update,compute=contexts.compute,elapsed_seconds=time.perf_counter()-start),indent=2))
    wb.finish()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--recipe',required=True);p.add_argument('--output',required=True);p.add_argument('--resume');p.add_argument('--stop-after-update',type=int);a=p.parse_args();run(a.recipe,a.output,a.resume,a.stop_after_update)
