"""Bounded fitting-only fresh-private adaptation with an immutable shared actor."""
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baseline.execution import enable_deterministic_execution
from rewriter.gpt_precision import configure_precision
from rewriter.coordinate_chisel import CoordinateChisel
from rewriter.coordinate_initialization import initialize_cached_weight_qr,initialize_warm_private
from rewriter.coordinate_optimization import clip_private_gradients
from rewriter.distributed_probe_store import _digest
from rewriter.multibank import file_sha
from rewriter.tokenwise_reference import TokenwiseConfig,TokenwiseDual
from scripts.continue_coordinate_chisel import stage_origin,resolve_actor_architecture
from scripts.qualify_coordinate_chisel import FixedContexts,parameter_motion
from scripts.train_multistate_chisel import save_atomic


def validate_recipe(r):
    if r['schema']!='coordinate_private_adaptation_v1':raise ValueError('explicit adaptation recipe required')
    if r['shared_frozen'] is not True or r['private_optimizer']['initialization']!='fresh' or r['dual_initialization']!='fresh_zero':
        raise ValueError('frozen shared / fresh private optimizer and dual required')
    if r['updates']!=128 or r['readout_updates']!=[0,1,4,16,64,128]:raise ValueError('predeclared fitting curve budget differs')
    if r['gpt_sequences']!=16 or r['cifar_images']!=1024:raise ValueError('fixed support panel differs')
    names=[c['name'] for c in r['cases']]
    if len(names)!=len(set(names)):raise ValueError('duplicate adaptation case')
    for case in r['cases']:
        if case['domain'] not in ('gpt','cifar10') or case['initializer'] not in ('random','cached_weight_qr','warm499'):
            raise ValueError('unknown fitting domain or initialization')
        if case['initializer']=='warm499' and (case['domain'],case['local_id'])!=('gpt',501):raise ValueError('warm control is only499 to501')


def measure_reused_context(contexts,context,*args,**kwargs):
    # FixedContexts normally reloads each state, resetting engine counters. This
    # curve deliberately retains one state: subtract prior cumulative counters
    # from its existing accounting so each actual forward/backward counts once.
    prior=context['engine'].compute.as_dict() if context['domain']=='gpt' else {}
    result=contexts.measure(context,*args,**kwargs)
    for name,value in prior.items():contexts.compute[name]-=value
    return result


def validate_fresh_optimizer(group,parent_group,recipe):
    for name in ('eps','betas'):
        if group[name]!=parent_group[name]:raise ValueError('fresh optimizer differs from parent')
    transition=recipe.get('private_lr_transition')
    if transition is None:
        if group['lr']!=parent_group['lr']:raise ValueError('fresh private LR differs without explicit transition')
    elif transition!={'from':parent_group['lr'],'to':group['lr']}:
        raise ValueError('private LR transition does not match pinned parent and recipe')


def run(recipe_path,output):
    r=json.loads(Path(recipe_path).read_text());validate_recipe(r)
    source=Path(r['checkpoint'])
    if file_sha(source)!=r['checkpoint_sha256']:raise ValueError('parent checkpoint SHA differs')
    saved=torch.load(source,map_location='cpu',weights_only=False)
    if saved['update']!=r['parent_update'] or file_sha(source)!=r['checkpoint_sha256']:raise ValueError('parent update/identity differs')
    origin=stage_origin(saved['metadata']);base=json.loads(Path(r['base_recipe']).read_text())
    if file_sha(r['base_recipe'])!=origin['base_recipe_sha256'] or file_sha(base['mixed_index'])!=origin['mixed_index_sha256']:
        raise ValueError('base model/data recipe changed')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    enable_deterministic_execution();torch.cuda.set_device(0);torch.set_num_threads(4);configure_precision('fp32');device=torch.device('cuda',0)
    _,architecture=resolve_actor_architecture({},saved['metadata']);architecture.pop('name');architecture.pop('decoder_scale')
    actor=CoordinateChisel(**architecture).to(device);actor.load_state_dict(saved['actor'],strict=True);actor.requires_grad_(False);actor.zero_grad(set_to_none=True)
    shared_digest=_digest(actor.state_dict());contexts=FixedContexts(base,device,gpt_sequences=r['gpt_sequences'],cifar_images=r['cifar_images'])
    configs={d:TokenwiseConfig(**v['config']) for d,v in saved['duals'].items()}
    root=Path(__file__).resolve().parents[1]
    sources={str(p.relative_to(root)):file_sha(p) for folder in ('rewriter','baseline') for p in (root/folder).rglob('*.py')}
    sources['scripts/adapt_coordinate_private.py']=file_sha(__file__)
    metadata=dict(recipe=r,sources=sources,parent_metadata=saved['metadata'],shared_digest=shared_digest,
        scope='fitting-only private adaptation; frozen shared model; fresh Adam and dual per case; no query or global updates',
        qr_scope='shared actor was trained with random-initialized fitted probes; QR comparison can reflect input-distribution mismatch')
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2))
    import wandb
    wb=wandb.init(project='chisel',mode='offline',dir=str(output),name=r['name'],config=metadata)
    (output/'wandb-path.json').write_text(json.dumps(dict(path=str(Path(wb.dir).parent),id=wb.id)))
    log_file=(output/'metrics.jsonl').open('w');beg=time.perf_counter();summaries=[]
    def log(row):
        row={**row,'elapsed_seconds':time.perf_counter()-beg};text=json.dumps(row,allow_nan=False);print(text,flush=True);log_file.write(text+'\n');log_file.flush()
        metrics={k:v for k,v in row.items() if isinstance(v,(int,float))};item=row.get('row',{})
        metrics.update({row.get('case','case')+'/'+k:v for k,v in item.items() if isinstance(v,(int,float))});wb.log(metrics)
    for case in r['cases']:
        domain,uid=case['domain'],case['local_id'];reader=contexts.catalog.readers[domain]
        if uid not in reader.manifest['split']['train']:raise ValueError('adaptation case is not a TRAIN model/state')
        torch.set_rng_state(saved['rng'].cpu());torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
        context=contexts.load(domain,uid);key=(domain,'fitting_private_adaptation',uid)
        private=actor.make_private(context['snapshot'],state_key=key,seed=case['seed'])
        initialization={'schema':'original_seeded_random_v1','seed':case['seed']}
        if case['initializer']=='cached_weight_qr':initialization=initialize_cached_weight_qr(private,context['snapshot'])
        elif case['initializer']=='warm499':
            source_record=reader.read(499,device='cpu',model_device='cpu')
            equal=all(torch.equal(o.anchors.A.cpu(),context['snapshot'].matrices[n].anchors.A.cpu()) and torch.equal(o.anchors.B.cpu(),context['snapshot'].matrices[n].anchors.B.cpu()) for n,o in source_record.snapshot.matrices.items())
            initialization=initialize_warm_private(private,saved['private'][('gpt',499)]['parameters'],source_state_sha256=reader.entries[498]['sha256'],anchors_equal=equal)
            del source_record
        before={n:p.detach().cpu().clone() for n,p in private.named_parameters()}
        cfg=r['private_optimizer'];opt=torch.optim.Adam(private.parameters(),lr=cfg['lr'],betas=tuple(cfg['betas']),eps=cfg['eps'])
        source_key=next(k for k in saved['private'] if k[0]==domain)
        parent_group=saved['private'][source_key]['optimizer']['param_groups'][0]
        validate_fresh_optimizer(opt.param_groups[0],parent_group,r)
        dual=TokenwiseDual([uid],config=configs[domain],dual_lr=saved['duals'][domain]['dual_lr'])
        case_dir=output/case['name'];case_dir.mkdir();(case_dir/'initialization.json').write_text(json.dumps(initialization,indent=2))
        with torch.no_grad():
            x,_,_=actor._inputs(context['snapshot'],private);token_rms=float(x.double().square().mean().sqrt());del x
        log(dict(event='initialization',case=case['name'],initialization=initialization,encoded_token_rms=token_rms,private_adam_entries=len(opt.state)))
        for update in range(r['updates']+1):
            if update in r['readout_updates']:
                row=measure_reused_context(contexts,context,actor,private,configs[domain],dual,learning=False)
                log(dict(event='fixed_support_readout',case=case['name'],private_update=update,row=row,private_motion=parameter_motion(before,private),compute=dict(contexts.compute)))
                save_atomic(case_dir/f'checkpoint_{update:06d}.pt',dict(metadata=metadata,case=case,initialization=initialization,parent_checkpoint_sha256=r['checkpoint_sha256'],private=private.state_dict(),optimizer=opt.state_dict(),dual=dual.state_dict(),private_update=update,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),compute=dict(contexts.compute)))
            if update==r['updates']:break
            tick=time.perf_counter();opt.zero_grad(set_to_none=True)
            row=measure_reused_context(contexts,context,actor,private,configs[domain],dual,learning=True,weight=1.)
            if any(p.grad is not None for p in actor.parameters()):raise RuntimeError('private adaptation leaked shared gradients')
            norm=float(clip_private_gradients(private.parameters(),cfg['clip']));opt.step();dual.step([(uid,row['mean_kl'])],expected_ids=[uid]);torch.cuda.synchronize()
            log(dict(event='private_update',case=case['name'],private_update=update+1,row=row,preclip_gradient_norm=norm,wall_seconds=time.perf_counter()-tick,compute=dict(contexts.compute)))
        if _digest(actor.state_dict())!=shared_digest:raise RuntimeError('frozen shared actor changed')
        summaries.append(dict(case=case,final=row,initialization=initialization));del private,opt,dual,context
    (output/'summary.json').write_text(json.dumps(dict(results=summaries,shared_unchanged=True,compute=contexts.compute,elapsed_seconds=time.perf_counter()-beg),indent=2));wb.finish()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--recipe',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.recipe,a.output)
