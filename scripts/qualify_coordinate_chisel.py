"""Bounded real-state E/D prototype; joint learning, no meta-gradient claim.

All private state is resident only for this tiny declared fixture. This is not the
10,000-checkpoint production store or a drop-in qualification of CUDA graphs.
"""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baseline.model import GPT
from baseline.execution import enable_deterministic_execution
from rewriter.gpt_precision import configure_precision
from rewriter.coordinate_chisel import CoordinateChisel
from rewriter.coordinate_optimization import clip_private_gradients
from rewriter.multibank import MultiStateBankReader,MultiTeacherCacheReader,file_sha
from rewriter.mixed_domain import MixedBankCatalog
from rewriter.cifar_bank import CifarBankReader,normalized_images,tensor_tree
from rewriter.cifar_adapter import CifarPreservation,cifar_groups
from rewriter.grouped_norm import GroupedNormObjective
from rewriter.accelerated_grouped_norm import AcceleratedGroupedNormObjective
from rewriter.offline_systems import CachedBaseNormPreservation,VerifiedStateConstants,select_panel_blocks
from rewriter.meta_optimizer import build_meta_adam
from rewriter.tokenwise_reference import TokenwiseConfig,TokenwiseDual,tokenwise_panel
from rewriter.image_tokenwise_panel import image_tokenwise_panel
from scripts.capture_cifar_chisel_bank import load_airbench
from scripts.train_multistate_chisel import build_proposal,norm_preservation_diagnostics,save_atomic


class FixedContexts:
    def __init__(self,base_recipe,device,*,gpt_sequences=16,cifar_images=128,action_family='fixed_anchored_v1',panel_backend='reference'):
        if panel_backend not in ('reference','direct_streamed'):raise ValueError('unknown panel backend')
        if panel_backend=='direct_streamed' and action_family!='adaptive_ucv_v1':raise ValueError('direct_streamed requires adaptive action family')
        self.panel_backend=panel_backend
        self.action_family=action_family
        self.proposal_builder=proposal_builder_for_family(action_family)
        self.device=device;self.gpt_sequences=gpt_sequences;self.cifar_images=cifar_images
        self.catalog=MixedBankCatalog(base_recipe['mixed_index'],dict(gpt=lambda p:MultiStateBankReader(p,verify_once=True,mmap_model=True),cifar10=lambda p:CifarBankReader(p.parent)))
        self.reader=self.catalog.readers['gpt'];self.cache=MultiTeacherCacheReader(self.reader,verify_once=True,mmap=True)
        model=GPT(**self.reader.manifest['provenance']['model_config']).to(device).requires_grad_(False)
        self.gpt=CachedBaseNormPreservation(model,microbatch_sequences=16,teacher_cache_device=device,kl_dtype=torch.float64,gpt_precision='fp32')
        self.constants=VerifiedStateConstants(self.gpt,AcceleratedGroupedNormObjective)
        data=Path(base_recipe['cifar_data']);raw=torch.load(data/'train.pt',map_location='cpu',weights_only=False)
        self.split=torch.load(data/'chisel_image_split.pt',map_location='cpu',weights_only=False)
        reader=self.catalog.readers['cifar10'];config=json.loads((reader.path/'config.json').read_text())
        if file_sha(reader.path/'config.json')!=reader.manifest['config_sha256']:raise ValueError('CIFAR bank config changed')
        if file_sha(data/'train.pt')!=config['train_sha256'] or file_sha(data/'chisel_image_split.pt')!=config['image_split_sha256']:raise ValueError('CIFAR data identity changed')
        if config['inference']['teacher_microbatch_images']!=128:raise ValueError('teacher convolution shape changed')
        self.images=normalized_images(raw['images'],device);self.labels=raw['labels'].to(device)
        module=load_airbench(data,defer_test_readouts=True)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;configure_precision('fp32')
        self.cnn=module.CifarNet().to(device).float().eval().requires_grad_(False)
        self.compute={}

    def add_counts(self,values):
        for key,value in values.items():self.compute[key]=self.compute.get(key,0)+value

    def load(self,domain,local_id,*,role='train',block_offset=0,purpose='fitting',shared_actor=None,gpt_sequences=None):
        reader=self.catalog.readers[domain]
        validate_context_access(reader.manifest['split'],local_id,role=role,purpose=purpose,shared_actor=shared_actor)
        if domain=='gpt':
            record=reader.read(local_id,device=self.device,model_device='cpu');self.constants.load(self.cache,record)
            panel=self.cache.read(local_id,role=role);count=(gpt_sequences or self.gpt_sequences)//16
            panel=select_panel_blocks(panel,list(range(block_offset,block_offset+count)))
            grouped=self.constants.grouped();engine=self.gpt
            return dict(domain=domain,local_id=local_id,snapshot=record.snapshot,engine=engine,grouped=grouped,
                inputs=panel.inputs,ids=list(panel.sequence_ids),teacher=lambda a,b:panel.teacher_batch(engine,a,b),
                norm_squared=self.constants.radius_squared(grouped))
        record=reader.read(local_id);record.assert_verified();self.cnn.load_state_dict(record.model,strict=True)
        engine=CifarPreservation(self.cnn,microbatch_images=128)
        grouped=GroupedNormObjective(engine.parameters,groups=cifar_groups(engine.model),zero_reference_policy='freeze')
        ids=self.split['fitting_indices' if role=='train' else 'heldout_indices'][block_offset*128:block_offset*128+self.cifar_images].tolist()
        indices=torch.tensor(ids);inputs=self.images[indices.to(self.device)].contiguous(memory_format=torch.channels_last)
        cached=record.teacher_logits[indices];target=self.labels[indices.to(self.device)]
        return dict(domain=domain,local_id=local_id,snapshot=tensor_tree(record.snapshot,self.device),engine=engine,grouped=grouped,
            inputs=inputs,ids=ids,target=target,norm_squared=None,
            teacher=lambda a,b:engine.teacher_batch(inputs[a:b],cached[a:b],ids[a:b],data_role='past_training' if role=='train' else 'withheld_training'))

    def measure(self,context,actor,private,config,dual,*,learning,weight=1.,panel_backend=None,gradient_components=False):
        engine=context['engine'];domain=context['domain'];uid=context['local_id']
        backend=panel_backend if panel_backend is not None else getattr(self,'panel_backend','reference')
        if backend not in ('reference','direct_streamed'):raise ValueError('unknown panel backend')
        capture=None
        if backend=='direct_streamed':
            if domain!='gpt' or self.action_family!='adaptive_ucv_v1':raise ValueError('direct_streamed requires GPT adaptive action family')
            if getattr(engine,'gpt_precision',None)!='fp32' or getattr(actor,'factor_precision',None)!='fp64':raise ValueError('direct_streamed requires FP32 GPT and FP64 factors')
            from rewriter.experimental_factor_panel import FactorPanelCapture
            from rewriter.experimental_factor_norm import prepare_factor_panel
            capture=FactorPanelCapture(chunk_tokens=128)
        def proposal(snapshot):
            if self.action_family=='adaptive_ucv_v1' and getattr(actor,'captures_fresh_weight_core',False):
                self.add_counts(actor.fresh_query_counts())
            if self.action_family=='adaptive_ucv_v1' and getattr(actor,'fresh_weight_core',False):
                from rewriter.adaptive_sensing import query_weight_cores
                # CNN observations include the frozen whitening matrix. It is
                # a valid read sensor, while engine.parameters stays writable-only.
                sensed_parameters=({name:p.flatten(1) if p.ndim==4 else p
                                    for name,p in engine._all_parameters.items()}
                                   if domain=='cifar10' else engine.parameters)
                embeddings=tuple(name for name,transpose in engine.transpose_storage.items() if transpose)
                cores=query_weight_cores(sensed_parameters,private,embedding_names=embeddings)
                self.add_counts(dict(fresh_weight_core_calls=1,
                    fresh_weight_core_matrix_queries=len(cores),
                    fresh_weight_core_weight_elements_scanned=sum(sensed_parameters[name].numel() for name in cores)))
                return actor(snapshot,private,weight_cores=cores)
            return actor(snapshot,private)
        with torch.enable_grad():
            if self.action_family=='adaptive_ucv_v1' and actor.configuration().get('action_family')!=self.action_family:
                raise ValueError('adaptive context requires an explicitly adaptive actor')
            options={} if capture is None else dict(execution_capture=capture,materialization_grad=False)
            parameters,regularizer,stats,editable,credit=self.proposal_builder(proposal,context['snapshot'],engine,context['grouped'],
                radius_fraction=.01,norm_squared=context['norm_squared'],shrinkage_coefficient=1.,**options)
            if capture is not None:
                parameters,regularizer=prepare_factor_panel(parameters,context['grouped'],capture,transpose_storage=engine.transpose_storage,chunk_rows=128)
            component_measurement=None
            if gradient_components and learning:
                from rewriter.gradient_components import GradientComponents
                head_ids={id(p) for head in actor.output_heads for p in head.parameters()}
                component_measurement=GradientComponents(regularizer,dict(
                    heads=[p for p in actor.parameters() if id(p) in head_ids],
                    body=[p for p in actor.parameters() if id(p) not in head_ids],
                    private=list(private.parameters())),weight=weight,actor_credit=credit)
        kwargs=dict(config=config,dual=dual,state_id=uid,state_weight=weight,actor_credit=credit,teacher_provider=context['teacher'],backward=learning)
        if domain=='gpt':
            before=getattr(engine,'cached_teacher_head_forwards',0)
            run=tokenwise_panel if capture is None else capture.run
            result=run(engine,parameters,regularizer,context['inputs'],**kwargs)
            self.add_counts(engine.compute.as_dict());self.add_counts(dict(cached_teacher_head_forwards=getattr(engine,'cached_teacher_head_forwards',0)-before))
        else:
            result=image_tokenwise_panel(engine,parameters,regularizer,context['inputs'],context['target'],**kwargs);self.add_counts(result['compute'])
        if component_measurement is not None:
            result['gradient_components']=component_measurement.finish()
        stats.update(norm_preservation_diagnostics(stats,context['grouped']))
        return {**result,**stats,'domain':domain,'local_id':uid,'context_ids':context['ids']}


def proposal_builder_for_family(action_family):
    """Explicit finite-edit dispatch; cached sensor anchors never choose a writer."""
    if action_family=='fixed_anchored_v1':return build_proposal
    if action_family=='adaptive_ucv_v1':
        from rewriter.adaptive_preservation import build_adaptive_proposal
        return build_adaptive_proposal
    raise ValueError('unknown coordinate action family')


def validate_context_access(split,local_id,*,role,purpose,shared_actor=None):
    if role not in ('train','audit'):raise ValueError('unknown context role')
    if purpose=='fitting':
        if role=='train' and local_id not in split['train']:raise ValueError('fitting context is not in fitting model/state split')
    elif purpose in ('private_support','private_query'):
        if local_id not in split['audit']:raise ValueError('transfer requires a wholly held-out model/state')
        if shared_actor is None or any(p.requires_grad or p.grad is not None for p in shared_actor.parameters()):
            raise ValueError('private support/query requires frozen and gradient-cleared shared actor')
        if role!=('train' if purpose=='private_support' else 'audit'):raise ValueError('support/query data role mismatch')
    else:raise ValueError('unknown fitting/support/query purpose')


def parameter_motion(before,module):
    result={}
    for family,predicate in [('encoder',lambda n:n.endswith('.encoder')),('decoder',lambda n:n.endswith('.decoder')),
                             ('trajectory',lambda n:n.startswith('trajectory_input.')),('blocks',lambda n:n.startswith('blocks.')),
                             ('heads',lambda n:n.startswith(('matrix_output','vector_output'))),('lifts',lambda n:n.startswith(('matrix_input','vector_input','norm','token_type')))]:
        names=[n for n,p in module.named_parameters() if predicate(n)]
        if not names:continue
        current=dict(module.named_parameters());old=torch.cat([before[n].double().flatten() for n in names]);new=torch.cat([current[n].detach().cpu().double().flatten() for n in names]);delta=new-old
        entry=dict(change_l2=float(delta.norm()),change_rms=float(delta.square().mean().sqrt()),initial_l2=float(old.norm()),relative_l2=float(delta.norm()/old.norm()) if old.norm()>0 else None)
        if family in ('encoder','decoder'):
            losses=[]
            for n in names:
                a=torch.nn.functional.normalize(before[n].double(),dim=0);b=torch.nn.functional.normalize(current[n].detach().cpu().double(),dim=0)
                losses.extend((1-(a*b).sum(0)).tolist())
            entry.update(mean_one_minus_column_cosine=sum(losses)/len(losses),max_one_minus_column_cosine=max(losses))
        result[family]=entry
    return result


def transfer_readout(protocol,contexts,actor,configs,output,log):
    """Fresh private support optimization; every global parameter stays frozen."""
    spec=protocol['transfer'];stage=protocol['joint_pilot'];actor.zero_grad(set_to_none=True);actor.requires_grad_(False)
    global_before={n:p.detach().cpu().clone() for n,p in actor.named_parameters()};results=[]
    keys=[('gpt',spec['gpt_state']),('cifar10',spec['cifar_seed']+1)]
    for index,(domain,uid) in enumerate(keys):
        ctx=contexts.load(domain,uid,purpose='private_support',shared_actor=actor)
        private=actor.make_private(ctx['snapshot'],state_key=(domain,'heldout',uid),seed=830001+index)
        before={n:p.detach().cpu().clone() for n,p in private.named_parameters()}
        optimizer=torch.optim.Adam(private.parameters(),lr=stage['private_lr'],betas=tuple(stage['adam_betas']),eps=stage.get('private_adam_epsilon',stage.get('adam_epsilon',1e-8)))
        dual=TokenwiseDual([uid],config=configs[domain],dual_lr=protocol['objective']['dual_lr']);support_ids=ctx['ids'];query_rows=[]
        for step in range(spec['support_steps']+1):
            if step in (0,spec['support_steps']):
                query=contexts.load(domain,uid,role='audit',purpose='private_query',shared_actor=actor,gpt_sequences=spec['query_gpt_sequences'])
                if set(support_ids)&set(query['ids']):raise ValueError('support and query contexts overlap')
                row=contexts.measure(query,actor,private,configs[domain],dual,learning=False);row['support_steps']=step;query_rows.append(row)
                log(dict(event='heldout_query',support_steps=step,rows=[row],private_motion=parameter_motion(before,private),compute=dict(contexts.compute)))
            if step==spec['support_steps']:break
            ctx=contexts.load(domain,uid,purpose='private_support',shared_actor=actor);optimizer.zero_grad(set_to_none=True)
            row=contexts.measure(ctx,actor,private,configs[domain],dual,learning=True,weight=1.)
            if any(p.grad is not None for p in actor.parameters()):raise RuntimeError('support leaked a shared gradient')
            row['private_preclip_gradient_norm']=float(clip_private_gradients(private.parameters(),stage['private_clip']));optimizer.step();dual.step([(uid,row['mean_kl'])],expected_ids=[uid])
            log(dict(event='private_support',support_steps=step+1,rows=[row],compute=dict(contexts.compute)))
        save_atomic(output/f'transfer_{domain}_{uid}.pt',dict(private=private.state_dict(),optimizer=optimizer.state_dict(),dual=dual.state_dict(),support_steps=spec['support_steps'],support_ids=support_ids,query=query_rows))
        results.append(dict(domain=domain,local_id=uid,query=query_rows,private_motion=parameter_motion(before,private)))
    for n,p in actor.named_parameters():
        if not torch.equal(p.detach().cpu(),global_before[n]):raise RuntimeError('global weights changed during private support')
    (output/'transfer_summary.json').write_text(json.dumps(dict(shared_unchanged=True,results=results,compute=contexts.compute),indent=2))


def validate_protocol(protocol):
    if protocol['schema']!='coordinate_chisel_bounded_prototype_v1':raise ValueError('explicit coordinate protocol required')
    stage=protocol['joint_pilot']
    if stage['gpt_sequences']%16 or stage['cifar_images']%128:raise ValueError('teacher batch shape mismatch')
    count=len(stage['gpt_states'])+len(stage['cifar_seeds'])
    if count*stage['state_weight']!=1 or len(stage['gpt_states'])!=len(stage['cifar_seeds']):raise ValueError('fixed equal-domain/state normalization required')
    if stage['updates']<3:raise ValueError('at least three visits required to observe zero-initialized gradient paths')


def run(protocol_path,base_recipe_path,output,*,frozen_backbone=False):
    protocol=json.loads(Path(protocol_path).read_text());validate_protocol(protocol)
    recipe=json.loads(Path(base_recipe_path).read_text());output=Path(output);output.mkdir(parents=True,exist_ok=False)
    enable_deterministic_execution();torch.cuda.set_device(0);configure_precision('fp32');torch.set_num_threads(4)
    device=torch.device('cuda',0);stage=protocol['joint_pilot'];architecture=dict(protocol['architecture']);architecture.pop('name')
    if architecture.pop('decoder_scale')!='sqrt_coordinates_over_latent_tokens':raise ValueError('decoder scale differs')
    torch.manual_seed(810001);actor=CoordinateChisel(**architecture).to(device)
    if frozen_backbone:actor.blocks.requires_grad_(False)
    contexts=FixedContexts(recipe,device,gpt_sequences=stage['gpt_sequences'],cifar_images=stage['cifar_images'])
    keys=[('gpt',i) for i in stage['gpt_states']]+[('cifar10',i+1) for i in stage['cifar_seeds']]
    private={};local_optim={}
    for j,(domain,uid) in enumerate(keys):
        ctx=contexts.load(domain,uid);p=actor.make_private(ctx['snapshot'],state_key=(domain,0,uid) if domain=='gpt' else(domain,uid-1),seed=820001+j)
        private[domain,uid]=p;local_optim[domain,uid]=torch.optim.Adam(p.parameters(),lr=stage['private_lr'],betas=tuple(stage['adam_betas']),eps=stage.get('private_adam_epsilon',stage.get('adam_epsilon',1e-8)))
    optimizer=build_meta_adam(actor,lr=stage['shared_head_lr'],representation_lr_multiplier=stage['shared_representation_multiplier'])
    shared_initial={n:p.detach().cpu().clone() for n,p in actor.named_parameters()};private_initial={key:{n:p.detach().cpu().clone() for n,p in module.named_parameters()} for key,module in private.items()}
    spec=protocol['objective'];configs={}
    for domain in ('gpt','cifar10'):
        configs[domain]=TokenwiseConfig(spec['mean_tolerance'],spec['tail_tolerance'],spec['scale'],spec['rho'],
            ranking_weight=spec['gpt_ranking_weight' if domain=='gpt' else 'cifar_ranking_weight'],margin_fraction=spec['margin_fraction'],
            dual_update_mode=spec['dual_mode'],loss_backend='tiled_tie_fp64_v1',ranking_k=100,ranking_tie_policy='exact_teacher_ties')
    duals={d:TokenwiseDual([i for domain,i in keys if domain==d],config=configs[d],dual_lr=spec['dual_lr']) for d in configs}
    root=Path(__file__).resolve().parents[1];sources={str(p.relative_to(root)):file_sha(p) for folder in ('rewriter','baseline') for p in sorted((root/folder).rglob('*.py'))}
    sources['scripts/qualify_coordinate_chisel.py']=file_sha(__file__)
    metadata=dict(protocol=protocol,base_recipe_sha256=file_sha(base_recipe_path),mixed_index_sha256=file_sha(recipe['mixed_index']),sources=sources,
        execution='eager coordinate actor and student; tiled exact-tie FP64 loss; no CUDA graph qualification',state_keys=keys,frozen_residual_blocks=frozen_backbone)
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2));logfile=(output/'metrics.jsonl').open('w');start=time.perf_counter();initial_rows=[];gradient_history=[]
    def log(row):
        row={**row,'elapsed_seconds':time.perf_counter()-start};text=json.dumps(row,allow_nan=False);print(text,flush=True);logfile.write(text+'\n');logfile.flush()
    def save(update):
        save_atomic(output/f'checkpoint_{update:06d}.pt',dict(metadata=metadata,update=update,actor=actor.state_dict(),shared_optimizer=optimizer.state_dict(),
            private={key:dict(parameters=p.state_dict(),optimizer=local_optim[key].state_dict()) for key,p in private.items()},duals={d:v.state_dict() for d,v in duals.items()},
            rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),compute=contexts.compute))
    for update in range(stage['updates']+1):
        if update in stage['readouts']:
            rows=[]
            for domain,uid in keys:
                ctx=contexts.load(domain,uid);rows.append(contexts.measure(ctx,actor,private[domain,uid],configs[domain],duals[domain],learning=False))
            if update==0:initial_rows=rows
            log(dict(event='fixed_fitting_readout',update=update,rows=rows,compute=dict(contexts.compute),shared_motion=parameter_motion(shared_initial,actor),private_motion={str(key):parameter_motion(private_initial[key],p) for key,p in private.items()}));save(update)
        if update==stage['updates']:break
        optimizer.zero_grad(set_to_none=True);rows=[];beg=time.perf_counter()
        for domain,uid in keys:
            key=(domain,uid);p=private[key];local_optim[key].zero_grad(set_to_none=True);ctx=contexts.load(domain,uid)
            row=contexts.measure(ctx,actor,p,configs[domain],duals[domain],learning=True,weight=stage['state_weight'])
            for parameter in p.parameters():
                if parameter.grad is not None:parameter.grad.div_(stage['state_weight'])
            row['private_preclip_gradient_norm']=float(clip_private_gradients(p.parameters(),stage['private_clip']))
            row['private_gradient_norms']={name:float(parameter.grad.norm()) for name,parameter in p.named_parameters() if parameter.grad is not None}
            local_optim[key].step();rows.append(row)
        shared_gradients={name:float(p.grad.norm()) for name,p in actor.named_parameters() if p.grad is not None}
        gradient_history.append(dict(shared=shared_gradients,private=[r['private_gradient_norms'] for r in rows]))
        shared_norm=float(torch.nn.utils.clip_grad_norm_(actor.parameters(),stage['shared_clip'],error_if_nonfinite=True));optimizer.step()
        for domain,dual in duals.items():
            values=[(row['local_id'],row['mean_kl']) for row in rows if row['domain']==domain];dual.step(values,expected_ids=[i for i,k in values])
        torch.cuda.synchronize();log(dict(event='joint_update',update=update+1,wall_seconds=time.perf_counter()-beg,shared_preclip_gradient_norm=shared_norm,shared_gradient_norms=shared_gradients,rows=rows,compute=dict(contexts.compute)))
    gates=dict(initial_realized_edit_zero=all(r['realized_edit_norm']==0 for r in initial_rows),shared_heads_first_visit=any(v>0 for k,v in gradient_history[0]['shared'].items() if k.startswith(('matrix_output','vector_output'))),private_encoder_decoder_later=all(any(v>0 for k,v in p.items() if k.endswith('.encoder')) and any(v>0 for k,v in p.items() if k.endswith('.decoder')) for p in gradient_history[-1]['private']),private_trajectory_contract=all((p.get('trajectory_input.weight',0)==0 if frozen_backbone else p.get('trajectory_input.weight',0)>0) for p in gradient_history[-1]['private']),optimizer_clocks=all(int(state['step'])==stage['updates'] for opt in [optimizer,*local_optim.values()] for state in opt.state.values()))
    (output/'summary.json').write_text(json.dumps(dict(gates=gates,status='bounded fitting fixture complete, not converged',updates=stage['updates'],compute=contexts.compute,elapsed_seconds=time.perf_counter()-start),indent=2))
    if not all(gates.values()):raise RuntimeError('coordinate mechanism gate failed; inspect preserved summary')
    if protocol['transfer'].get('enabled',False):
        transfer_readout(protocol,contexts,actor,configs,output,log)



if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);parser.add_argument('--base-recipe',required=True);parser.add_argument('--output',required=True);parser.add_argument('--frozen-backbone',action='store_true')
    args=parser.parse_args();run(args.protocol,args.base_recipe,args.output,frozen_backbone=args.frozen_backbone)
