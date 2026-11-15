"""Explicit staged GPT/CIFAR fitting; domain-balanced losses and independent audits.

No base-model optimizer runs here. All targets are immutable cached training data.
A bounded pilot ends at a review checkpoint, never a convergence declaration.
"""
from __future__ import annotations
import argparse,dataclasses,hashlib,json,os,time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from baseline.model import GPT
from baseline.execution import enable_deterministic_execution
from rewriter.gpt_precision import configure_precision
from rewriter.multibank import MultiStateBankReader,MultiTeacherCacheReader,file_sha
from rewriter.mixed_domain import MixedBankCatalog,BalancedDomainBatches
from rewriter.cifar_bank import CifarBankReader,normalized_images,tensor_tree
from rewriter.cifar_adapter import CifarPreservation,cifar_groups
from rewriter.grouped_norm import GroupedNormObjective
from rewriter.accelerated_grouped_norm import AcceleratedGroupedNormObjective
from rewriter.offline_systems import CachedBaseNormPreservation,VerifiedStateConstants,select_panel_blocks
from rewriter.chisel_factory import build_chisel
from rewriter.meta_optimizer import build_meta_adam
from rewriter.tokenwise_reference import TokenwiseConfig,TokenwiseDual,tokenwise_panel
from rewriter.image_tokenwise_panel import image_tokenwise_panel
from scripts.capture_cifar_chisel_bank import load_airbench
from scripts.train_multistate_chisel import (build_proposal,aggregate_controller_gradients,all_rank_objects,
    norm_preservation_diagnostics,norm_consistency_summary,tokenwise_aggregate,choose_audit_states,stable_seed,save_atomic)


def validate_recipe(recipe):
    if recipe.get('schema')!='chisel_mixed_pilot_v1':raise ValueError('explicit mixed pilot schema required')
    if recipe['domain_weights']!={'gpt':.5,'cifar10':.5}:raise ValueError('pilot fixes equal domain weights')
    if not recipe['architecture'].get('trajectory_mask_features'):raise ValueError('mixed actor requires explicit trajectory mask')
    for domain in ('gpt','cifar10'):
        c=TokenwiseConfig(**recipe['objectives'][domain])
        if c.ranking_k!=100 or c.ranking_tie_policy!='exact_teacher_ties':raise ValueError('mixed pilot requires exact teacher ties and top100')
    if recipe['image_microbatch']!=128 or recipe['images_per_model']%128:raise ValueError('cached CNN convolution shape must remain128')
    if recipe['gpt_microbatch']!=16 or recipe['sequences_per_checkpoint']%16:raise ValueError('cached GPT shape must remain16')
    if recipe['head_lr']<=0 or recipe['representation_lr_multiplier']<=0:raise ValueError('positive actor learning rates required')
    if recipe['updates']<1:raise ValueError('positive pilot review budget required')
    backend=recipe.get('gpt_backend','reference')
    if backend not in ('reference','tokenwise_tie_graph'):raise ValueError('unknown mixed GPT backend')
    if backend=='tokenwise_tie_graph' and (recipe['sequences_per_checkpoint']!=64 or recipe.get('shrinkage_coefficient')!=1. or not recipe.get('compile_actor',True)):
        raise ValueError('mixed graph requires compiled actor,64sequences and unit norm coefficient')


def measurement_row(measured,stats,**identity):
    if measured.get('domain',identity['domain'])!=identity['domain']:raise ValueError('measurement domain mismatch')
    return {**measured,**stats,**identity}


def require_graph_identity(record,uid):
    if record.state_id!=uid:raise ValueError('GPT graph state namespace differs from mixed dual namespace')


def image_teacher_role(panel_role):
    if panel_role not in ('train','audit'):raise ValueError('unknown image panel role')
    return 'past_training' if panel_role=='train' else 'withheld_training'


def write_config_header(output,metadata,*,previous=None,updates=0):
    output=Path(output);header=output/'config.json'
    if not header.exists():header.write_text(json.dumps(previous or metadata,indent=2))
    elif previous is None:raise ValueError('initial run config already exists')
    if previous is not None:
        target=output/f'config_resume_{updates:06d}.json'
        text=json.dumps(metadata,indent=2)
        if target.exists() and target.read_text()!=text:raise ValueError('resume provenance record already differs')
        target.write_text(text)


def graph_denominator(gpt_batch,domain_weight):
    value=gpt_batch/domain_weight
    if value!=int(value) or value<1:raise ValueError('graph normalization must have an exact integer denominator')
    return int(value)


def domain_summary(rows):
    if not rows:raise ValueError('empty domain readout')
    return dict(states=len(rows),mean_kl=sum(r['mean_kl'] for r in rows)/len(rows),
        worst_checkpoint_mean_kl=max(r['mean_kl'] for r in rows),
        mean_group_l2_percentage_reduction=sum(r['mean_group_l2_percentage_reduction'] for r in rows)/len(rows),
        **norm_consistency_summary(rows),**tokenwise_aggregate(rows))


def validate_resume(old,new):
    import copy
    old=copy.deepcopy(old);new=copy.deepcopy(new)
    before=old['recipe'];after=new['recipe']
    if after['updates']<before['updates']:raise ValueError('cannot shorten the saved review budget')
    if not set(before['readout_updates']).issubset(after['readout_updates']):raise ValueError('cannot remove declared readout milestones')
    for value in (before,after):
        value.pop('updates');value.pop('readout_updates')
    if old!=new:raise ValueError('mixed resume source/config/data identity differs')


def run(recipe_path,output,resume=None):
    recipe=json.loads(Path(recipe_path).read_text());validate_recipe(recipe)
    world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'));local=int(os.environ.get('LOCAL_RANK','0'))
    enable_deterministic_execution();device=torch.device('cuda',local);torch.cuda.set_device(device);torch.set_num_threads(4)
    configure_precision('fp32')
    if world>1:dist.init_process_group('nccl')
    catalog=MixedBankCatalog(recipe['mixed_index'],dict(gpt=lambda p:MultiStateBankReader(p,verify_once=True,mmap_model=True),
        cifar10=lambda p:CifarBankReader(p.parent)))
    if catalog.weights!=recipe['domain_weights']:raise ValueError('index domain weights differ')
    sampler=BalancedDomainBatches(catalog,recipe['batch_counts'],world_size=world,seed=recipe['sampler_seed'])
    gpt_reader=catalog.readers['gpt'];cache=MultiTeacherCacheReader(gpt_reader,verify_once=True,mmap=True)
    gpt_model=GPT(**gpt_reader.manifest['provenance']['model_config']).to(device).requires_grad_(False)
    engine=CachedBaseNormPreservation(gpt_model,microbatch_sequences=16,teacher_cache_device=device,kl_dtype=torch.float64,gpt_precision='fp32')
    constants=VerifiedStateConstants(engine,AcceleratedGroupedNormObjective)
    data=Path(recipe['cifar_data']);raw=torch.load(data/'train.pt',map_location='cpu',weights_only=False)
    split=torch.load(data/'chisel_image_split.pt',map_location='cpu',weights_only=False)
    bank_config=json.loads((catalog.readers['cifar10'].path/'config.json').read_text())
    if file_sha(catalog.readers['cifar10'].path/'config.json')!=catalog.readers['cifar10'].manifest['config_sha256']:raise ValueError('CIFAR config identity differs')
    if file_sha(data/'train.pt')!=bank_config['train_sha256'] or file_sha(data/'chisel_image_split.pt')!=bank_config['image_split_sha256']:
        raise ValueError('CIFAR source or split identity differs')
    if bank_config['inference']['teacher_microbatch_images']!=128:raise ValueError('CIFAR cached convolution shape differs')
    images=normalized_images(raw['images'],device);labels=raw['labels'].to(device)
    module=load_airbench(data,defer_test_readouts=True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;configure_precision('fp32')
    cnn_model=module.CifarNet().to(device).float().eval().requires_grad_(False)
    with torch.random.fork_rng(devices=[local]):
        torch.manual_seed(recipe['actor_seed']);actor=build_chisel(recipe['architecture']).to(device)
    optimizer=build_meta_adam(actor,lr=recipe['head_lr'],representation_lr_multiplier=recipe['representation_lr_multiplier'])
    configs={d:TokenwiseConfig(**c) for d,c in recipe['objectives'].items()}
    duals={d:TokenwiseDual([i for i in catalog.split['train'] if catalog.route(i)[0]==d],config=configs[d],dual_lr=recipe['dual_lr']) for d in configs}
    root=Path(__file__).resolve().parents[1]
    sources={str(p.relative_to(root)):file_sha(p) for folder in ('rewriter','baseline') for p in sorted((root/folder).rglob('*.py'))}
    sources['scripts/train_mixed_chisel.py']=file_sha(__file__)
    sources['scripts/train_multistate_chisel.py']=file_sha(root/'scripts/train_multistate_chisel.py')
    sources['scripts/capture_cifar_chisel_bank.py']=file_sha(root/'scripts/capture_cifar_chisel_bank.py')
    metadata=dict(recipe=recipe,mixed_index_sha256=file_sha(recipe['mixed_index']),source_sha256=sources,world_size=world,
        semantics='Each domain mean gets weight0.5; each per-state dual updates only on that state visit; no validation gradients',
        context_units={'gpt':'token','cifar10':'image'},initialization='fresh masked actor, Adam and per-state duals')
    output=Path(output);updates=0;compute={};history=[]
    if resume:
        saved=torch.load(resume,map_location=device,weights_only=False)
        validate_resume(saved['metadata'],metadata)
        actor.load_state_dict(saved['controller']);optimizer.load_state_dict(saved['optimizer'])
        for d in duals:duals[d].load_state_dict(saved['duals'][d])
        updates=saved['updates'];compute=saved['per_rank_compute'][rank];history=saved['history']
        torch.set_rng_state(saved['per_rank_rng'][rank]['torch'].cpu());torch.cuda.set_rng_state(saved['per_rank_rng'][rank]['cuda'].cpu(),device)
    if world>1:
        for p in actor.parameters():dist.broadcast(p.data,src=0)
    actor_forward=torch.compile(actor,fullgraph=True,dynamic=False) if recipe.get('compile_actor',True) else actor
    graph=None;graph_compute_seen={}
    if recipe.get('gpt_backend','reference')=='tokenwise_tie_graph':
        from rewriter.graph_tokenwise_tie import TieAwareTokenwiseGraphAccumulator
        graph=TieAwareTokenwiseGraphAccumulator(actor,actor_forward,engine,cache,constants,radius_fraction=recipe['radius_fraction'],shrinkage_coefficient=1.)
    if rank==0:
        output.mkdir(parents=True,exist_ok=bool(resume));write_config_header(output,metadata,previous=saved['metadata'] if resume else None,updates=updates)
    if world>1:dist.barrier()
    logfile=(output/'metrics.jsonl').open('a' if resume else 'w') if rank==0 else None
    start=time.perf_counter();wb=None
    if rank==0 and recipe.get('wandb',True):
        import wandb
        wb=wandb.init(project='chisel',mode='offline',dir=str(output),config=metadata,name=output.name)
    def log(event,**kw):
        if rank==0:
            row=dict(event=event,update=updates,elapsed_seconds=time.perf_counter()-start,**kw)
            text=json.dumps(row,allow_nan=False);print(text,flush=True);logfile.write(text+'\n');logfile.flush()
            if wb:wb.log({k:v for k,v in row.items() if isinstance(v,(int,float))})
    def count(values):
        for k,v in values.items():compute[k]=compute.get(k,0)+v
    def select_gpt_panel(local_id,epoch,learning,role):
        panel=cache.read(local_id,role=role);size=panel.metadata['microbatch_sequences'];n=recipe['sequences_per_checkpoint']
        blocks=list(range(len(panel.inputs)//size));g=torch.Generator().manual_seed(stable_seed(recipe['sampler_seed'],local_id,max(epoch,0)))
        if learning:blocks=torch.randperm(len(blocks),generator=g).tolist()
        return select_panel_blocks(panel,blocks[:n//size])
    def graph_state(uid,epoch):
        _,local_id=catalog.route(uid);record=gpt_reader.read(local_id,device=device,model_device='cpu');require_graph_identity(record,uid);panel=select_gpt_panel(local_id,epoch,True,'train')
        measured,stats=graph.accumulate(record,panel,duals['gpt']);stats.update(norm_preservation_diagnostics(stats,graph.graph.grouped))
        for key,value in graph.compute.items():
            count({key:value-graph_compute_seen.get(key,0)});graph_compute_seen[key]=value
        return dict(state_id=uid,local_state_id=local_id,domain='gpt',panel_role='train',context_ids=list(panel.sequence_ids),**measured,**stats)
    def evaluate(uid,epoch,learning,weight=1.,role='train'):
        domain,local_id=catalog.route(uid);config=configs[domain]
        t=time.perf_counter()
        if domain=='gpt':
            record=gpt_reader.read(local_id,device=device,model_device='cpu');constants.load(cache,record);grouped=constants.grouped()
            panel=select_gpt_panel(local_id,epoch,learning,role);active=engine;snapshot=record.snapshot
            inputs=panel.inputs;ids=list(panel.sequence_ids);provider=lambda a,b:panel.teacher_batch(engine,a,b)
        else:
            record=catalog.readers[domain].read(local_id);record.assert_verified();cnn_model.load_state_dict(record.model,strict=True)
            active=CifarPreservation(cnn_model,microbatch_images=128);snapshot=tensor_tree(record.snapshot,device)
            grouped=GroupedNormObjective(active.parameters,groups=cifar_groups(active.model),zero_reference_policy='freeze')
            pool=split['fitting_indices' if role=='train' else 'heldout_indices']
            g=torch.Generator().manual_seed(stable_seed('images',recipe['sampler_seed'],uid,max(epoch,0)))
            ids=pool[torch.randperm(len(pool),generator=g)[:recipe['images_per_model']]].tolist()
            image_ids=torch.tensor(ids,dtype=torch.long);inputs=images[image_ids.to(device)].contiguous(memory_format=torch.channels_last)
            target=labels[image_ids.to(device)];cached=record.teacher_logits[image_ids]
            provider=lambda a,b:active.teacher_batch(inputs[a:b],cached[a:b],ids[a:b],data_role=image_teacher_role(role))
        with torch.enable_grad():
            params,reg,stats,editable,credit=build_proposal(actor_forward,snapshot,active,grouped,radius_fraction=recipe['radius_fraction'],
                norm_squared=constants.radius_squared(grouped) if domain=='gpt' else None,shrinkage_coefficient=recipe['shrinkage_coefficient'])
        if domain=='gpt':
            before_head=getattr(engine,'cached_teacher_head_forwards',0)
            result=tokenwise_panel(active,params,reg,inputs,config=config,dual=duals[domain],state_id=uid,state_weight=weight,
                actor_credit=credit,teacher_provider=provider,backward=learning,return_ce=not learning)
            count(engine.compute.as_dict());count({'cached_teacher_head_forwards':getattr(engine,'cached_teacher_head_forwards',0)-before_head})
        else:
            result=image_tokenwise_panel(active,params,reg,inputs,target,config=config,dual=duals[domain],state_id=uid,state_weight=weight,
                actor_credit=credit,teacher_provider=provider,backward=learning);count(result['compute'])
        stats.update(norm_preservation_diagnostics(stats,grouped))
        row=measurement_row(result,stats,state_id=uid,local_state_id=local_id,domain=domain,panel_role=role,context_ids=ids,wall_seconds=time.perf_counter()-t)
        return row
    def audit():
        if graph is not None:graph.close()
        rows=[]
        for domain in configs:
            train=[i for i in catalog.split['train'] if catalog.route(i)[0]==domain]
            dev=[i for i in catalog.split['audit'] if catalog.route(i)[0]==domain]
            for population,pool,number in [('fitting',train,recipe['audit_fitting_states'][domain]),('development',dev,recipe['audit_development_states'][domain])]:
                for uid in choose_audit_states(pool,number)[rank::world]:
                    row=evaluate(uid,-1,False,role='train' if population=='fitting' else 'audit');row['population']=population;rows.append(row)
        rows=[r for group in all_rank_objects(rows,world) for r in group]
        summary={d:{p:domain_summary([r for r in rows if r['domain']==d and r['population']==p]) for p in ('fitting','development')} for d in configs}
        if rank==0:
            save_atomic(output/f'audit_{updates:06d}.pt',dict(rows=rows,summary=summary));(output/f'audit_{updates:06d}.json').write_text(json.dumps(dict(rows=rows,summary=summary),indent=2))
        history.append(dict(update=updates,summary=summary));log('audit',summary=summary)
    def checkpoint():
        rng=all_rank_objects(dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device)),world)
        counts=all_rank_objects(compute,world)
        if rank==0:
            saved=dict(controller=actor.state_dict(),optimizer=optimizer.state_dict(),duals={d:v.state_dict() for d,v in duals.items()},
                metadata=metadata,updates=updates,next_epoch=updates//sampler.steps_per_epoch,next_batch=updates%sampler.steps_per_epoch,
                per_rank_rng=rng,per_rank_compute=counts,history=history,completed=False,stop_reason='bounded review, not converged')
            save_atomic(output/f'controller_update_{updates:06d}.pt',saved);save_atomic(output/'controller_latest.pt',saved)
    if updates==0:audit();checkpoint()
    while updates<recipe['updates']:
        epoch,batch=divmod(updates,sampler.steps_per_epoch);assigned,clocks=sampler.batch(epoch,batch,rank=rank)
        optimizer.zero_grad(set_to_none=True);beg=time.perf_counter();rows=[]
        if graph is not None:
            gpt_items=[x for x in assigned if x['domain']=='gpt']
            denominator=graph_denominator(recipe['batch_counts']['gpt'],recipe['domain_weights']['gpt'])
            graph.begin_update([x['state_id'] for x in gpt_items],global_checkpoint_count=denominator)
            for item in gpt_items:rows.append(graph_state(item['state_id'],clocks['gpt']['completed_cycles']))
            graph.finish_update()  # Attach GPT gradients first; CNN backwards ADD to these buffers.
        for item in assigned:
            if graph is not None and item['domain']=='gpt':continue
            cycle=clocks[item['domain']]['completed_cycles']
            rows.append(evaluate(item['state_id'],cycle,True,item['state_weight']))
        aggregate_controller_gradients(actor,world)
        grad=float(torch.nn.utils.clip_grad_norm_(actor.parameters(),recipe['gradient_clip']))
        if not torch.isfinite(torch.tensor(grad)):raise FloatingPointError('nonfinite shared actor gradient')
        optimizer.step();all_rows=[r for group in all_rank_objects(rows,world) for r in group]
        for d,dual in duals.items():
            values=[(r['state_id'],r['mean_kl']) for r in all_rows if r['domain']==d];dual.step(values,expected_ids=[i for i,k in values])
        updates+=1;torch.cuda.synchronize();log('update',wall_seconds=time.perf_counter()-beg,preclip_gradient_norm=grad,domain_clocks=clocks,
            summary={d:domain_summary([r for r in all_rows if r['domain']==d]) for d in configs},rows=all_rows)
        if updates in recipe['readout_updates']:audit()
        if updates%recipe['checkpoint_every']==0 or updates in recipe['readout_updates'] or updates==recipe['updates']:checkpoint()
    if graph is not None:graph.close()
    log('review_complete',converged=False,updates_completed=updates)
    if wb:wb.finish()
    if world>1:dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser();p.add_argument('--recipe',required=True);p.add_argument('--output',required=True);p.add_argument('--resume')
    a=p.parse_args();run(a.recipe,a.output,a.resume)
if __name__=='__main__':main()
