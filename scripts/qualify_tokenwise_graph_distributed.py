"""Saved-state distributed numerical and timing check; never writes a live fit.

Both paths sum checkpoint-weighted gradients globally BEFORE clipping/Adam.
Each timing repetition restores the same actor, complete Adam and dual state.
The final output check covers every selected checkpoint and token position.
"""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist


def sum_actor_gradients(actor):
    """Contributions already contain 1/global_checkpoint_count. Never average again."""
    from scripts.train_multistate_chisel import aggregate_controller_gradients
    aggregate_controller_gradients(actor, dist.get_world_size() if dist.is_initialized() else 1)


def relative(candidate, reference):
    a, b = candidate.double(), reference.double()
    if not bool(torch.isfinite(a).all() & torch.isfinite(b).all()):
        raise FloatingPointError('nonfinite numerical comparison')
    return float((a-b).norm()/b.norm().clamp_min(1e-300))


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def actor_hash(state):
    h=hashlib.sha256()
    for name,value in state.items():
        h.update(name.encode());h.update(value.contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    from baseline.execution import enable_deterministic_execution
    from baseline.model import GPT
    from rewriter.chisel_factory import build_chisel
    from rewriter.state_bank import StateBankReader
    from rewriter.teacher_cache import TeacherCacheReader
    from rewriter.offline_systems import CachedBaseNormPreservation, VerifiedStateConstants, select_panel_blocks
    from rewriter.accelerated_grouped_norm import AcceleratedGroupedNormObjective
    from rewriter.tokenwise_reference import TokenwiseConfig, TokenwiseDual, tokenwise_panel
    from rewriter.offline_tokenwise_graph import TokenwiseGraphAccumulator
    from rewriter.meta_optimizer import build_meta_adam
    from scripts.train_multistate_chisel import build_proposal
    from scripts.compare_tokenwise_trainer_outputs import compare

    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('state-bank','teacher-cache','policy','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--policy-sha256', required=True)
    parser.add_argument('--checkpoints', type=int, default=200)
    parser.add_argument('--sequences', type=int, default=64)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--gradient-moments', action='store_true',
        help='Observe existing final-repeat graph gradients; timings include diagnostic overhead')
    args=parser.parse_args()
    rank=int(os.environ['RANK']); world=int(os.environ['WORLD_SIZE'])
    if args.checkpoints % world or args.sequences % 16 or min(args.checkpoints,args.sequences,args.repeats)<1:
        raise ValueError('positive full checkpoint shards and 16-sequence microbatches required')
    enable_deterministic_execution(); torch.set_num_threads(1)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK'])); device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    if rank==0: args.output.mkdir(parents=True,exist_ok=False)
    dist.barrier()
    stream=(args.output/f'rank{rank}.jsonl').open('w')
    def log(event, **values):
        row=dict(event=event, rank=rank, **values)
        stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
        if rank==0: print(json.dumps(row,allow_nan=False),flush=True)
    import wandb
    run=wandb.init(project='chisel',mode='offline',dir=str(args.output),config=vars(args)) if rank==0 else None
    digest=hashlib.sha256(args.policy.read_bytes()).hexdigest()
    if digest!=args.policy_sha256: raise ValueError('saved fixture SHA mismatch')
    saved=torch.load(args.policy,map_location='cpu',weights_only=False)
    cfg=TokenwiseConfig(**saved['metadata']['tokenwise_objective'])
    if cfg.loss_backend!='tiled_fp64_v1':raise ValueError('qualification baseline must match deployed tiled loss')
    reader=StateBankReader(args.state_bank,verify_once=True,mmap_model=True)
    cache=TeacherCacheReader(args.teacher_cache,verify_once=True,mmap=True)
    ids=sorted(reader.manifest['split']['train'])[:args.checkpoints]
    if len(ids)!=args.checkpoints:raise ValueError('insufficient fitting checkpoints')
    local_ids=ids[rank::world]
    engine=CachedBaseNormPreservation(GPT(**reader.manifest['provenance']['model_config']).to(device).requires_grad_(False),
        microbatch_sequences=16,teacher_cache_device=device,kl_dtype=torch.float64,gpt_precision='fp32')
    constants=VerifiedStateConstants(engine,AcceleratedGroupedNormObjective)
    actor=build_chisel(saved['metadata']['architecture']).to(device); parameters=list(actor.parameters())
    forward=torch.compile(actor,fullgraph=True,dynamic=False)
    options=saved['metadata']['options']
    optimizer=build_meta_adam(actor,lr=options['lr'],representation_lr_multiplier=options['representation_lr_multiplier'])
    ds=saved['primal_dual'];dual=TokenwiseDual(ds['multipliers'],config=cfg,dual_lr=ds['dual_lr'])
    def restore():
        actor.load_state_dict(saved['controller']);optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
        dual.load_state_dict(copy.deepcopy(ds));actor.zero_grad(set_to_none=True)
        actual=optimizer.state_dict()
        if actual['param_groups']!=saved['optimizer']['param_groups']:raise ValueError('Adam groups differ after restore')
        for key,values in saved['optimizer']['state'].items():
            for name,value in values.items():
                got=actual['state'][key][name]
                if isinstance(value,torch.Tensor):
                    if not torch.equal(got.cpu(),value.cpu()):raise ValueError('Adam moment/clock restore differs')
                elif value!=got:raise ValueError('Adam scalar restore differs')
        if dual.state_dict()!=ds:raise ValueError('dual restore differs')
    def read(state):
        return reader.read(state,model_device='cpu',device=device),select_panel_blocks(cache.read(state,'train'),range(args.sequences//16))
    radius=options['radius_fraction']; shrinkage=options['shrinkage_coefficient']
    timings=[]; results={}; graphs=None; batch_diagnostic=None
    log('start',policy_sha256=digest,world_size=world,states=ids,local_states=local_ids,
        state_weight=1/args.checkpoints,sequences=args.sequences,restored_saved_adam_and_duals=True)
    for backend in ('tiled_eager','tokenwise_graph'):
        for repeat in range(args.repeats):
            restore()
            observer=None
            if args.gradient_moments and backend=='tokenwise_graph' and repeat==args.repeats-1:
                from rewriter.gradient_batch_diagnostic import CheckpointGradientMoments
                observer=CheckpointGradientMoments(device=device,state_weight=1/args.checkpoints)
            buffers=[torch.zeros_like(p) for p in parameters] if backend=='tiled_eager' else None
            if backend=='tokenwise_graph' and graphs is None:
                graphs=TokenwiseGraphAccumulator(actor,forward,engine,cache,constants,
                    radius_fraction=radius,shrinkage_coefficient=shrinkage,capture_warmups=1)
            if graphs is not None:graphs.begin_update(local_ids,global_checkpoint_count=args.checkpoints)
            torch.cuda.synchronize();dist.barrier();started=time.perf_counter()
            for state in local_ids:
                record,panel=read(state)
                if backend=='tiled_eager':
                    constants.load(cache,record);grouped=constants.grouped();norm=constants.radius_squared(grouped)
                    actor.zero_grad(set_to_none=True)
                    edited,regularizer,_,_,credit=build_proposal(forward,record.snapshot,engine,grouped,
                        radius_fraction=radius,norm_squared=norm,shrinkage_coefficient=shrinkage)
                    tokenwise_panel(engine,edited,regularizer,panel.inputs,config=cfg,dual=dual,state_id=state,
                        state_weight=1/args.checkpoints,actor_credit=credit,
                        teacher_provider=lambda first,last:panel.teacher_batch(engine,first,last))
                    torch._foreach_add_(buffers,[p.grad for p in parameters])
                    del edited,regularizer,credit,grouped
                else:
                    graphs.accumulate(record,panel,dual)
                    if observer is not None:observer.observe(graphs.graph.gradient_buffers)
                del record,panel
            if backend=='tiled_eager':
                for p,g in zip(parameters,buffers):p.grad=g
            else:graphs.finish_update()
            sum_actor_gradients(actor)
            if observer is not None:
                from rewriter.gradient_batch_diagnostic import summarize_checkpoint_gradients
                dist.all_reduce(observer.statistics,op=dist.ReduceOp.SUM)
                batch_diagnostic=summarize_checkpoint_gradients(observer.statistics,
                    [p.grad for p in parameters],batch_sizes=sorted({n for n in
                        (8,16,32,64,100,args.checkpoints) if n<=args.checkpoints}))
                log('checkpoint_gradient_noise',**batch_diagnostic)
            # Host export is explicit and identical in both timed paths.
            gradient=torch.cat([p.grad.flatten() for p in parameters]).detach().cpu()
            grad_norm=torch.nn.utils.clip_grad_norm_(parameters,options['gradient_clip'],error_if_nonfinite=True)
            optimizer.step();torch.cuda.synchronize();elapsed=time.perf_counter()-started
            elapsed_tensor=torch.tensor(elapsed,device=device);dist.all_reduce(elapsed_tensor,op=dist.ReduceOp.MAX)
            timing=dict(backend=backend,repeat=repeat,maximum_rank_seconds=float(elapsed_tensor),local_seconds=elapsed,
                includes_gradient_diagnostic=observer is not None)
            timings.append(timing);log('update',**timing,gradient_norm=float(grad_norm))
        results[backend]=dict(actor=cpu_tree(actor.state_dict()),optimizer=cpu_tree(optimizer.state_dict()),gradient=gradient.clone())
        del gradient,buffers
    capture_count=graphs.capture_count
    graphs.close();graphs=None;actor.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
    reference=results['tiled_eager'];candidate=results['tokenwise_graph']
    gradient_error=relative(candidate['gradient'],reference['gradient'])
    delta=lambda model:torch.cat([(model[k]-saved['controller'][k]).flatten() for k in model])
    delta_error=relative(delta(candidate['actor']),delta(reference['actor']))
    # Full same-context comparison after the finite updates, every selected state.
    direct=[]
    for state in local_ids:
        record,panel=read(state);constants.load(cache,record);grouped=constants.grouped();norm=constants.radius_squared(grouped)
        edits=[]
        for policy in (reference['actor'],candidate['actor']):
            actor.load_state_dict(policy)
            with torch.enable_grad(): proposal=build_proposal(forward,record.snapshot,engine,grouped,
                radius_fraction=radius,norm_squared=norm,shrinkage_coefficient=shrinkage)
            edits.append({n:v.detach() for n,v in proposal[0].items()});del proposal
        for first in range(0,args.sequences,16):
            inputs=panel.inputs[first:first+16].to(device)
            with torch.no_grad():
                a=engine._forward(edits[0],inputs);b=engine._forward(edits[1],inputs);metrics=compare(a,b)
            row=dict(state=state,sequence_ids=panel.sequence_ids[first:first+16],**metrics)
            direct.append(row);log('direct_output',**row);del a,b
        del edits,record,panel,grouped
    local=dict(rank=rank,gradient_relative=gradient_error,adam_delta_relative=delta_error,capture_count=capture_count,
        timings=timings,direct_outputs=direct,peak_bytes=torch.cuda.max_memory_allocated(),
        checkpoint_gradient_diagnostic=batch_diagnostic,
        actor_hashes={name:actor_hash(value['actor']) for name,value in results.items()},
        compilation_counters={str(k):dict(v) for k,v in torch._dynamo.utils.counters.items()})
    (args.output/f'rank{rank}.json').write_text(json.dumps(local,indent=2,allow_nan=False)+'\n')
    torch.save(results,args.output/f'finite_updates_rank{rank}.pt')
    records=[None]*world;dist.all_gather_object(records,local)
    if rank==0:
        all_outputs=[x for r in records for x in r['direct_outputs']]
        gates=dict(raw_gradient=all(r['gradient_relative']<=.003 for r in records),
            adam_delta=all(r['adam_delta_relative']<=.003 for r in records),
            one_capture=all(r['capture_count']==1 for r in records),
            identical_rank_actors=all(len({r['actor_hashes'][name] for r in records})==1 for name in results),
            mutual_mean=all(max(x['reference_to_tiled_kl'],x['tiled_to_reference_kl'])<=1e-11 for x in all_outputs),
            mutual_token=all(max(x['max_reference_to_tiled_token_kl'],x['max_tiled_to_reference_token_kl'])<=1e-10 for x in all_outputs))
        summary=dict(scope='One full distributed saved-Adam update; repetitions restore same complete state; no live training',
            policy_sha256=digest,states=ids,world_size=world,sequences=args.sequences,token_count=sum(x['tokens'] for x in all_outputs),
            gates=gates,rank_records=records,expected_student_backward_tokens=2*args.repeats*len(ids)*args.sequences*1024,
            diagnostic_direct_forward_tokens=2*sum(x['tokens'] for x in all_outputs))
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(gates=gates,timings=timings)),flush=True)
        run.log({'objective':timings[-1]['maximum_rank_seconds'],'numerical_limits_met':all(gates.values())});run.finish()
    accepted=torch.tensor(int(all(gates.values())) if rank==0 else 0,device=device)
    dist.broadcast(accepted,src=0);ok=bool(accepted)
    stream.close();dist.barrier();dist.destroy_process_group()
    if not ok:raise RuntimeError('distributed qualification exceeded unchanged limits; evidence retained')


if __name__=='__main__':main()
