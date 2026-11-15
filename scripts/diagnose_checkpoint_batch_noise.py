"""Measure existing per-checkpoint gradients at one frozen Chisel policy.

No optimizer updates or new training recipe. This standalone diagnostic costs
one ordinary offline gradient evaluation per selected checkpoint; observing its
moments needs no additional GPT execution. Selection uses its own Python RNG.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

def actor_hash(state):
    digest=hashlib.sha256()
    for name,value in state.items():
        digest.update(name.encode());digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    from baseline.execution import enable_deterministic_execution
    from baseline.model import GPT
    from rewriter.chisel_factory import build_chisel
    from rewriter.state_bank import StateBankReader
    from rewriter.teacher_cache import TeacherCacheReader
    from rewriter.offline_systems import CachedBaseNormPreservation, VerifiedStateConstants, select_panel_blocks
    from rewriter.accelerated_grouped_norm import AcceleratedGroupedNormObjective
    from rewriter.tokenwise_reference import TokenwiseConfig, TokenwiseDual
    from rewriter.offline_tokenwise_graph import TokenwiseGraphAccumulator
    from rewriter.gradient_batch_diagnostic import CheckpointGradientMoments, summarize_checkpoint_gradients

    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('state-bank','teacher-cache','policy','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--policy-sha256',required=True)
    parser.add_argument('--checkpoints',type=int,default=200)
    parser.add_argument('--selection-seed',type=int,default=20261110)
    args=parser.parse_args()
    if args.checkpoints<2:raise ValueError('at least two checkpoints required')
    enable_deterministic_execution();torch.set_num_threads(1)
    device=torch.device('cuda',0);torch.cuda.set_device(device)
    digest=hashlib.sha256(args.policy.read_bytes()).hexdigest()
    if digest!=args.policy_sha256:raise ValueError('policy SHA mismatch')
    saved=torch.load(args.policy,map_location='cpu',weights_only=False)
    reader=StateBankReader(args.state_bank,verify_once=True,mmap_model=True)
    cache=TeacherCacheReader(args.teacher_cache,verify_once=True,mmap=True)
    ids=sorted(random.Random(args.selection_seed).sample(sorted(reader.manifest['split']['train']),args.checkpoints))
    args.output.mkdir(parents=True,exist_ok=False)
    log_file=(args.output/'progress.jsonl').open('w')
    def log(event,**values):
        row=dict(event=event,**values);line=json.dumps(row,allow_nan=False)
        log_file.write(line+'\n');log_file.flush();print(line,flush=True)
    import wandb
    run=wandb.init(project='chisel',mode='offline',dir=str(args.output),config=vars(args))
    engine=CachedBaseNormPreservation(GPT(**reader.manifest['provenance']['model_config']).to(device).requires_grad_(False),
        microbatch_sequences=16,teacher_cache_device=device,kl_dtype=torch.float64,gpt_precision='fp32')
    constants=VerifiedStateConstants(engine,AcceleratedGroupedNormObjective)
    actor=build_chisel(saved['metadata']['architecture']).to(device)
    actor.load_state_dict(saved['controller']);before=actor_hash(saved['controller'])
    forward=torch.compile(actor,fullgraph=True,dynamic=False)
    cfg=TokenwiseConfig(**saved['metadata']['tokenwise_objective'])
    if cfg.loss_backend!='tiled_fp64_v1':raise ValueError('pinned tiled loss required')
    ds=saved['primal_dual'];dual=TokenwiseDual(ds['multipliers'],config=cfg,dual_lr=ds['dual_lr'])
    dual.load_state_dict(ds)
    options=saved['metadata']['options']
    adapter=TokenwiseGraphAccumulator(actor,forward,engine,cache,constants,
        radius_fraction=options['radius_fraction'],shrinkage_coefficient=options['shrinkage_coefficient'],capture_warmups=1)
    observer=CheckpointGradientMoments(device=device,state_weight=1/len(ids))
    adapter.begin_update(ids,global_checkpoint_count=len(ids))
    log('start',policy_sha256=digest,state_ids=ids,state_weight=1/len(ids),sequences_per_checkpoint=64,
        selection_seed=args.selection_seed,optimizer_updates=0)
    started=time.perf_counter();norms=[]
    for index,state in enumerate(ids):
        record=reader.read(state,model_device='cpu',device=device)
        panel=select_panel_blocks(cache.read(state,'train'),range(4))
        measured,_=adapter.accumulate(record,panel,dual)
        squared=observer.observe(adapter.graph.gradient_buffers);norms.append(squared)
        log('checkpoint',index=index,state=state,sequence_ids=panel.sequence_ids,
            gradient_norm=float(squared.sqrt()),mean_kl=measured['mean_kl'],elapsed_seconds=time.perf_counter()-started)
        del record,panel
    gradients=adapter.finish_update()
    result=summarize_checkpoint_gradients(observer.statistics,gradients,
        batch_sizes=sorted({n for n in (8,16,32,64,100,len(ids)) if n<=len(ids)}))
    gradient=torch.cat([g.flatten() for g in gradients]).detach().cpu()
    after=actor_hash({n:p.detach().cpu() for n,p in actor.state_dict().items()})
    if before!=after or dual.state_dict()!=ds:raise RuntimeError('frozen actor or dual state mutated')
    result.update(policy_sha256=digest,state_ids=ids,selection_seed=args.selection_seed,
        state_manifest_sha256=hashlib.sha256((args.state_bank/'manifest.json').read_bytes()).hexdigest(),
        teacher_manifest_sha256=hashlib.sha256((args.teacher_cache/'manifest.json').read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        checkpoint_squared_gradient_norms=torch.stack(norms).cpu().tolist(),
        elapsed_seconds=time.perf_counter()-started,actor_unchanged=True,dual_unchanged=True,
        optimizer_updates=0,compute=dict(adapter.compute),capture_count=adapter.capture_count,
        peak_cuda_bytes=torch.cuda.max_memory_allocated())
    torch.save(gradient,args.output/'panel_mean_gradient.pt')
    (args.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    log('summary',**result);run.log({'objective':result['panel_mean_gradient_norm']});run.finish()
    adapter.close();log_file.close()


if __name__=='__main__':main()
