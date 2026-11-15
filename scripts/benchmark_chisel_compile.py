"""Compare eager/compiled Chisel with identical data and explicit graph-reuse checks.

Uses the historical cached diagnostic objective to reproduce the 454ms profile;
no GPT calls, no trained checkpoint writes, no claim about full online runtime.
"""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys
import time

import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.profile_controller import _load,_hash,profile,signatures,compare_signatures
from rewriter.edit_scope import editable_names


def clone_snapshot(value):
    if isinstance(value,torch.Tensor):return value.detach().clone()
    if dataclasses.is_dataclass(value):
        return dataclasses.replace(value,**{f.name:clone_snapshot(getattr(value,f.name)) for f in dataclasses.fields(value)})
    if isinstance(value,dict):return {k:clone_snapshot(v) for k,v in value.items()}
    if isinstance(value,tuple):return tuple(clone_snapshot(v) for v in value)
    return value


def counters():
    return {str(k):dict(v) for k,v in torch._dynamo.utils.counters.items() if v}


def graphs():
    return int(torch._dynamo.utils.counters['stats']['unique_graphs'])


def numerical_check(eager,compiled):
    result=compare_signatures(eager,compiled)
    # Retain the original elementwise diagnostic, including its failures near
    # zero. Qualify FP32 equivalence by full-vector error rather than asking
    # small gradient components to meet a disproportionately tight relative bound.
    result['relative_l2_limit']=1e-5
    result['loss_relative_error']=abs(eager['loss']-compiled['loss'])/max(1.,abs(eager['loss']))
    result['normwise_pass']=all(result[k]['relative_l2_error']<=1e-5 for k in ('action','gradient')) and result['loss_relative_error']<=1e-5
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--observation',type=Path,required=True)
    p.add_argument('--actor-checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=30)
    args=p.parse_args()
    args.measurements=None;args.synthetic_measurements=True;args.device='cuda:0'
    args.output.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        (args.output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    actor,snapshot,measurements,label,scope=_load(args)
    editable=editable_names(dict(snapshot.weights,**snapshot.vector_weights),scope)
    save('environment.json',dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        cpu_threads=torch.get_num_threads(),float32_matmul_precision=torch.get_float32_matmul_precision(),allow_tf32=torch.backends.cuda.matmul.allow_tf32,scope=scope,objective=label,compile_mode='default',fullgraph=True,
        observation_sha256=_hash(args.observation),actor_checkpoint_sha256=_hash(args.actor_checkpoint),
        script_sha256=_hash(__file__),timing_scope='Cached actor forward, cached diagnostic loss, actor backward; no Adam or GPT'))
    eager=profile(actor,snapshot,measurements,warmup=3,iterations=args.iterations,editable=editable)
    save('eager.json',eager);print('eager_complete',eager['timing_ms']['total_wall_ms'],flush=True)
    reference=signatures(actor,snapshot,measurements,editable=editable)
    torch._dynamo.reset();torch._dynamo.utils.counters.clear()
    compiled=torch.compile(actor,fullgraph=True,dynamic=False)
    start=time.perf_counter()
    first=signatures(compiled,snapshot,measurements,editable=editable)
    torch.cuda.synchronize();cold=time.perf_counter()-start
    comparison=numerical_check(reference,first)
    save('cold_compile.json',dict(first_forward_backward_seconds=cold,comparison=comparison,counters=counters()))
    print('compile_complete',cold,comparison['within_tolerance'],flush=True)
    if not comparison['normwise_pass']:raise RuntimeError('compiled output/gradient comparison failed')
    before=graphs()
    steady=profile(compiled,snapshot,measurements,warmup=3,iterations=args.iterations,editable=editable)
    save('compiled.json',steady)
    save('after_steady_counters.json',counters())
    if graphs()!=before:raise RuntimeError('recompiled during fixed-input steady-state benchmark')
    reuse=[]
    # In-place parameter updates have the same version semantics as Adam. Fresh
    # snapshots also replace tensor storage; both must reuse the compiled graph.
    for i in range(5):
        fresh=clone_snapshot(snapshot)
        for obs in fresh.matrices.values():
            obs.rows.mul_(1+(i+1)*1e-4);obs.columns.mul_(1-(i+1)*1e-4)
        fresh.trajectories.mul_(1+(i+1)*1e-4)
        with torch.no_grad():
            for parameter in actor.parameters():parameter.mul_(1-1e-5)
        expected=signatures(actor,fresh,measurements,editable=editable)
        actual=signatures(compiled,fresh,measurements,editable=editable)
        check=numerical_check(expected,actual)
        reuse.append(dict(index=i,unique_graphs=graphs(),comparison=check))
        save('reuse_checks.json',dict(checks=reuse,counters=counters()))
        if graphs()!=before:raise RuntimeError('recompiled for new snapshot or updated Chisel parameters')
        if not check['normwise_pass']:raise RuntimeError('changed-input correctness failed')
    alternating=[]
    for mode,model in [('eager',actor),('compiled',compiled),('eager',actor),('compiled',compiled)]:
        block=profile(model,snapshot,measurements,warmup=2,iterations=20,editable=editable)
        alternating.append(dict(mode=mode,profile=block))
    save('alternating.json',alternating)
    if graphs()!=before:raise RuntimeError('recompiled in alternating timing blocks')
    result=dict(status='ok',alternating=alternating,cold_forward_backward_seconds=cold,eager=eager,compiled=steady,
        speedup=eager['timing_ms']['total_wall_ms']['median']/steady['timing_ms']['total_wall_ms']['median'],
        initial_comparison=comparison,reuse_checks=reuse,initial_unique_graphs=before,final_unique_graphs=graphs(),
        counters=counters(),gpt_forwards=0,gpt_backwards=0,
        scope='same historical hidden-scope cached diagnostic as earlier profile; not full offline or online GPT timing')
    save('result.json',result);print(json.dumps({k:result[k] for k in ('status','cold_forward_backward_seconds','speedup','initial_unique_graphs','final_unique_graphs')}),flush=True)


if __name__=='__main__':main()
