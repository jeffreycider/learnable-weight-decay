"""Bounded cached-only controller timing; never imports or evaluates GPT.

Use --observation completed_run/observation.pt [--actor-checkpoint controller.pt].
Without measured banks, the objective is norm+trust only. Explicitly selecting
--synthetic-measurements adds four fake banks for cost measurement, not learning
or preservation evidence. No optimizer step or checkpoint mutation is performed.
--compile runs a separate, killable process (300 second default limit), comparing
actions and all controller gradients against eager arithmetic before timing.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rewriter.algebra import Measurements, auxiliary_objective
from rewriter.controller import GlobalRewriter
from rewriter.edit_scope import editable_names,scoped_action


def move_tree(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device)
    if dataclasses.is_dataclass(value):
        return dataclasses.replace(value, **{f.name:move_tree(getattr(value,f.name),device) for f in dataclasses.fields(value)})
    if isinstance(value, dict):
        return {k:move_tree(v,device) for k,v in value.items()}
    if isinstance(value, tuple):
        return tuple(move_tree(v,device) for v in value)
    return value


def synthetic_banks(snapshot):
    """Four independent random banks, solely for timing their contraction cost."""
    generator = torch.Generator().manual_seed(912003)
    def bank(value):
        return torch.randn((4,*value.shape),generator=generator) * 1e-4
    return Measurements({n:tuple(bank(x) for x in obs.weight_probes) for n,obs in snapshot.matrices.items()},
                        {n:bank(obs.weight) for n,obs in snapshot.vectors.items()})


def validate_measurements(snapshot, measurements):
    if set(measurements.matrices)!=set(snapshot.matrices) or set(measurements.vectors)!=set(snapshot.vectors):
        raise ValueError('measurement names do not match snapshot')
    for name,obs in snapshot.matrices.items():
        if len(measurements.matrices[name])!=2:
            raise ValueError('each matrix needs both projected measurement banks')
        for value,expected in zip(measurements.matrices[name],obs.weight_probes):
            if value.shape!=(4,*expected.shape):
                raise ValueError('expected exactly four training probes; audit banks are not accepted')
    for name,obs in snapshot.vectors.items():
        if measurements.vectors[name].shape!=(4,*obs.weight.shape):
            raise ValueError('expected exactly four vector training probes')


def cached_loss(action, snapshot, measurements):
    if measurements is not None:
        return auxiliary_objective(action,snapshot.weights,snapshot.vector_weights,measurements,
            weight_norm=snapshot.weight_norm,radius=snapshot.radius,loss_budget=.001,beta=1.,gamma=.01)
    radius = snapshot.radius
    return (action.norm_change(snapshot.weights,snapshot.vector_weights)/
            (2*snapshot.weight_norm*radius+radius.square()+1e-24)
            + .01*action.norm_sq()/(radius.square()+1e-24))


def signatures(actor, snapshot, measurements, *, editable=None):
    actor.zero_grad(set_to_none=True)
    action = actor(snapshot)
    if editable is not None:action=scoped_action(action,editable)
    loss = cached_loss(action,snapshot,measurements)
    loss.backward()
    output = torch.cat([x.detach().flatten() for f in action.matrices.values() for x in (f.P,f.Q)] +
                       [v.detach().flatten() for v in action.vectors.values()]).cpu()
    gradient = torch.cat([p.grad.detach().flatten() if p.grad is not None else torch.zeros_like(p).flatten()
                          for p in actor.parameters()]).cpu()
    return dict(action=output,gradient=gradient,loss=float(loss.detach()))


def compare_signatures(eager, compiled, *, rtol=2e-3, atol=2e-6):
    result = dict(rtol=rtol,atol=atol,within_tolerance=True)
    for key in ('action','gradient'):
        left,right = eager[key],compiled[key]
        if left.shape!=right.shape:
            raise ValueError('compiled action/gradient shape differs')
        good = bool(torch.allclose(left,right,rtol=rtol,atol=atol))
        result[key] = dict(max_absolute_error=float((left-right).abs().max()),
            relative_l2_error=float((left-right).double().norm()/left.double().norm().clamp_min(1e-30)),
            eager_l2_norm=float(left.double().norm()),compiled_l2_norm=float(right.double().norm()),
            within_tolerance=good)
        result['within_tolerance'] &= good
    result['loss_absolute_error'] = abs(eager['loss']-compiled['loss'])
    return result


def profile(actor, snapshot, measurements, *, warmup=3, iterations=10, editable=None):
    if not 0<=warmup<=20 or not 1<=iterations<=100:
        raise ValueError('bounded diagnostic requires warmup 0..20 and iterations 1..100')
    cuda = next(actor.parameters()).device.type=='cuda'
    rows = []
    def sync():
        if cuda:torch.cuda.synchronize()
    for iteration in range(warmup+iterations):
        actor.zero_grad(set_to_none=True)
        sync()
        if iteration==warmup and cuda:
            torch.cuda.reset_peak_memory_stats()
            resident = torch.cuda.memory_allocated()
        events = [torch.cuda.Event(enable_timing=True) for _ in range(4)] if cuda else None
        start = time.perf_counter()
        if events:events[0].record()
        action = actor(snapshot)
        after_actor = time.perf_counter()
        if events:events[1].record()
        if editable is not None:action=scoped_action(action,editable)
        loss = cached_loss(action,snapshot,measurements)
        after_loss = time.perf_counter()
        if events:events[2].record()
        loss.backward()
        if events:events[3].record()
        sync()
        done = time.perf_counter()
        if not torch.isfinite(loss):raise ValueError('nonfinite cached diagnostic objective')
        if iteration>=warmup:
            row=dict(total_wall_ms=1000*(done-start),actor_dispatch_ms=1000*(after_actor-start),
                     objective_dispatch_ms=1000*(after_loss-after_actor))
            if events:
                row.update(actor_cuda_ms=events[0].elapsed_time(events[1]),
                           objective_cuda_ms=events[1].elapsed_time(events[2]),
                           backward_cuda_ms=events[2].elapsed_time(events[3]))
            rows.append(row)
        del action,loss
    summary = {key:dict(mean=statistics.mean(row[key] for row in rows),
                        median=statistics.median(row[key] for row in rows),
                        minimum=min(row[key] for row in rows),maximum=max(row[key] for row in rows)) for key in rows[0]}
    grad_norm = sum(p.grad.detach().double().square().sum() for p in actor.parameters() if p.grad is not None).sqrt()
    return dict(iterations=iterations,warmup=warmup,timing_ms=summary,measurements=rows,
        controller_gradient_norm=float(grad_norm),controller_parameters=sum(p.numel() for p in actor.parameters()),
        peak_allocated_bytes=torch.cuda.max_memory_allocated() if cuda else None,
        resident_allocated_bytes=resident if cuda else None,
        incremental_peak_bytes=torch.cuda.max_memory_allocated()-resident if cuda else None,
        k8_estimated_cached_wall_seconds=8*summary['total_wall_ms']['mean']/1000,
        optimizer_steps=0,gpt_forwards=0,gpt_backwards=0,
        timing_scope='actor + cached objective + controller backward; excludes Adam, clipping, GPT, '
                     'snapshot collection, anchor refresh, dense commits, and delayed-credit bookkeeping')


def _load(args):
    record = torch.load(args.observation,map_location='cpu',weights_only=False)
    snapshot = record['snapshot']
    scope=record.get('calibration',{}).get('editable_scope','all')
    if scope=='grouped':
        raise NotImplementedError('grouped-L2 cached profiling needs per-group reference statistics; this profiler supports historical all/hidden objectives only')
    if args.measurements:
        saved = torch.load(args.measurements,map_location='cpu',weights_only=False,mmap=True)
        if isinstance(saved,Measurements):
            measurements=saved
        elif 'measurements' in saved:
            measurements=saved['measurements']
        else:
            controller=saved.get('controller',saved)
            diagnostic=controller.get('diagnostic_training')
            if diagnostic is None:raise ValueError('checkpoint has no exported training measurements')
            measurements=diagnostic['measurements']
        label='saved_four_training_probes'
        del saved
    elif args.synthetic_measurements:
        measurements=synthetic_banks(snapshot);label='synthetic_four_probe_banks_for_performance_only'
    else:
        measurements=None;label='norm_and_trust_only_no_preservation_banks'
    if measurements is not None:validate_measurements(snapshot,measurements)
    rank=next(iter(snapshot.matrices.values())).anchors.A.shape[1]
    trajectory_dim=snapshot.trajectories[0].numel()
    with torch.random.fork_rng():
        torch.manual_seed(200001)
        actor=GlobalRewriter(rank=rank,trajectory_dim=trajectory_dim)
    if args.actor_checkpoint:
        state=torch.load(args.actor_checkpoint,map_location='cpu',weights_only=False,mmap=True)
        if state.get('config',{}).get('editable_scope','all')!=scope:
            raise ValueError('actor and observation editable scopes differ')
        actor.load_state_dict(state['controller'])
        del state
    device=torch.device(args.device)
    return actor.to(device),move_tree(snapshot,device),move_tree(measurements,device),label,scope


def _hash(path):
    if path is None:return None
    digest=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):digest.update(chunk)
    return digest.hexdigest()


def run(args):
    started=time.perf_counter()
    actor,snapshot,measurements,label,scope=_load(args)
    editable=editable_names(dict(snapshot.weights,**snapshot.vector_weights),scope)
    comparison=None
    if args.compile_worker:
        reference=signatures(actor,snapshot,measurements,editable=editable)
        actor=torch.compile(actor,dynamic=False)
        comparison=compare_signatures(reference,signatures(actor,snapshot,measurements,editable=editable))
        if not comparison['within_tolerance']:
            return dict(status='compiled_comparison_failed',comparison=comparison,gpt_forwards=0,gpt_backwards=0)
    result=profile(actor,snapshot,measurements,warmup=args.warmup,iterations=args.iterations,editable=editable)
    result.update(status='ok',mode='compiled' if args.compile_worker else 'eager',comparison=comparison,
        objective_banks=label,actor_initialization='loaded_for_diagnostic_only' if args.actor_checkpoint else 'fresh_zero_output_heads',
        editable_scope=scope,
        observation_sha256=_hash(args.observation),actor_checkpoint_sha256=_hash(args.actor_checkpoint),
        measurement_sha256=_hash(args.measurements),script_sha256=_hash(__file__),
        total_process_seconds=time.perf_counter()-started,torch_version=torch.__version__,device=args.device)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observation',type=Path,required=True)
    parser.add_argument('--actor-checkpoint',type=Path)
    bank=parser.add_mutually_exclusive_group()
    bank.add_argument('--measurements',type=Path)
    bank.add_argument('--synthetic-measurements',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--warmup',type=int,default=3)
    parser.add_argument('--iterations',type=int,default=10)
    parser.add_argument('--compile',action='store_true')
    parser.add_argument('--compile-timeout',type=float,default=300.)
    parser.add_argument('--compile-worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.output.exists():raise FileExistsError(args.output)
    if not 0<args.compile_timeout<=600:parser.error('compile timeout must be in (0,600] seconds')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result=run(args)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if args.compile and not args.compile_worker:
        # Release eager tensors before the second process claims GPU memory.
        if torch.cuda.is_available():torch.cuda.empty_cache()
        child_output=args.output.with_suffix('.compiled.json')
        command=[sys.executable,str(Path(__file__).resolve()),'--observation',str(args.observation),
                 '--output',str(child_output),'--device',args.device,'--warmup',str(args.warmup),
                 '--iterations',str(args.iterations),'--compile-worker']
        for flag,value in (('--actor-checkpoint',args.actor_checkpoint),('--measurements',args.measurements)):
            if value is not None:command.extend([flag,str(value)])
        if args.synthetic_measurements:command.append('--synthetic-measurements')
        logpath=args.output.with_suffix('.compiled.log')
        with logpath.open('x') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                code=child.wait(timeout=args.compile_timeout)
                result['compiled_process']=dict(status='finished' if code==0 else 'failed',returncode=code,
                    output=str(child_output),log=str(logpath),timeout_seconds=args.compile_timeout)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait()
                result['compiled_process']=dict(status='timeout_killed',log=str(logpath),timeout_seconds=args.compile_timeout)
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='measurements'},allow_nan=False),flush=True)


if __name__=='__main__':main()
