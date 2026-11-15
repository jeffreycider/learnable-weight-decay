"""Paired complete offline-step engineering benchmark using a fresh Chisel.

Checkpoint and synthetic token/observation inputs are fixed across backends.
This script benchmarks implementation only, not offline learning/generalization.
All steps include original GPT, edited GPT, KL, fifteen-group norm objective,
full student/controller backward and Adam. Cold compile and steady state differ.
"""
import argparse
import copy
import gc
import json
from pathlib import Path
import statistics
import sys
import time

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline.model import GPT
from rewriter.controller import GlobalRewriter
from rewriter.algebra import RawAction, Factors
from rewriter.accelerated_controller import PackedGlobalRewriter
from rewriter.accelerated_grouped_norm import AcceleratedGroupedNormObjective
from rewriter.accelerated_kl import fused_teacher_kl
from rewriter.function_preservation import FunctionalPreservation, chunked_teacher_kl
from rewriter.grouped_norm import GroupedNormObjective
from rewriter.edit_scope import editable_names, editable_weight_norm_sq, scoped_action, keep_readonly_parameters
from scripts.benchmark_chisel_kernels import synthetic_snapshot, error


def graphs():
    return int(torch._dynamo.utils.counters['stats']['unique_graphs'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--batches', default='1,4,16')
    p.add_argument('--iterations', type=int, default=8)
    p.add_argument('--backends', default='reference,packed_fused,packed_fused_compiled')
    p.add_argument('--profile', action='store_true')
    p.add_argument('--qualify-kl-gradient', action='store_true',
                   help='extra KL-only backward during first-step correctness check, outside timing')
    p.add_argument('--check-state-reuse', action='store_true')
    p.add_argument('--head-std', type=float, default=.001)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import wandb
    run = wandb.init(project='chisel-kernel-engineering', mode='offline', dir=str(args.output))
    torch.set_num_threads(8)
    torch.manual_seed(231)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.qualify_kl_gradient:
        from torch._functorch import config as functorch_config
        functorch_config.donated_buffer = False
    # Match baseline Flash attention; actor explicitly uses FP32 math attention.
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    if saved.get('controller') is not None or saved['config']['mode'] != 'baseline':
        raise ValueError('benchmark requires a baseline GPT, never learned Chisel state')
    model = GPT(**saved['config']['model']).cuda()
    model.load_state_dict(saved['model']);model.requires_grad_(False)
    snapshot = synthetic_snapshot(model)
    fresh = GlobalRewriter().cuda()
    with torch.no_grad():
        for decoder in (fresh.row_decoder, fresh.column_decoder, fresh.vector_decoder):
            decoder[-1].weight.normal_(std=args.head_std)
    initial = copy.deepcopy(fresh.state_dict())
    del fresh
    engine = FunctionalPreservation(model, microbatch_sequences=max(map(int,args.batches.split(','))),
                                    teacher_cache_device='cuda', kl_dtype=torch.float64)
    root_reference = GroupedNormObjective(engine.parameters, zero_reference_policy='freeze')
    editable = editable_names(engine.parameters, 'grouped')-root_reference.frozen_parameter_names
    radius = editable_weight_norm_sq(engine.parameters, editable).sqrt().float()*.01
    scales = {}
    with torch.no_grad():
        for name, value in engine.parameters.items():
            dimension = 8*(sum(value.shape)-8) if value.ndim==2 else value.numel()
            scales[name] = value.float().square().mean().sqrt().clamp_min(1e-3)*(value.numel()/dimension)**.5
    rows = []
    references = {}
    result = dict(scope=__doc__, hardware=torch.cuda.get_device_name(), torch=torch.__version__,
                  checkpoint_step=saved['step'], data='synthetic tokens and synthetic thin observation banks',
                  loaded_controller=False, precision='FP32 Chisel, baseline BF16 GPT casts, FP64 KL/norm', rows=rows)
    def persist():
        (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    for backend in args.backends.split(','):
        actor = GlobalRewriter().cuda();actor.load_state_dict(initial)
        module = actor if backend in ('reference','fused_only') else PackedGlobalRewriter(actor)
        capture_seconds = None
        if backend == 'packed_graph':
            from rewriter.accelerated_controller_graph import GraphedGlobalRewriter
            capture_start=time.perf_counter()
            module=GraphedGlobalRewriter(actor,snapshot)
            torch.cuda.synchronize()
            capture_seconds=time.perf_counter()-capture_start
        if backend.endswith('_compiled'):
            module = torch.compile(module, fullgraph=True, dynamic=False)
            engine.enable_compiled_forward(fullgraph=True, dynamic=False)
        else:
            engine._forward_callable = engine._functional_forward
        objective = (root_reference if backend == 'reference' else
                     AcceleratedGroupedNormObjective(engine.parameters, zero_reference_policy='freeze'))
        kl_fn = chunked_teacher_kl if backend == 'reference' else fused_teacher_kl
        for batch in map(int,args.batches.split(',')):
            actor.load_state_dict(initial)
            optimizer = torch.optim.Adam(actor.parameters(), lr=2e-4, betas=(.9,.99), eps=1e-8)
            generator = torch.Generator(device='cuda').manual_seed(952)
            inputs = torch.randint(50304, (batch,1024), device='cuda', generator=generator)
            def step(signature=False):
                optimizer.zero_grad(set_to_none=True)
                teacher, = engine.cache_teacher(inputs, data_role='synthetic')
                action = scoped_action(module(snapshot), editable)
                action = RawAction({n:Factors(f.P*scales[n], f.Q*scales[n], f.anchors)
                                    for n,f in action.matrices.items()},
                                   {n:v*scales[n] for n,v in action.vectors.items()})
                raw_norm = action.norm_sq().clamp_min(1e-24).sqrt()
                scale = torch.minimum(raw_norm.new_ones(()), radius/(raw_norm+1e-12))
                parameters = keep_readonly_parameters(engine.edited_parameters(action, scale), engine.parameters, editable)
                norm = objective.evaluate(parameters)
                logits = engine._forward(parameters, inputs)
                kl = kl_fn(logits, teacher.log_probs)
                regularizer = 100*norm.mean_relative_change + .01*norm.edit_norm_sq/radius.double().square()
                penalty = (kl/.001-1).clamp_min(0).square()
                loss = regularizer + penalty
                kl_gradient = None
                if signature and args.qualify_kl_gradient:
                    kg = torch.autograd.grad(kl, tuple(actor.parameters()), retain_graph=True)
                    kl_gradient = torch.cat([v.flatten() for v in kg]).detach().clone()
                loss.backward()
                if signature:
                    gradient = torch.cat([value.grad.flatten() for value in actor.parameters()]).detach().clone()
                    values = torch.stack((loss,kl,norm.mean_relative_change)).detach()
                torch.nn.utils.clip_grad_norm_(actor.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                if signature:
                    updated = torch.cat([value.flatten() for value in actor.parameters()]).detach().clone()
                    return values, gradient, updated, kl_gradient
            before_graphs = graphs()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            try:
                current = step(signature=True)
            except torch.cuda.OutOfMemoryError as exc:
                row=dict(backend=backend,batch=batch,tokens=batch*1024,status='out_of_memory',
                         detail=str(exc),peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9)
                rows.append(row);persist();print(json.dumps(row),flush=True)
                optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
                continue
            torch.cuda.synchronize()
            cold = time.perf_counter()-started
            if backend == 'reference': references[batch] = tuple(x.cpu() if x is not None else None for x in current)
            comparison = None
            if batch in references:
                comparison = {name:error(a.cuda(),b) for name,a,b in zip(
                    ('loss_kl_norm','controller_gradient','adam_parameters','kl_only_gradient'),references[batch],current)
                    if a is not None}
                if backend != 'reference':
                    # BF16 edited weight rounding makes tiny actor differences
                    # produce discrete logit differences. Quantify, don't demand
                    # bit identity or hide the actual discrepancy.
                    comparison['gradient_tolerance'] = 5e-3
                    if comparison['controller_gradient']['relative_l2']>5e-3:
                        raise RuntimeError('complete-step controller gradient comparison failed: '+str(comparison))
            try:
                for _ in range(2): step()
            except torch.cuda.OutOfMemoryError as exc:
                row=dict(backend=backend,batch=batch,tokens=batch*1024,status='out_of_memory',
                         phase='warmup',detail=str(exc),peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9)
                rows.append(row);persist();print(json.dumps(row),flush=True)
                optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
                continue
            torch.cuda.synchronize()
            steady_graphs = graphs()
            samples=[]
            try:
                for _ in range(args.iterations):
                    start=time.perf_counter();step();torch.cuda.synchronize()
                    samples.append((time.perf_counter()-start)*1000)
            except torch.cuda.OutOfMemoryError as exc:
                row=dict(backend=backend,batch=batch,tokens=batch*1024,status='out_of_memory',
                         phase='steady',detail=str(exc),peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9)
                rows.append(row);persist();print(json.dumps(row),flush=True)
                optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
                continue
            row=dict(backend=backend,batch=batch,tokens=batch*1024,cold_seconds=cold,
                actor_graph_capture_seconds=capture_seconds,
                cold_includes_extra_kl_validation=args.qualify_kl_gradient,
                first_loss_kl_mean_relative_norm=current[0].tolist(),
                first_kl_gradient_norm=float(current[3].norm()) if current[3] is not None else None,
                median_ms=statistics.median(samples),samples_ms=samples,
                tokens_per_second=batch*1024/(statistics.median(samples)/1000),comparison=comparison,
                graphs_before=before_graphs,graphs_after_warmup=steady_graphs,graphs_after_steady=graphs(),
                peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9)
            if (args.check_state_reuse and backend.endswith('_compiled')
                and batch == int(args.batches.split(',')[-1])):
                before_reuse=graphs()
                # Change a full checkpoint's storage/value identity, actor
                # parameters, and observation values without changing shapes.
                changed={n:v.detach().clone() for n,v in engine.parameters.items()}
                changed['proj.bias'].add_(1e-4)
                engine.load_teacher_state(changed,engine.buffers)
                objective=AcceleratedGroupedNormObjective(engine.parameters,zero_reference_policy='freeze')
                snapshot.trajectories.add_(.001)
                with torch.no_grad():
                    actor.row_decoder[-1].weight.mul_(1.00001)
                step();torch.cuda.synchronize()
                row['changed_checkpoint_reuse']=dict(graphs_before=before_reuse,graphs_after=graphs())
                if graphs()!=before_reuse:raise RuntimeError('recompile after same-shaped checkpoint/value change')
            if args.profile and batch==1:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA]) as profile:
                    step()
                path=args.output/f'{backend}-step-trace.json';profile.export_chrome_trace(str(path))
                trace=json.loads(path.read_text())
                kernels=[x for x in trace['traceEvents'] if x.get('cat')=='kernel']
                row['kernel_count']=len(kernels)
                row['sum_kernel_ms']=sum(x.get('dur',0) for x in kernels)/1000
                (args.output/f'{backend}-operators.txt').write_text(profile.key_averages().table(
                    sort_by='self_cuda_time_total',row_limit=35))
            rows.append(row);persist();print(json.dumps(row),flush=True)
            run.log(dict(backend=backend,batch=batch,objective=row['tokens_per_second'],**{k:row[k] for k in ('median_ms','tokens_per_second')}))
        del actor,module,optimizer,objective
        torch.cuda.empty_cache()
    run.finish()


if __name__=='__main__':main()
