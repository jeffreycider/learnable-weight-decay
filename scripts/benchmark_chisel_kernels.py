"""Fresh-actor synthetic kernel qualification; no learning claims or checkpoints.

Uses PR323 tensor shapes and random observation banks for timing. Outputs are
engineering diagnostics, never evidence of preservation/generalization.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline.model import GPT
from rewriter.algebra import Anchors
from rewriter.controller import GlobalRewriter, ControllerSnapshot, MatrixObservation, VectorObservation
from rewriter.accelerated_controller import PackedGlobalRewriter
from rewriter.accelerated_kl import fused_teacher_kl
from rewriter.function_preservation import chunked_teacher_kl


def synthetic_snapshot(model, rank=8):
    matrices, vectors = {}, {}
    for name, value in model.named_parameters():
        if value.ndim == 2:
            n, m = reversed(value.shape) if name == 'embed.weight' else value.shape
            anchors = Anchors(torch.linalg.qr(torch.randn(n, rank, device='cuda'))[0],
                              torch.linalg.qr(torch.randn(m, rank, device='cuda'))[0])
            matrices[name] = MatrixObservation(anchors, torch.randn(n, 5*rank+15, device='cuda'),
                torch.randn(m, 5*rank+15, device='cuda'), (torch.zeros_like(anchors.A), torch.zeros_like(anchors.B)))
        else:
            vectors[name] = VectorObservation(torch.randn(value.numel(), 18, device='cuda'),
                                              torch.zeros(value.numel(), device='cuda'))
    return ControllerSnapshot(matrices, vectors, torch.randn(64, 400, device='cuda'),
                              torch.tensor(100., device='cuda'), torch.tensor(1., device='cuda'))


def flat_action(action):
    return torch.cat([x.flatten() for factor in action.matrices.values() for x in (factor.P, factor.Q)]
                     + [x.flatten() for x in action.vectors.values()])


def measure(fn, iterations=10):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        values.append((time.perf_counter()-start)*1000)
    return dict(median_ms=statistics.median(values), minimum_ms=min(values), samples_ms=values)


def error(a, b):
    return dict(relative_l2=float((a-b).double().norm()/a.double().norm().clamp_min(1e-30)),
                max_absolute=float((a-b).abs().max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--graph', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import wandb
    run = wandb.init(project='chisel-kernel-engineering', mode='offline', dir=str(args.output))
    torch.set_num_threads(8)
    torch.manual_seed(231)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = GPT(50304, 12, 768)
    snapshot = synthetic_snapshot(model)
    del model
    actor = GlobalRewriter().cuda()
    with torch.no_grad():
        for decoder in (actor.row_decoder, actor.column_decoder, actor.vector_decoder):
            decoder[-1].weight.normal_(std=.001)
    packed = PackedGlobalRewriter(actor)
    direction = torch.randn_like(flat_action(actor(snapshot))).detach()
    def signature(module):
        actor.zero_grad(set_to_none=True)
        output = flat_action(module(snapshot))
        (output*direction).sum().backward()
        return output.detach(), torch.cat([p.grad.flatten() for p in actor.parameters()]).detach().clone()
    eager_signature, packed_signature = signature(actor), signature(packed)
    comparison = dict(action=error(eager_signature[0], packed_signature[0]),
                      gradients=error(eager_signature[1], packed_signature[1]))
    assert comparison['action']['relative_l2'] < 2e-5 and comparison['gradients']['relative_l2'] < 2e-5
    result = dict(scope='synthetic engineering timing; no GPT calls or learned controller state',
        hardware=torch.cuda.get_device_name(), torch=torch.__version__,
        comparison=comparison, actor={})
    for label, module in [('reference', actor), ('packed', packed)]:
        result['actor'][label] = measure(lambda: signature(module), args.iterations)
        print(label, result['actor'][label], flush=True)
        run.log({f'actor/{label}_ms':result['actor'][label]['median_ms'], 'objective': result['actor'][label]['median_ms']})
    if args.compile:
        compiled = torch.compile(packed, fullgraph=True, dynamic=False)
        start = time.perf_counter()
        csig = signature(compiled)
        torch.cuda.synchronize()
        result['compile_seconds'] = time.perf_counter()-start
        result['compiled_comparison'] = dict(action=error(eager_signature[0], csig[0]),
            gradients=error(eager_signature[1], csig[1]))
        result['actor']['packed_compiled'] = measure(lambda: signature(compiled), args.iterations)
        print('packed_compiled',result['actor']['packed_compiled'],flush=True)
    if args.graph:
        from rewriter.accelerated_controller_graph import GraphedGlobalRewriter
        start=time.perf_counter()
        graphed=GraphedGlobalRewriter(actor,snapshot)
        torch.cuda.synchronize()
        result['graph_capture_seconds']=time.perf_counter()-start
        gsig=signature(graphed)
        result['graphed_comparison']=dict(action=error(eager_signature[0],gsig[0]),
            gradients=error(eager_signature[1],gsig[1]))
        result['actor']['packed_graphed']=measure(lambda:signature(graphed),args.iterations)
        print('packed_graphed',result['actor']['packed_graphed'],flush=True)
    if args.profile:
        for label, module in [('reference', actor), ('packed', packed)]:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as profile:
                signature(module)
            profile.export_chrome_trace(str(args.output/f'{label}-trace.json'))
            (args.output/f'{label}-operators.txt').write_text(profile.key_averages().table(
                sort_by='self_cuda_time_total', row_limit=30))
    result['kl'] = []
    for tokens in (128, 1024, 4096):
        logits = (torch.randn(tokens, 50304, device='cuda')*3).requires_grad_()
        teacher = (logits.detach()+.001*torch.randn_like(logits)).double().log_softmax(-1)
        def kl_step(fn):
            logits.grad = None
            fn(logits, teacher).backward()
        row = dict(tokens=tokens)
        for label, fn in [('chunked', chunked_teacher_kl), ('fused', fused_teacher_kl)]:
            row[label] = measure(lambda: kl_step(fn), args.iterations)
        row['speedup'] = row['chunked']['median_ms']/row['fused']['median_ms']
        result['kl'].append(row)
        print('kl', row, flush=True)
        run.log({'kl/tokens': tokens, 'kl/speedup':row['speedup'], 'objective':row['fused']['median_ms']})
        del logits, teacher
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    run.finish()


if __name__ == '__main__': main()
