"""Actual actor parameter shapes, synthetic gradients; no GPT/actor forward."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from baseline.execution import enable_deterministic_execution
from rewriter.chisel_factory import build_chisel
from rewriter.gradient_batch_diagnostic import CheckpointGradientMoments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    enable_deterministic_execution()
    torch.set_num_threads(1)
    torch.manual_seed(314159)
    saved = torch.load(args.policy, map_location='cpu', weights_only=False)
    actor = build_chisel(saved['metadata']['architecture']).to('cuda:0')
    gradients = tuple(torch.randn_like(p) * 1e-4 for p in actor.parameters())
    reference = tuple(g.clone() for g in gradients)
    observer = CheckpointGradientMoments(device=gradients[0].device, state_weight=1/200)
    for _ in range(10): observer.observe(gradients)
    torch.cuda.synchronize()
    rows = []
    graph = None
    for mode in ('eager', 'graph'):
        if mode == 'graph':
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): observer.observe(gradients)
        call = (lambda: observer.observe(gradients)) if graph is None else graph.replay
        observer.statistics.zero_()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        initial = torch.cuda.memory_allocated()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall = time.perf_counter()
        start.record()
        for _ in range(100): call()
        end.record()
        end.synchronize()
        wall = time.perf_counter() - wall
        assert float(observer.statistics[0]) == 100
        rows.append(dict(mode=mode, repetitions=100,
                         event_ms_per_observation=start.elapsed_time(end)/100,
                         wall_ms_per_observation=1000*wall/100,
                         incremental_peak_bytes=torch.cuda.max_memory_allocated()-initial,
                         statistics=observer.statistics.cpu().tolist()))
    assert all(torch.equal(a, b) for a, b in zip(gradients, reference))
    result = dict(scope='Actual Chisel parameter shapes with synthetic FP32 gradients; no model forwards/backwards or updates',
                  policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),
                  tensors=len(gradients), elements=sum(g.numel() for g in gradients),
                  gradient_bytes=sum(g.numel()*g.element_size() for g in gradients),
                  shapes=[list(g.shape) for g in gradients], rows=rows,
                  gradients_unchanged=True, cuda_version=torch.version.cuda, torch_version=torch.__version__)
    (args.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    import wandb
    run = wandb.init(project='chisel', mode='offline', dir=str(args.output),
                     config={'scope': result['scope'], 'telemetry': 'component measurement logged at completion'})
    for row in rows: run.log({row['mode']+'_ms': row['event_ms_per_observation']})
    run.finish()
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
