"""Isolated eager/compiled FP32 loss benchmark; no GPT or training updates.

Synthetic raw logits use the production microbatch shape by default. This tests
loss/gradient arithmetic and loss-only cost, not complete online throughput.
Run on an explicitly handed-off GPU, never alongside an active experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from baseline.execution import enable_deterministic_execution, softcap_cross_entropy
from rewriter.gpt_precision import configure_precision


def counters():
    return {str(k): dict(v) for k, v in torch._dynamo.utils.counters.items() if v}


@torch.no_grad()
def gradient_error(reference, candidate):
    # Bound temporary FP64 storage; all comparisons are outside timing.
    totals = torch.zeros(5, dtype=torch.float64, device=reference.device)
    for start in range(0, len(reference), 128):
        a = reference[start:start+128].double()
        b = candidate[start:start+128].double()
        difference = b-a
        totals[:3] += torch.stack((a.square().sum(), difference.square().sum(), (a*b).sum()))
        totals[3] = torch.maximum(totals[3], difference.abs().max())
        totals[4] += b.square().sum()
    norm_sq, error_sq, dot, maximum, candidate_sq = totals.cpu().tolist()
    candidate_norm = candidate_sq**.5
    return dict(reference_l2=norm_sq**.5, absolute_l2_error=error_sq**.5,
                relative_l2_error=(error_sq/norm_sq)**.5,
                cosine=dot/(norm_sq**.5*candidate_norm), maximum_absolute_error=maximum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokens', type=int, default=16*1024)
    parser.add_argument('--vocab', type=int, default=50304)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.tokens < 1 or args.vocab < 2 or args.repeats < 2:
        parser.error('positive tokens, vocab >=2 and repeats >=2 required')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    enable_deterministic_execution()
    configure_precision('fp32')
    generator = torch.Generator(device='cuda').manual_seed(701001)
    logits = (8*torch.randn(args.tokens, args.vocab, device='cuda', dtype=torch.float32,
                           generator=generator)).requires_grad_()
    targets = torch.randint(args.vocab, (args.tokens,), device='cuda', generator=generator)
    compiled = torch.compile(softcap_cross_entropy, fullgraph=True, dynamic=False)
    functions = dict(eager=softcap_cross_entropy, compiled=compiled)

    def evaluate(name):
        loss = functions[name](logits, targets)
        gradient, = torch.autograd.grad(loss, logits)
        return loss.detach(), gradient.detach()

    cold = {}
    outputs = {}
    for name in functions:
        torch.cuda.synchronize()
        began = time.perf_counter()
        outputs[name] = evaluate(name)
        torch.cuda.synchronize()
        cold[name] = time.perf_counter()-began
    a, b = outputs['eager'], outputs['compiled']
    errors = gradient_error(a[1], b[1])
    errors.update(eager_mean_loss=float(a[0])/args.tokens,
                  compiled_mean_loss=float(b[0])/args.tokens,
                  absolute_mean_loss_difference=float((b[0]-a[0]).abs())/args.tokens)
    del outputs, a, b
    for name in functions:
        for _ in range(2):
            result = evaluate(name)
            del result
    torch.cuda.synchronize()
    before = counters()
    rows = []
    for repeat in range(args.repeats):
        order = ('eager', 'compiled') if repeat % 2 == 0 else ('compiled', 'eager')
        for name in order:
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            began = time.perf_counter()
            start.record()
            result = evaluate(name)
            end.record()
            torch.cuda.synchronize()
            rows.append(dict(repeat=repeat, implementation=name,
                             wall_seconds=time.perf_counter()-began,
                             cuda_seconds=start.elapsed_time(end)/1000))
            del result
    after = counters()
    report = dict(scope='synthetic raw logits; FP32 softcap plus SUM cross-entropy and its logit gradient only; no GPT evaluation',
                  shape=[args.tokens, args.vocab], logit_standard_deviation=8,
                  seed=701001, torch_version=torch.__version__, gpu=torch.cuda.get_device_name(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  loss_source_sha256=hashlib.sha256(Path(__file__).resolve().parents[1].joinpath('baseline/execution.py').read_bytes()).hexdigest(),
                  cold_wall_seconds=cold, arithmetic=errors, rows=rows,
                  median_wall_seconds={name:statistics.median(r['wall_seconds'] for r in rows if r['implementation']==name)
                                       for name in functions},
                  compile_counters_before=before, compile_counters_after=after,
                  no_warm_recompilation=before==after,
                  limits='Component evidence only. An actual matched online-step comparison is required before a throughput claim or enabling this option in training.')
    (args.output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    if before != after:
        raise RuntimeError('compilation counters changed during warm repetitions')
    print(json.dumps({k:report[k] for k in ('cold_wall_seconds','arithmetic','median_wall_seconds','no_warm_recompilation')}), flush=True)


if __name__ == '__main__':
    main()
