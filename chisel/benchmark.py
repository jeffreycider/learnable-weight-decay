"""Measure complete optimizer steps and compilation reuse for a YAML-selected module."""
import argparse
import json
from pathlib import Path
import statistics
import time
import torch
from torch._dynamo.utils import counters
from chisel.harness import build, resolve


def run(config):
    results = []
    for variant in config['variants']:
        torch._dynamo.reset()
        counters.clear()
        torch.manual_seed(config['seed'])
        model = build(config['model'] | variant.get('model', {})).cuda()
        optimizer = build(config['optimizer'], params=model.parameters())
        forward = torch.compile(model, **variant['compile']) if variant['compile'] else model
        inputs = torch.randn(*config['shape'], device='cuda', dtype=getattr(torch, config['dtype']))
        times, graph_counts = [], []
        torch.cuda.reset_peak_memory_stats()
        for step in range(config['warmup'] + config['steps']):
            torch.compiler.cudagraph_mark_step_begin()
            torch.cuda.synchronize()
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = forward(inputs).float().square().mean()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
            graph_counts.append(counters['stats']['unique_graphs'])
        warm = times[config['warmup']:]
        result = dict(name=variant['name'], median_step_ms=statistics.median(warm),
                      first_step_ms=times[0], warmup_total_ms=sum(times[:config['warmup']]),
                      graph_counts=graph_counts, peak_bytes=torch.cuda.max_memory_allocated())
        if len(set(graph_counts[config['warmup']:])) != 1:
            raise RuntimeError(f'continued compilation in measured steps: {result}')
        results.append(result)
        del optimizer, forward, model, loss
        torch.cuda.empty_cache()
    payload = dict(config=config, torch=str(torch.__version__), cuda=torch.version.cuda,
                   gpu=torch.cuda.get_device_name(), results=results)
    path = Path(config['output'])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(results, indent=2))
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    run(resolve(parser.parse_args().config))
