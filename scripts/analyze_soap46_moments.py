"""CPU-only Adam denominator diagnostics on preserved SOAP checkpoints.

This reads moments without changing model/controller state or executing GPT.
Fractions are descriptive: an epsilon-dominated denominator is not an optimizer bug.
"""
import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def summarize(path):
    state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    optimizer = state['hyper']['optimizer']
    rows = []
    for group in optimizer['param_groups']:
        beta2, epsilon = group['betas'][1], group['eps']
        for key in group['params']:
            moments = optimizer['state'].get(key)
            if not moments:
                continue
            step = float(moments['step'])
            correction = 1 - beta2 ** step
            below, tenth, total, sum_value = 0, 0, 0, 0.
            max_value = 0.
            for chunk in moments['exp_avg_sq'].view(-1).split(1_000_000):
                denominator = (chunk.double() / correction).sqrt()
                below += int((denominator < epsilon).sum())
                tenth += int((denominator < epsilon * .1).sum())
                total += denominator.numel()
                sum_value += float(denominator.sum())
                max_value = max(max_value, float(denominator.max()))
            rows.append(dict(elements=total, below_epsilon=below,
                             below_tenth_epsilon=tenth, mean_sqrt_v=sum_value/total,
                             max_sqrt_v=max_value, epsilon=epsilon, step=step))
    total = sum(r['elements'] for r in rows)
    return dict(source=str(path), completed_steps=state['completed_steps'],
                seed=state['seed'], elements=total, rank_scope='one supplied shard',
                fraction_below_epsilon=sum(r['below_epsilon'] for r in rows)/total,
                fraction_below_tenth_epsilon=sum(r['below_tenth_epsilon'] for r in rows)/total,
                matrices=rows, gpt_forwards=0, gpt_backwards=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = [summarize(path) for path in args.checkpoint]
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps([{k: v for k, v in row.items() if k != 'matrices'} for row in result]))
