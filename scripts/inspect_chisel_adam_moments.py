"""Read-only Adam conditioning evidence from an offline Chisel checkpoint.

Moments describe post-clipping gradients. Same-moment epsilon comparisons are
algebraic diagnostics, not predictions of another training trajectory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def inspect(optimizer):
    results = []
    for index, group in enumerate(optimizer['param_groups']):
        values = []
        steps = set()
        update_squared = {str(group['eps']): 0., str(1e-12): 0.}
        missing = 0
        for pid in group['params']:
            moment = optimizer['state'].get(pid)
            if moment is None:
                missing += 1
                continue
            step = int(moment['step'])
            assert step > 0
            steps.add(step)
            b1, b2 = group['betas']
            rms = (moment['exp_avg_sq'].double() / (1 - b2**step)).sqrt()
            mean = moment['exp_avg'].double() / (1 - b1**step)
            assert torch.isfinite(rms).all() and torch.isfinite(mean).all()
            values.append(rms.flatten())
            for epsilon in update_squared:
                update_squared[epsilon] += float((group['lr'] * mean / (rms + float(epsilon))).square().sum())
        if not values:
            results.append(dict(group=index, role=group.get('role'), missing_state_parameters=missing))
            continue
        rms = torch.cat(values)
        active = rms > 0
        quantiles = torch.quantile(rms, torch.tensor([0., .5, .9, .99, 1.], dtype=torch.float64)).tolist()
        results.append(dict(
            group=index, role=group.get('role'), lr=group['lr'], epsilon=group['eps'],
            moment_steps=sorted(steps), coordinates=rms.numel(), missing_state_parameters=missing,
            fraction_zero_second_moment=float((~active).double().mean()),
            fraction_below_epsilon=float((rms < group['eps']).double().mean()),
            fraction_active_below_epsilon=float((rms[active] < group['eps']).double().mean()) if active.any() else None,
            bias_corrected_rms_quantiles=dict(zip(('min', 'median', 'p90', 'p99', 'max'), quantiles)),
            same_moments_nominal_adam_update_l2={eps: squared**.5 for eps, squared in update_squared.items()},
        ))
    return results


def shared_parameter_evidence(saved, initial):
    from rewriter.grouped_coordinate_chisel import GroupedCoordinateChisel
    config = saved['actor']['_extra_state']
    if initial['completed_updates'] != 0 or initial['actor']['_extra_state'] != config:
        raise ValueError('matching initial shared architecture required')
    keys = ('rank', 'trajectory_dim', 'model_dim', 'num_layers', 'latent_tokens',
            'fresh_weight_core', 'factor_precision', 'residual_precision')
    with torch.random.fork_rng(devices=[]):
        actor = GroupedCoordinateChisel(**{key: config[key] for key in keys})
    if actor.configuration() != config:
        raise ValueError('unsupported shared architecture')
    names = [name for name, _ in actor.named_parameters()]
    result = []
    for group in saved['shared_optimizer']['param_groups']:
        if group['role'] not in ('heads', 'body'):
            raise ValueError('unknown diagnostic optimizer group role')
        selected = [name for name in names if name.startswith(('matrix_output.', 'vector_output.')) == (group['role'] == 'heads')]
        if len(selected) != len(group['params']):
            raise ValueError('optimizer parameter ordering differs')
        for name, pid in zip(selected, group['params']):
            moment = saved['shared_optimizer']['state'].get(pid)
            if moment is None:
                continue
            rms = (moment['exp_avg_sq'].double() / (1-group['betas'][1]**int(moment['step']))).sqrt()
            original = initial['actor'][name].double()
            change = saved['actor'][name].double()-original
            result.append(dict(name=name, role=group['role'], coordinates=change.numel(),
                initial_l2=float(original.norm()), motion_l2=float(change.norm()),
                changed_coordinates=int(torch.count_nonzero(change)),
                rms_median=float(rms.median()), rms_max=float(rms.max()),
                fraction_below_epsilon=float((rms < group['eps']).double().mean())))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--initial', type=Path, help='Optional update-zero checkpoint for per-parameter shared motion')
    args = parser.parse_args()
    torch.set_num_threads(1)
    saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    digest = hashlib.sha256()
    with args.checkpoint.open('rb') as source:
        for chunk in iter(lambda: source.read(8 << 20), b''):
            digest.update(chunk)
    result = dict(
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=digest.hexdigest(),
        completed_updates=saved['completed_updates'],
        scope='Stored post-clipping Adam moments only; no GPT calls or optimizer updates.',
        limitation='Nominal FP64 Adam formula omits parameter rounding; changing epsilon would also change future moments.',
        shared=inspect(saved['shared_optimizer']),
        private={str(key): inspect(value) for key, value in saved['private_optimizers'].items()},
    )
    if args.initial:
        initial = torch.load(args.initial, map_location='cpu', mmap=True, weights_only=False)
        result['initial_checkpoint'] = str(args.initial.resolve())
        result['initial_sha256'] = hashlib.sha256(args.initial.read_bytes()).hexdigest()
        result['shared_parameters'] = shared_parameter_evidence(saved, initial)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
