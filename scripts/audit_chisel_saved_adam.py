"""Estimate gradient/step alignment from saved Adam states; no model evaluations.

This diagnostic applies only to the fixed two-checkpoint CPU fits. It recovers
clipped gradients from consecutive first moments, then restores each role's
logged preclip norm. Private gradients were doubled before clipping, so their
contribution to the mean objective must be halved again. Finite precision makes
the reconstruction approximate. No checkpoint or optimizer state is modified.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

import torch


def recover_gradient(previous, current, beta):
    """Recover the latest clipped gradient and a conservative roundoff scale.

The error scale covers ordinary first-moment arithmetic, not arbitrary corrupt
checkpoints or nonstandard Adam implementations. It is a numerical diagnostic,
not an interval-arithmetic certificate.
"""
    if not 0 <= beta < 1 or previous.dtype != current.dtype:
        raise ValueError('matching moment dtypes and 0 <= beta < 1 required')
    old, new = previous.double(), current.double()
    gradient = (new - beta * old) / (1 - beta)
    error = 16 * torch.finfo(current.dtype).eps * (
        old.abs() + new.abs() + (1 - beta) * gradient.abs()
    ) / (1 - beta)
    return gradient, error


def summarize_role(pairs, preclip_norm, objective_weight=1.):
    totals = dict(g2=0., error2=0., actual2=0., ideal2=0., actual_dot=0., ideal_dot=0.)
    for old_value, new_value, old_moments, new_moments, group in pairs:
        if group.get('weight_decay', 0) or group.get('amsgrad', False) or group.get('maximize', False):
            raise ValueError('requires zero-decay ordinary Adam moments')
        step = int(new_moments['step'])
        if old_moments and int(old_moments['step']) != step - 1:
            raise ValueError('nonconsecutive optimizer moments')
        if not old_moments and step != 1:
            raise ValueError('missing previous moments')
        new_m = new_moments['exp_avg']
        old_m = old_moments['exp_avg'] if old_moments else torch.zeros_like(new_m)
        g, error = recover_gradient(old_m, new_m, group['betas'][0])
        actual = new_value.double() - old_value.double()
        beta1, beta2 = group['betas']
        m = new_m.double() / (1 - beta1 ** step)
        v = (new_moments['exp_avg_sq'].double() / (1 - beta2 ** step)).sqrt()
        ideal = -group['lr'] * m / (v + group['eps'])
        for key, value in [('g2', g.square().sum()), ('error2', error.square().sum()),
                           ('actual2', actual.square().sum()), ('ideal2', ideal.square().sum()),
                           ('actual_dot', (g * actual).sum()), ('ideal_dot', (g * ideal).sum())]:
            totals[key] += float(value)
    gn = math.sqrt(totals['g2'])
    if not gn or not math.isfinite(gn):
        raise ValueError('zero/nonfinite reconstructed gradient')
    raw_norm = preclip_norm * objective_weight
    # Normalize the recovered direction to the independently logged raw norm.
    scale = raw_norm / gn
    gradient_error = 2 * raw_norm * math.sqrt(totals['error2']) / gn
    result = dict(recovered_clipped_gradient_l2=gn, raw_objective_gradient_l2=raw_norm,
                  gradient_roundoff_l2_estimate=gradient_error)
    for name in ('actual', 'ideal'):
        motion = math.sqrt(totals[name + '2'])
        dot = scale * totals[name + '_dot']
        result[name] = dict(step_l2=motion, gradient_dot_step=dot,
                            cosine=dot / (raw_norm * motion) if motion else None,
                            dot_roundoff_estimate=gradient_error * motion)
    return result


def read_metrics(paths):
    records = {}
    for path in paths:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            records[row['completed_updates_before_measure']] = row
    return records


def parameter_pairs(old_values, new_values, old_opt, new_opt, names, group, step):
    pairs = []
    for name, index in zip(names, group['params']):
        moments = new_opt['state'].get(index)
        if moments is None:
            # Some architecture parameters do not participate in this action
            # path. Adam leaves parameters with grad=None completely untouched.
            assert index not in old_opt['state']
            assert torch.equal(old_values[name], new_values[name])
            continue
        assert int(moments['step']) == step
        pairs.append((old_values[name], new_values[name], old_opt['state'].get(index), moments, group))
    return pairs


def measure_pair(before, after, record, next_record, names):
    assert before['completed_updates'] == record['completed_updates_before_measure']
    assert after['completed_updates'] == before['completed_updates'] + 1
    assert record['updates_after'] == after['completed_updates']
    assert next_record['completed_updates_before_measure'] == after['completed_updates']
    # Objective changes are comparable only with the same data and dual weights.
    for a, b in zip(record['rows'], next_record['rows']):
        assert (a['local_id'], a['context_ids']) == (b['local_id'], b['context_ids'])
        assert a['dual_multiplier'] == b['dual_multiplier'] == 0
        assert a['dual_linear_term'] == a['augmented_penalty'] == a['tail_quadratic_penalty'] == 0
        assert b['dual_linear_term'] == b['augmented_penalty'] == b['tail_quadratic_penalty'] == 0
    assert before['shared_optimizer']['param_groups'] == after['shared_optimizer']['param_groups']
    roles = {}
    old_opt, new_opt = before['shared_optimizer'], after['shared_optimizer']
    for group in new_opt['param_groups']:
        role = group['role']
        selected = [n for n in names if n.startswith(('matrix_output.', 'vector_output.')) == (role == 'heads')]
        assert len(selected) == len(group['params'])
        pairs = parameter_pairs(before['actor'], after['actor'], old_opt, new_opt, selected, group, after['completed_updates'])
        roles['shared/' + role] = summarize_role(pairs, record['role_preclip_grad_norms'][role])
        recorded_clipped_norm = record['role_postclip_grad_norms'][role]
        assert math.isclose(roles['shared/' + role]['recovered_clipped_gradient_l2'],
                            recorded_clipped_norm, rel_tol=2e-4)
    for uid, new_opt in after['private_optimizers'].items():
        old_opt = before['private_optimizers'][uid]
        assert old_opt['param_groups'] == new_opt['param_groups']
        assert len(new_opt['param_groups']) == 1
        group = new_opt['param_groups'][0]
        # AdaptiveCheckpointProbes contains raw frame parameters and a Linear;
        # there are no tensor buffers. Reject new layouts instead of guessing.
        selected = [n for n, v in after['private'][uid].items() if isinstance(v, torch.Tensor)]
        assert all(n.endswith('.raw') or n in ('trajectory_input.weight', 'trajectory_input.bias') for n in selected)
        assert len(selected) == len(group['params'])
        pairs = parameter_pairs(before['private'][uid], after['private'][uid], old_opt, new_opt, selected, group, after['completed_updates'])
        row = next(r for r in record['rows'] if str(r['local_id']) == str(uid))
        roles['private/' + str(uid)] = summarize_role(pairs, row['private_preclip_grad_norm_fp64'], .5)
    result = dict(update=after['completed_updates'], roles=roles,
                  objective_before=record['mean_objective'], objective_after=next_record['mean_objective'],
                  measured_objective_change=next_record['mean_objective']-record['mean_objective'])
    for kind in ('actual', 'ideal'):
        result[kind + '_gradient_dot_step'] = sum(r[kind]['gradient_dot_step'] for r in roles.values())
        result[kind + '_dot_roundoff_estimate'] = sum(r[kind]['dot_roundoff_estimate'] for r in roles.values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('runs/main-objective-local'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    root = args.root.resolve()
    short = root / 'known-target-small-role-lrs-four'
    long = root / 'known-target-small-role-lrs-sixtyfour'
    comp = root / 'known-target-compensated-four'
    plan = json.loads((long / 'plan.json').read_text())
    sys.path.insert(0, plan['runtime'])
    from rewriter.grouped_coordinate_chisel import GroupedCoordinateChisel
    actor = GroupedCoordinateChisel(rank=8, trajectory_dim=400, model_dim=256,
                                   num_layers=2, latent_tokens=8, fresh_weight_core=True,
                                   factor_precision='fp64', residual_precision='fp32')
    names = [n for n, _ in actor.named_parameters()]
    del actor
    sources = {}

    def load(path):
        if str(path) not in sources:
            with path.open('rb') as f:
                sources[str(path)] = hashlib.file_digest(f, 'sha256').hexdigest()
        return torch.load(path, map_location='cpu', weights_only=False, mmap=True)

    experiments = []
    for name, folders, steps in [('ordinary', [short, long], [1, 2, 3, 4, 58, 59, 60, 61, 62, 63, 64]),
                                 ('compensated', [comp], [1, 2, 3, 4])]:
        metrics = [f / 'training/metrics.jsonl' for f in folders]
        records = read_metrics(metrics)
        for path in metrics:
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        points = []
        for step in steps:
            def path(t):
                folder = (short if t <= 4 else long) if name == 'ordinary' else comp
                return folder / f'training/checkpoint_{t:04d}.pt'
            before, after = load(path(step-1)), load(path(step))
            point = measure_pair(before, after, records[step-1], records[step], names)
            points.append(point)
            print(json.dumps(dict(experiment=name, **{k:v for k,v in point.items() if k != 'roles'})), flush=True)
            del before, after
        experiments.append(dict(name=name, points=points))
    output = dict(scope=__doc__, experiments=experiments, sources=sources,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  computation=dict(gpt_forwards=0, gpt_backwards=0, chisel_forwards=0, optimizer_updates=0),
                  limitations=['Moment reconstruction and inverse clipping are approximate.',
                               'Dot products are local derivatives, not finite-change predictions or candidate evaluations.',
                               'Ideal step uses recorded moments without parameter rounding; it is not a counterfactual training run.',
                               'Only two fitted early GPT checkpoints and their fixed training panels are analyzed.'],
                  elapsed_seconds=time.perf_counter()-started,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()
