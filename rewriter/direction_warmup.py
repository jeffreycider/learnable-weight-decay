"""Same-run supervised warmup for the existing learned anchored write heads.

The targets are the Euclidean projection of negative weights into the anchored
action space. They teach a direction, not a preservation policy. Deployment
still evaluates the learned network and trains finite preservation loss; no
analytic direction is added to its output.
"""
from dataclasses import dataclass
import math
import torch
from .algebra import Factors, RawAction


@dataclass
class DirectionTargets:
    action: RawAction
    groups: dict
    # Squared target magnitude in decoder coordinates, separately per role.
    group_squared: dict


@torch.no_grad()
def direction_targets(snapshot, groups, reference_squared, parameter_shapes):
    """Target the canonical projected -W in the existing decoder's units."""
    matrices, vectors = {}, {}
    for group, names in groups.items():
        count = sum(math.prod(parameter_shapes[n]) for n in names)
        ref = reference_squared[group]
        rms = (ref / count).sqrt().float()
        active = bool(ref > 0)
        for name in names:
            if name in snapshot.matrices:
                obs = snapshot.matrices[name]
                wb, wta = obs.weight_probes
                rank = obs.anchors.A.shape[1]
                shape = parameter_shapes[name]
                scale = rms * math.sqrt(math.prod(shape) / (rank * (sum(shape) - rank)))
                if active:
                    p = -(wb.float() - obs.anchors.A @ (obs.anchors.A.T @ wb.float())) / scale
                    q = -wta.float() / scale
                else:
                    p, q = torch.zeros_like(wb), torch.zeros_like(wta)
                matrices[name] = Factors(p, q, obs.anchors)
            else:
                value = snapshot.vectors[name].weight
                vectors[name] = -value.float() / rms if active else torch.zeros_like(value)
    action = RawAction(matrices, vectors)
    squared = {}
    for group, names in groups.items():
        parts = [v.square().sum() for n in names for v in coordinates(action, n)]
        squared[group] = torch.stack(parts).sum().detach()
    return DirectionTargets(action, groups, squared)


def coordinates(action, name):
    if name in action.matrices:
        factor = action.matrices[name]
        return factor.P, factor.Q
    return (action.vectors[name],)


def direction_loss(action, targets):
    """Equal functional-role relative squared errors; zero groups stay zero."""
    losses = []
    for group, names in targets.groups.items():
        denom = targets.group_squared[group]
        residuals = [(a - t).square().sum() for n in names
                     for a, t in zip(coordinates(action, n), coordinates(targets.action, n))]
        numerator = torch.stack(residuals).sum()
        # Both branches are finite, including exactly-zero reference groups.
        losses.append(torch.where(denom > 0, numerator / denom.clamp_min(1e-30), numerator * 0))
    return torch.stack(losses).mean()


@torch.no_grad()
def direction_metrics(action, targets):
    result = {}
    for group, names in targets.groups.items():
        a = [v for n in names for v in coordinates(action, n)]
        t = [v for n in names for v in coordinates(targets.action, n)]
        dot = sum((x * y).sum() for x, y in zip(a, t))
        norm = sum(x.square().sum() for x in a).sqrt()
        target_norm = targets.group_squared[group].sqrt()
        active = bool(target_norm > 0)
        result[group] = dict(active_target=active,
            cosine=float(dot / (norm * target_norm).clamp_min(1e-30)) if active else None,
            norm_ratio=float(norm / target_norm) if active else None)
    return result


def aggregate_direction_metrics(rows):
    """Exclude undefined zero-target directions; retain their explicit count."""
    if not rows:
        raise ValueError('nonempty directional measurements required')
    result = {}
    for group in rows[0]:
        active = [row[group] for row in rows if row[group]['active_target']]
        result[group] = dict(active_states=len(active), inactive_states=len(rows)-len(active),
            **{key:sum(row[key] for row in active)/len(active) if active else None
               for key in ('cosine', 'norm_ratio')})
    return result
