"""Consolidation and direct one-step credit through unchanged MuonH algebra."""
import torch
from chisel.muonh import sphere
from chisel.network import propose


def bound_edits(snapshot, edits, radius_floor):
    """One differentiable multiplier guarantees ||R+delta_R|| >= floor by triangle inequality."""
    names = tuple(edits)
    radii = torch.stack([snapshot[name]['raw'].norm() for name in names])
    sizes = torch.stack([edits[name]['raw'].norm() for name in names])
    if any(not torch.isfinite(item['consolidated']).all() for item in edits.values()):
        raise FloatingPointError('non-finite consolidated edit')
    if not torch.isfinite(radii).all() or not torch.isfinite(sizes).all():
        raise FloatingPointError('non-finite raw matrix or proposed transfer')
    if (radii.detach() < radius_floor).any():
        raise ValueError('raw radius already below floor')
    # A floor below arithmetic precision can round an edit all the way to R=0. Reserve a small, dtype-relative margin.
    floor = torch.maximum(radii.new_full(radii.shape, radius_floor), 8 * torch.finfo(radii.dtype).eps * radii)
    scale = torch.minimum(radii.new_ones(()), ((radii - floor).clamp_min(0) / sizes.clamp_min(1e-30)).min())
    return {name: {bank: scale * value for bank, value in item.items()} for name, item in edits.items()}


def apply_edits(snapshot, edits):
    raw = {name: snapshot[name]['raw'] + item['raw'] for name, item in edits.items()}
    consolidated = {name: snapshot[name]['consolidated'] + item['consolidated'] for name, item in edits.items()}
    effective = {name: raw[name] + consolidated[name] for name in edits}
    return raw, consolidated, effective


def apply_transition(snapshot, edits, directions, learning_rates):
    raw, consolidated, _ = apply_edits(snapshot, edits)
    effective = {name: sphere(raw[name], directions[name], learning_rates[name]) + consolidated[name] for name in edits}
    return effective, consolidated


def penalty(consolidated):
    return sum(value.square().sum() for value in consolidated.values()) / sum(value.numel() for value in consolidated.values())


def action(actor, probes, snapshot, directions, learning_rates, radius_floor, advance_optimizer=True):
    edits = bound_edits(snapshot, propose(actor, probes, snapshot), radius_floor)
    if advance_optimizer:
        effective, consolidated = apply_transition(snapshot, edits, directions, learning_rates)
    else:
        _, consolidated, effective = apply_edits(snapshot, edits)
    return edits, effective, consolidated


def delayed_loss(effective, consolidated, next_gradient, coefficient):
    credit = sum((value * next_gradient[name].detach()).sum() for name, value in effective.items())
    return credit + coefficient * penalty(consolidated)


def observe(parameters, raw, consolidated, optimizer, activity):
    """Freeze all action-time inputs before the ordinary optimizer mutates storage."""
    directions = {name: optimizer.prepared[p].detach().clone() for name, p in parameters.items()}
    groups = {p: group for group in optimizer.param_groups for p in group['params']}
    learning_rates = {name: groups[p]['lr'].detach().clone() if torch.is_tensor(groups[p]['lr']) else groups[p]['lr']
                      for name, p in parameters.items()}
    snapshot = {}
    for name, parameter in parameters.items():
        snapshot[name] = dict(raw=raw[name].detach().clone(), consolidated=consolidated[name].detach().clone(),
                              gradient=parameter.grad.detach().clone(),
                              momentum=optimizer.state[parameter]['momentum'].detach().clone(),
                              proposal=(sphere(raw[name], directions[name], learning_rates[name]) - raw[name]).detach().clone(),
                              activity={key: value.detach().clone() for key, value in activity.get(name, {}).items()})
    return snapshot, directions, learning_rates
