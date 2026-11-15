"""Consolidation and direct one-step credit through unchanged MuonH algebra."""
import torch
from chisel.muonh import sphere
from chisel.network import propose


def bound_transfer(snapshot, transfer, radius_floor):
    """One differentiable multiplier guarantees ||R-D|| >= floor by triangle inequality."""
    bounds = [value.new_ones(()) for value in transfer.values()]
    for name, value in transfer.items():
        radius = snapshot[name]['raw'].norm()
        if radius.detach() < radius_floor:
            raise ValueError(f'raw radius already below floor: {name}')
        bounds.append((radius - radius_floor).clamp_min(0) / value.norm().clamp_min(1e-30))
    scale = torch.stack(bounds).min()
    return {name: scale * value for name, value in transfer.items()}


def apply_transition(snapshot, transfer, directions, learning_rates):
    effective, consolidated = {}, {}
    for name, delta in transfer.items():
        raw = snapshot[name]['raw'] - delta
        consolidated[name] = snapshot[name]['consolidated'] + delta
        effective[name] = sphere(raw, directions[name], learning_rates[name]) + consolidated[name]
    return effective, consolidated


def penalty(consolidated):
    return sum(value.square().sum() for value in consolidated.values()) / sum(value.numel() for value in consolidated.values())


def action(actor, probes, snapshot, directions, learning_rates, radius_floor):
    transfer = bound_transfer(snapshot, propose(actor, probes, snapshot), radius_floor)
    effective, consolidated = apply_transition(snapshot, transfer, directions, learning_rates)
    return transfer, effective, consolidated


def delayed_loss(effective, consolidated, next_gradient, coefficient):
    credit = sum((value * next_gradient[name].detach()).sum() for name, value in effective.items())
    return credit + coefficient * penalty(consolidated)


def observe(parameters, raw, consolidated, optimizer, activity):
    """Freeze all action-time inputs before the ordinary optimizer mutates storage."""
    directions = {name: optimizer.prepared[p].detach().clone() for name, p in parameters.items()}
    groups = {p: group for group in optimizer.param_groups for p in group['params']}
    learning_rates = {name: torch.as_tensor(groups[p]['lr'], device=p.device).detach().clone() for name, p in parameters.items()}
    snapshot = {}
    for name, parameter in parameters.items():
        snapshot[name] = dict(raw=raw[name].detach().clone(), consolidated=consolidated[name].detach().clone(),
                              gradient=parameter.grad.detach().clone(),
                              momentum=optimizer.state[parameter]['momentum'].detach().clone(),
                              proposal=(sphere(raw[name], directions[name], learning_rates[name]) - raw[name]).detach().clone(),
                              activity={key: value.detach().clone() for key, value in activity.get(name, {}).items()})
    return snapshot, directions, learning_rates
