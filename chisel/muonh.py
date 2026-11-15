"""PR267 MuonH, batched over equal shapes; pure algebra is shared with Chisel.

Pinned record: KellerJordan/modded-nanogpt@99604285f4da6afc43b5b49f55a4548ebf5318ef,
records/track_3_optimization/results/20260430_muonh/9319c798-6643-464a-b407-b05468e468f5.txt.
"""
from collections import defaultdict
import torch


def direction(gradient, momentum, mu, ns_steps):
    updated_momentum = momentum.lerp(gradient, 1 - mu)
    update = gradient.lerp(updated_momentum, mu).bfloat16()
    transposed = update.shape[-2] > update.shape[-1]
    if transposed:
        update = update.mT
    update = update / (update.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(ns_steps):
        gram = update @ update.mT
        polynomial = -1.5 * gram + .5 * gram @ gram
        update = 2 * update + polynomial @ update
    if transposed:
        update = update.mT
    update = update * max(1, gradient.shape[-2] / gradient.shape[-1])**.5
    return updated_momentum, update


def sphere(raw, update, lr, eps=1e-10):
    """Preserve PR267's BF16 intermediate rounding even for stacked matrices."""
    radius = raw.norm(dim=(-2, -1), keepdim=True)
    delta = (lr * update.to(raw.dtype)).to(update.dtype)
    delta = (delta.to(raw.dtype) * radius).to(update.dtype)
    delta = delta / update.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)
    proposal = raw - delta
    return proposal / proposal.norm(dim=(-2, -1), keepdim=True).clamp_min(eps) * radius


class MuonH(torch.optim.Optimizer):
    def __init__(self, params, lr, mu, ns_steps, compile=False):
        super().__init__(params, dict(lr=lr, mu=mu, ns_steps=ns_steps))
        self.calculate_direction = torch.compile(direction, fullgraph=True, dynamic=False) if compile else direction
        self.calculate_step = torch.compile(sphere, fullgraph=True, dynamic=False) if compile else sphere
        self.prepared = {}

    @torch.no_grad()
    def prepare(self):
        if self.prepared:
            raise RuntimeError('previous MuonH proposal was not consumed')
        for group in self.param_groups:
            shapes = defaultdict(list)
            for parameter in group['params']:
                if parameter.grad is None:
                    raise RuntimeError('MuonH requires a gradient for every selected matrix')
                shapes[tuple(parameter.shape)].append(parameter)
            for parameters in shapes.values():
                momentum = [self.state[p].setdefault('momentum', torch.zeros_like(p)) for p in parameters]
                new_momentum, updates = self.calculate_direction(
                    torch.stack([p.grad for p in parameters]), torch.stack(momentum), group['mu'], group['ns_steps'])
                torch._foreach_copy_(momentum, list(new_momentum.unbind()))
                for parameter, update in zip(parameters, updates.unbind()):
                    self.prepared[parameter] = update
        return self.prepared

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError('MuonH does not evaluate closures')
        if not self.prepared:
            self.prepare()
        for group in self.param_groups:
            shapes = defaultdict(list)
            for parameter in group['params']:
                shapes[tuple(parameter.shape)].append(parameter)
            for parameters in shapes.values():
                updated = self.calculate_step(torch.stack(parameters), torch.stack([self.prepared[p] for p in parameters]), group['lr'])
                torch._foreach_copy_(parameters, list(updated.unbind()))
        self.prepared.clear()


class Optimizers:
    """Named optimizers with a single lifecycle and serialization interface."""
    def __init__(self, groups):
        self.groups = groups

    def step(self):
        for optimizer in self.groups.values():
            optimizer.step()

    def zero_grad(self, **kwargs):
        for optimizer in self.groups.values():
            optimizer.zero_grad(**kwargs)

    def state_dict(self):
        return {name: optimizer.state_dict() for name, optimizer in self.groups.items()}

    def load_state_dict(self, state):
        if state.keys() != self.groups.keys():
            raise ValueError('optimizer group names changed')
        for name, optimizer in self.groups.items():
            optimizer.load_state_dict(state[name])
