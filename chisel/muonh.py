"""PR267 MuonH with its scalar floating-point operations; pure algebra is shared with Chisel.

Pinned record: KellerJordan/modded-nanogpt@99604285f4da6afc43b5b49f55a4548ebf5318ef,
records/track_3_optimization/results/20260430_muonh/9319c798-6643-464a-b407-b05468e468f5.txt.
"""
import torch


def direction(gradient, momentum, mu, ns_steps):
    momentum.lerp_(gradient, 1 - mu)
    update = gradient.lerp_(momentum, mu).bfloat16()
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
    update *= max(1, gradient.shape[-2] / gradient.shape[-1])**.5
    return update


def sphere(raw, update, lr, eps=1e-10):
    """Literal PR267 scalar arithmetic; use this same map for execution and credit."""
    if raw.ndim != 2:
        raise ValueError('sphere requires one matrix: CUDA scalar promotion matters')
    radius = raw.norm()
    proposal = raw - lr * update * radius / update.norm().clamp_min(eps)
    return proposal / proposal.norm().clamp_min(eps) * radius


class MuonH(torch.optim.Optimizer):
    def __init__(self, params, lr, mu, ns_steps, compile=False):
        super().__init__(params, dict(lr=lr, mu=mu, ns_steps=ns_steps))
        self.calculate_direction = torch.compile(direction, fullgraph=True, dynamic=False) if compile else direction
        self.prepared = {}

    @torch.no_grad()
    def prepare(self):
        if self.prepared:
            raise RuntimeError('previous MuonH proposal was not consumed')
        for group in self.param_groups:
            for parameter in group['params']:
                if parameter.grad is None:
                    raise RuntimeError('MuonH requires a gradient for every selected matrix')
                momentum = self.state[parameter].setdefault('momentum', torch.zeros_like(parameter))
                # PR267 mutates its input gradient; own that scratch buffer so sensors
                # and delayed credit retain the unmodified ordinary task gradient.
                self.prepared[parameter] = self.calculate_direction(parameter.grad.clone(), momentum, group['mu'], group['ns_steps'])
        return self.prepared

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError('MuonH does not evaluate closures')
        if not self.prepared:
            self.prepare()
        for group in self.param_groups:
            for parameter in group['params']:
                parameter.copy_(sphere(parameter, self.prepared[parameter], group['lr']))
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
