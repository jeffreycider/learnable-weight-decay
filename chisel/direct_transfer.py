"""Injected optimizer: preview a successor, differentiate MuonH, commit one edit."""
import torch
from chisel.harness import build, locate
from chisel.hypergradient import transfer_gradient, normalized_transfer, joint_norm
from chisel.muonh import MuonH, sphere


class DirectTransfer:
    def __init__(self, base, state, config, optimizer_group, schedule,
                 lookahead, gradient_floor, gradient_relative_floor=1e-6, preview_zero=False):
        self.base, self.state, self.config = base, state, config
        self.groups = base.groups
        self.hidden = self.groups[optimizer_group]
        if not isinstance(self.hidden, MuonH):
            raise TypeError('direct transfers require the literal MuonH optimizer')
        ids = {id(p) for group in self.hidden.param_groups for p in group['params']}
        self.parameters = {name: p for name, p in state.model.named_parameters() if id(p) in ids}
        self.raw = {name: p.detach().clone() for name, p in self.parameters.items()}
        self.consolidated = {name: torch.zeros_like(p) for name, p in self.parameters.items()}
        self.schedule, self.lookahead = schedule, locate(lookahead)
        self.gradient_floor, self.preview_zero = gradient_floor, preview_zero
        self.gradient_relative_floor = gradient_relative_floor
        self.counts = dict(ordinary_batches=0, lookahead_batches=0, ordinary_microbatches=0,
                           lookahead_microbatches=0, ordinary_tokens=0, lookahead_tokens=0)

    def zero_grad(self, **kwargs):
        self.base.zero_grad(**kwargs)

    def state_dict(self):
        return dict(base=self.base.state_dict(), raw=self.raw,
                    consolidated=self.consolidated, counts=self.counts)

    def load_state_dict(self, saved):
        self.base.load_state_dict(saved['base'])
        with torch.no_grad():
            for bank in ('raw', 'consolidated'):
                if saved[bank].keys() != self.parameters.keys():
                    raise ValueError('transfer parameter selection changed')
                for name, value in getattr(self, bank).items():
                    value.copy_(saved[bank][name])
            for name, p in self.parameters.items():
                if not torch.equal(p, self.raw[name] + self.consolidated[name]):
                    raise ValueError('checkpoint effective weights differ from R+C')
        self.counts = dict(saved['counts'])

    @torch.no_grad()
    def step(self):
        state, config = self.state, self.config
        size = locate(self.schedule['name'])(state.step, config['train_steps'], **self.schedule['kwargs'])
        batch = state.extras['batch']
        targets = batch['targets']
        count = (len(targets) if targets.is_floating_point() else targets.numel()) * state.extras['world_size']
        microbatches = (len(targets) + config['microbatch_size'] - 1) // config['microbatch_size']
        self.counts['ordinary_batches'] += 1
        self.counts['ordinary_microbatches'] += microbatches
        self.counts['ordinary_tokens'] += count
        # This consumes g_t and advances Muon momentum exactly once.
        directions = dict(self.hidden.prepare())
        rates = {p: group['lr'] for group in self.hidden.param_groups for p in group['params']}
        for name, p in self.parameters.items():
            p.copy_(self.raw[name])
        self.base.step()
        proposals = {name: p.detach().clone() for name, p in self.parameters.items()}
        for name, p in self.parameters.items():
            p.add_(self.consolidated[name])
        before_norm = joint_norm([proposals[n] + self.consolidated[n] for n in proposals])
        metrics = dict(edit_size=size, transfer_norm=0., transfer_gradient_norm=0., lookahead_loss=None,
                       base_displacement_norm=joint_norm([proposals[n] - self.raw[n] for n in proposals]).item())
        if size != 0 or self.preview_zero:
            # All other parameters are already at their ordinary successors too.
            with torch.enable_grad():
                q, observed = self.lookahead(config, state, self.parameters)
            self.counts['lookahead_batches'] += 1
            self.counts['lookahead_microbatches'] += observed['lookahead_microbatches']
            self.counts['lookahead_tokens'] += count
            gradients = {n: transfer_gradient(self.raw[n], directions[p], rates[p], q[n])
                         for n, p in self.parameters.items()}
            floor = max(self.gradient_floor, self.gradient_relative_floor * joint_norm(q.values()).item())
            edits, hnorm, unorm = normalized_transfer(self.raw, proposals, gradients, size, floor)
            metrics.update(observed, transfer_gradient_norm=hnorm.item(), base_displacement_norm=unorm.item(),
                           transfer_norm=joint_norm(edits.values()).item())
            if size != 0:
                for name, p in self.parameters.items():
                    proposals[name] = sphere(self.raw[name] + edits[name], directions[p], rates[p])
                    self.consolidated[name].sub_(edits[name])
        for name, p in self.parameters.items():
            self.raw[name].copy_(proposals[name])
            p.copy_(self.raw[name] + self.consolidated[name])
        after_norm = joint_norm(self.parameters.values())
        metrics.update(effective_norm=after_norm.item(),
                       effective_norm_relative_change=((after_norm - before_norm) / before_norm.clamp_min(1e-30)).item(),
                       consolidated_norm=joint_norm(self.consolidated.values()).item(), **self.counts)
        metrics['model_backward_microbatches'] = self.counts['ordinary_microbatches'] + self.counts['lookahead_microbatches']
        metrics['model_training_tokens'] = self.counts['ordinary_tokens'] + self.counts['lookahead_tokens']
        if next(iter(self.parameters.values())).is_cuda:
            metrics['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        state.extras.setdefault('metrics', {}).update(metrics)


def setup():
    def hook(config, state):
        state.optimizer = build(config['direct_transfer'], base=state.optimizer, state=state, config=config)
    return hook
