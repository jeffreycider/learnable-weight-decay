"""Effective-weight shrinkage and direct, differentiable prediction preservation."""
import re
import torch


def group_norm_change(snapshot, effective):
    """Mean fractional norm change by weight type; redistribution earns zero."""
    groups = {}
    for name, after in effective.items():
        before = snapshot[name]['raw'] + snapshot[name]['consolidated']
        delta = after - before
        family = re.sub(r'^blocks\.\d+\.', '', name)
        norm, change = groups.get(family, (before.new_zeros(()), before.new_zeros(())))
        groups[family] = (norm + before.square().sum(), change + (2 * before * delta + delta.square()).sum())
    changes = []
    for norm, change in groups.values():
        relative_squared = change / norm.clamp_min(1e-30)
        relative_norm = relative_squared / ((1 + relative_squared).clamp_min(0).sqrt() + 1)
        changes.append(torch.where(norm > 0, relative_norm, change))
    return torch.stack(changes).mean()


def effective_norm_change(snapshot, effective):
    """Fractional Frobenius-norm change over all editable effective matrices."""
    norm, change = next(iter(effective.values())).new_zeros(2)
    for name, after in effective.items():
        before = snapshot[name]['raw'] + snapshot[name]['consolidated']
        delta = after - before
        norm = norm + before.square().sum()
        change = change + (2 * before * delta + delta.square()).sum()
    relative = change / norm.clamp_min(1e-30)
    return relative / ((1 + relative).clamp_min(0).sqrt() + 1)


def prediction_terms(student, teacher, top_k=100, margin_fraction=.5, ranking_power=2):
    """No GPT approximation: compare actual predictions on the same examples."""
    if ranking_power not in (1, 2):
        raise ValueError('ranking_power must select linear or squared hinge')
    student, teacher = student.flatten(0, -2), teacher.detach().flatten(0, -2)
    difference = student - teacher
    centered = difference - difference.mean(-1, keepdim=True)
    mse = centered.square().mean(-1)
    # A centered cumulant avoids subtracting nearly equal cross-entropies.
    probability = teacher.double().softmax(-1)
    delta = difference.double()
    delta = delta - (probability * delta).sum(-1, keepdim=True)
    kl = torch.log1p((probability * torch.expm1(delta)).sum(-1)).clamp_min(0)
    k = min(top_k, teacher.shape[-1])
    values, indices = teacher.topk(k, dim=-1, sorted=True)
    ordered = student.gather(-1, indices)
    gaps = values[:, :-1] - values[:, 1:]
    groups = torch.cat((torch.zeros_like(indices[:, :1]), (gaps > 0).long().cumsum(-1)), dim=-1)
    minimum = torch.full_like(ordered, float('inf')).scatter_reduce(1, groups, ordered, reduce='amin')
    maximum = torch.full_like(ordered, float('-inf')).scatter_reduce(1, groups, ordered, reduce='amax')
    if k < teacher.shape[-1]:
        # A tie crossing the cutoff may select any required number of tied tokens.
        tied = teacher == values[:, -1:]
        inside_count = (values == values[:, -1:]).sum(-1)
        boundary_ties = tied.sum(-1) > inside_count
        cutoff = minimum.gather(1, groups[:, -1:])[:, 0]
        if boundary_ties.any():
            candidates = student[boundary_ties].masked_fill(~tied[boundary_ties], float('-inf'))
            best = candidates.topk(k, dim=-1).values
            cutoff = cutoff.clone()
            cutoff[boundary_ties] = best.gather(1, (inside_count[boundary_ties] - 1)[:, None])[:, 0]
            last_max = maximum.gather(1, groups[:, -1:]).clone()
            last_max[boundary_ties, 0] = best[:, 0]
            maximum = maximum.scatter(1, groups[:, -1:], last_max)
        lower = teacher < values[:, -1:]
        outside = student.masked_fill(~lower, float('-inf')).max(-1).values
        teacher_outside = teacher.masked_fill(~lower, float('-inf')).max(-1).values
        boundary = values[:, -1] - teacher_outside
        boundary = torch.where(torch.isfinite(boundary), boundary, 0)
        boundary_loss = (margin_fraction * boundary - (cutoff - outside)).clamp_min(0).pow(ranking_power)
    # Every member of an upper tie group must precede every lower group.
    separation = minimum.gather(1, groups[:, :-1]) - maximum.gather(1, groups[:, 1:])
    violations = (margin_fraction * gaps - separation).clamp_min(0)
    rank_loss = torch.where(gaps > 0, violations.pow(ranking_power), 0).sum(-1) / max(1, k - 1)
    if k < teacher.shape[-1]:
        rank_loss = rank_loss + boundary_loss
    return dict(mse=mse, kl=kl, ranking=rank_loss)


def ranking_changes(student, teacher, top_k=100):
    student, teacher = student.flatten(0, -2), teacher.flatten(0, -2)
    k = min(top_k, teacher.shape[-1])
    expected, _ = teacher.topk(k, dim=-1, sorted=True)
    _, actual = student.topk(k, dim=-1, sorted=True)
    changed = teacher.gather(-1, actual) != expected
    greedy = teacher.gather(-1, student.argmax(-1, keepdim=True))[:, 0] != teacher.max(-1).values
    return dict(rank_changed=changed.any(-1).sum(), greedy_changed=greedy.sum(), positions=teacher.shape[0])


class Dual:
    def __init__(self, initial, learning_rate, tolerance, maximum):
        self.value, self.learning_rate, self.tolerance, self.maximum = initial, learning_rate, tolerance, maximum

    def update(self, kl):
        self.value = max(0., min(self.maximum, self.value + self.learning_rate * (kl - self.tolerance)))

    def state_dict(self):
        return dict(value=self.value)

    def load_state_dict(self, state):
        self.value = state['value']
