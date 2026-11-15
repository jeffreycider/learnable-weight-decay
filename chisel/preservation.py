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


def prediction_terms(student, teacher, top_k=100, margin_fraction=.5):
    """No GPT approximation: compare actual predictions on the same examples."""
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
    violations = (margin_fraction * gaps - (ordered[:, :-1] - ordered[:, 1:])).clamp_min(0)
    # Exact teacher ties have no ordering constraint.
    rank_loss = torch.where(gaps > 0, violations.square(), 0).mean(-1)
    if k < teacher.shape[-1]:
        outside = student.scatter(-1, indices, float('-inf')).max(-1).values
        teacher_outside = teacher.scatter(-1, indices, float('-inf')).max(-1).values
        boundary = values[:, -1] - teacher_outside
        violation = (margin_fraction * boundary - (ordered[:, -1] - outside)).clamp_min(0)
        rank_loss = rank_loss + torch.where(boundary > 0, violation.square(), 0)
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
