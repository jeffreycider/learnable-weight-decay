"""Increase learned read/write bandwidth while preserving the initial Chisel function."""
import math
import torch


def block_indices(old, new, blocks):
    return [bank * new * new + row * new + col
            for bank in range(blocks) for row in range(old) for col in range(old)]


@torch.no_grad()
def actor(model, saved):
    old = math.isqrt(saved['proj.bias'].numel() // 2)
    new = model.rank
    if new < old or 2 * old * old != saved['proj.bias'].numel():
        raise ValueError('rank expansion requires an integral, nondecreasing probe rank')
    inputs = block_indices(old, new, 5)
    inputs += [5 * new * new + bank * new + col for bank in range(2) for col in range(old)]
    inputs += list(range(5 * new * new + 2 * new, 5 * new * new + 2 * new + 9))
    outputs = block_indices(old, new, 2)
    result = dict(saved)
    result['embed.weight'] = torch.zeros_like(model.embed.weight)
    result['embed.weight'][:, inputs] = saved['embed.weight'].to(model.embed.weight)
    result['proj.weight'] = torch.zeros_like(model.proj.weight)
    result['proj.bias'] = torch.zeros_like(model.proj.bias)
    result['proj.weight'][outputs] = saved['proj.weight'].to(model.proj.weight)
    result['proj.bias'][outputs] = saved['proj.bias'].to(model.proj.bias)
    if model.gain_proj is not None:
        indices = [side * (1 + 2 * new) + index for side in range(2)
                   for index in [0, *range(1, 1 + old), *range(1 + new, 1 + new + old)]]
        for suffix in ('weight', 'bias'):
            name = 'gain_proj.' + suffix
            target = torch.zeros_like(getattr(model.gain_proj, suffix))
            target[indices] = saved[name].to(target)
            result[name] = target
    model.load_state_dict(result, strict=True)


@torch.no_grad()
def probes(model, saved):
    current = model.state_dict()
    if current.keys() != saved.keys():
        raise ValueError('checkpoint-private probe inventory changed')
    for name, value in saved.items():
        target = current[name]
        if value.shape[0] != target.shape[0] or value.shape[1] > target.shape[1]:
            raise ValueError('probe geometry is not a rank expansion')
        target[:, :value.shape[1]].copy_(value)
    # New map columns retain their seeded random initialization. The newly added
    # output codes are zero, but their derivatives can immediately use these maps.
