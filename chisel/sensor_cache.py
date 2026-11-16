"""Immutable sensor normalization shared by repeated fits to one snapshot."""
import math
import torch

BANKS = ('raw', 'consolidated', 'gradient', 'momentum', 'proposal')


@torch.no_grad()
def normalized(item):
    banks, scales, activity = [], [], []
    raw_scale = None
    for name in BANKS:
        value = item[name].detach()
        rms = value.square().mean().sqrt()
        if name == 'raw':
            raw_scale = rms
        scale = rms.clamp_min(1e-12)
        banks.append(value / scale)
        scales.append(scale.log1p().reshape(1))
    rows, columns = item['raw'].shape
    for name, size in [('input', columns), ('adjoint', rows)]:
        value = item.get('activity', {}).get(name, item['raw'].new_zeros(size)).detach()
        scale = value.square().mean().sqrt().clamp_min(1e-12)
        activity.append(value / scale)
        scales.append(scale.log1p().reshape(1))
    return dict(banks=tuple(banks), activity=tuple(activity), scales=tuple(scales),
        dimensions=item['raw'].new_tensor([math.log(rows), math.log(columns)]), raw_scale=raw_scale)


def prepare(snapshot):
    # Call only on immutable observation snapshots, never on live GPT parameters.
    return {name: item | {'encoder_cache': normalized(item)} for name, item in snapshot.items()}
