"""Batched finite validation for an already materialized online proposal.

This optional helper does not mutate parameters or commit an action. Call it
before any live-weight copy, just as the sequential finite guard does. It makes
one host scalar decision per device instead of one per parameter tensor. Device
reductions are unchanged; no measured runtime improvement is implied.
"""
from collections import defaultdict

import torch


@torch.no_grad()
def require_finite_proposal(proposal):
    """Raise before commit if any proposed value is NaN or infinite.

    Empty proposals/tensors retain the sequential guard's vacuous truth.
    Mixed devices are checked separately, avoiding transfers between devices.
    The caller owns proposal stability until all live copies complete.
    """
    flags_by_device = defaultdict(list)
    for value in proposal.values():
        flags_by_device[value.device].append(torch.isfinite(value).all())
    for flags in flags_by_device.values():
        if not bool(torch.stack(flags).all()):
            raise FloatingPointError('nonfinite proposal; live weights unchanged')
