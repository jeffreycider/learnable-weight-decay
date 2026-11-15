"""Optional whole-model edit budget relative to actual base optimizer motion.

Pure optimizer algebra, not wired into any trainer by importing this module.
Build the radius once after the COMPLETE base update, with its measured actual
stored-parameter displacement norm. Reuse that immutable radius for every
candidate and the final proposal. Do not substitute LR or momentum norm for U.

The caller must pin the norm's parameter scope and checkpoint this policy.
The reference scope is the whole GPT, including parameters Chisel cannot edit.
This is a global action bound, not an edit-size loss or per-layer clipping.
It bounds the proposed edit; committing it in finite precision can change its
realized norm slightly. Continue logging proposed and stored-weight motion.
"""
import math

import torch


def _scalar(value, name):
    if not isinstance(value, torch.Tensor) or value.ndim != 0 or not value.is_floating_point():
        raise ValueError(name+' must be a floating scalar tensor')
    if not bool(torch.isfinite(value)) or bool(value < 0):
        raise ValueError(name+' must be finite and nonnegative')


def base_motion_radius(base_update_norm, *, fraction):
    """Return detached/copied R = fraction * ||U|| in parameter-norm units.

    No positive semantic floor: a zero base update permits only an identity edit.
    Numerical stability belongs in the action-norm denominator, not in a floor
    that permits finite edits after base optimization has stopped.
    """
    _scalar(base_update_norm, 'base_update_norm')
    if base_update_norm.requires_grad:
        raise ValueError('base displacement must be frozen optimizer information')
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or fraction <= 0:
        raise ValueError('positive finite base-motion fraction required')
    radius = base_update_norm.detach().clone() * fraction
    _scalar(radius, 'radius')
    return radius


def restrict_global_scale(action_norm_squared, existing_scale, radius, *, norm_floor=1e-12):
    """Differentiably contract an existing global scale to obey the new radius.

    Use the UNscaled raw action norm and the same single scalar for every matrix
    and vector. The existing scale may already enforce another bound; it remains
    connected to autograd. None means disabled and returns that exact object.
    The action norm must describe the same factors later committed to GPT.

    The scalar radius is frozen and copied so mutation of a caller's reused
    statistic cannot change the delayed VJP. Keep this graph for lag-one credit.
    """
    if radius is None:
        return existing_scale
    for name, value in [('action_norm_squared', action_norm_squared),
                        ('existing_scale', existing_scale), ('radius', radius)]:
        _scalar(value, name)
    if radius.requires_grad:
        raise ValueError('action radius must not depend on learned parameters')
    if bool(existing_scale > 1):
        raise ValueError('existing scale must be a contraction in [0, 1]')
    if not (action_norm_squared.device == existing_scale.device == radius.device):
        raise ValueError('all budget scalars must be on the same device')
    if isinstance(norm_floor, bool) or not isinstance(norm_floor, (int, float)) or not math.isfinite(norm_floor) or norm_floor <= 0:
        raise ValueError('positive finite numerical norm floor required')
    # FP32 minimum prevents BF16/FP16 squaring underflow at the floor. Preserve
    # FP64 tests/production statistics when any input already uses FP64.
    dtype = torch.float64 if any(v.dtype == torch.float64 for v in
                               (action_norm_squared, existing_scale, radius)) else torch.float32
    square = action_norm_squared.to(dtype)
    floor_squared = square.new_tensor(norm_floor * norm_floor)
    if not bool(torch.isfinite(floor_squared)) or bool(floor_squared == 0):
        raise ValueError('norm floor squared is not representable')
    norm = square.clamp_min(floor_squared).sqrt()
    bound = radius.detach().clone().to(dtype) / (norm + norm_floor)
    return torch.minimum(existing_scale.to(dtype), bound)
