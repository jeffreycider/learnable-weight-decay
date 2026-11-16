"""Direct transfer through the literal MuonH step, with observed gradients frozen."""
import math
import torch
from chisel.muonh import sphere


def transfer_gradient(raw, direction, lr, cotangent):
    """D_R U(R)^T q - q; no graph through the task gradient or optimizer history."""
    with torch.enable_grad():
        variable = raw.detach().requires_grad_(True)
        rate = lr.detach() if torch.is_tensor(lr) else lr
        successor = sphere(variable, direction.detach(), rate)
        q = cotangent.detach()
        return torch.autograd.grad(successor, variable, q)[0].detach() - q


def joint_norm(values):
    # FP32 accumulation in production; preserve FP64 for finite-difference tests.
    return torch.stack([value.to(torch.float64 if value.dtype == torch.float64 else torch.float32)
                        .square().sum() for value in values]).sum().sqrt()


def normalized_transfer(raw, proposals, gradients, size, gradient_floor):
    """One multiplier across all replicated matrices; do not sum again across ranks."""
    if not math.isfinite(size) or size < 0 or gradient_floor <= 0:
        raise ValueError('transfer size must be finite/nonnegative; gradient floor positive')
    gradient_norm = joint_norm(gradients.values())
    step_norm = joint_norm([proposals[name] - value for name, value in raw.items()])
    if not torch.isfinite(gradient_norm) or not torch.isfinite(step_norm):
        raise FloatingPointError('non-finite transfer gradient or base displacement')
    scale = torch.where(gradient_norm > gradient_floor,
                        -size * step_norm / gradient_norm.clamp_min(gradient_floor), 0.)
    return {name: gradient * scale for name, gradient in gradients.items()}, gradient_norm, step_norm


def edit_schedule(step, total, size, start_step=0, ramp_steps=0, interval=1, terminal_identity=False):
    """Zero-indexed start; optional linear ramp, then a fixed transfer/base-step ratio."""
    if interval < 1 or ramp_steps < 0 or start_step < 0:
        raise ValueError('invalid edit schedule')
    if step < start_step or (step - start_step) % interval or (terminal_identity and step == total - 1):
        return 0.
    return size * min(1., (step - start_step + 1) / max(1, ramp_steps))
