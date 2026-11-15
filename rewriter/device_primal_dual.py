"""Optional native-op scalar path for offline CUDA-graph qualification.

No GPT/actor arithmetic, precision, objective or optimizer is replaced. Capture
these eager tensor operations; compiling/fusing them is a separate numerical
qualification because Python's original FP64 operation order must be retained.
The caller must validate the returned finite flag before Adam, outside capture.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def ordered_panel_mean(losses: Sequence[torch.Tensor], sizes: Sequence[int]):
    """Match ``total += float(loss.detach()) * (size / panel_size)`` grouping.

    Production computes ``fraction = size/panel_size`` first, then multiplies
    each Python double loss and adds sequentially. Keep these as separate native
    FP64 operations. In particular, do not stack/sum or use a fused addcmul.
    Only detached scalar losses are retained; no GPT activation graph is kept.
    """
    if not losses or len(losses) != len(sizes):
        raise ValueError('one scalar loss per microbatch size required')
    if any(type(n) is not int or n <= 0 for n in sizes):
        raise ValueError('positive integer microbatch sizes required')
    device = losses[0].device
    total_size = sum(sizes)
    total = torch.zeros((), dtype=torch.float64, device=device)
    for loss, size in zip(losses, sizes):
        if loss.ndim != 0 or loss.dtype not in (torch.float32, torch.float64) or loss.device != device:
            raise ValueError('FP32/FP64 scalar losses on one device required')
        weighted = loss.detach().to(torch.float64) * (size / total_size)
        total = total + weighted
    return total


def device_primal_dual_terms(kl, multiplier, tolerance, rho):
    """Native FP64 equivalent of PrimalDual.terms, plus a device finite flag.

    Inputs are resident FP64 scalar tensors. A negative finite KL retains the
    production negative-violation penalty and zero coefficient. At exact zero
    the coefficient is multiplier/tolerance, matching the host convention.
    """
    for value in (kl, multiplier, tolerance, rho):
        if value.ndim != 0 or value.dtype != torch.float64 or value.device != kl.device:
            raise ValueError('FP64 scalars on one device required')
    zero = torch.zeros_like(kl)
    violation = torch.maximum(kl, zero) / tolerance - 1.
    positive = torch.maximum(violation, zero)
    linear = multiplier * violation
    # Preserve .5*rho followed by positive**2, then multiplication and addition.
    quadratic = (.5 * rho) * positive.square()
    coefficient = (multiplier + rho * positive) / tolerance
    coefficient = torch.where(kl >= 0., coefficient, zero)
    return dict(kl_penalty=linear + quadratic, kl_coefficient=coefficient,
                constraint_violation=violation, positive_violation=positive,
                dual_multiplier=multiplier, dual_linear_term=linear,
                augmented_penalty=quadratic, kl_tolerance=tolerance,
                feasible=violation <= 0., finite=torch.isfinite(kl))


class DevicePrimalDualInputs:
    """Persistent scalar storage; bind the current state BEFORE each replay.

    Capture keeps these tensor addresses. Replacing them or capturing a Python
    multiplier silently uses stale dual state, so bind() only copies values.
    Binding never updates the host PrimalDual; its ordinary global step still
    runs once after Adam. Held-out state binding requires explicit audit=True.
    """
    def __init__(self, dual, *, device):
        self._configuration = (dual.tolerance, dual.rho, dual.dual_lr)
        self.multiplier = torch.zeros((), dtype=torch.float64, device=device)
        self.tolerance = torch.tensor(dual.tolerance, dtype=torch.float64, device=device)
        self.rho = torch.tensor(dual.rho, dtype=torch.float64, device=device)
        self.state_id = None
        self.bound_update = None

    @torch.no_grad()
    def bind(self, dual, state_id, *, audit=False):
        if (dual.tolerance, dual.rho, dual.dual_lr) != self._configuration:
            raise ValueError('primal-dual configuration changed')
        if state_id not in dual.multipliers and not audit:
            raise ValueError('training requires a fitting checkpoint multiplier')
        multiplier = dual.multipliers.get(state_id, 0.)
        if not math.isfinite(multiplier) or multiplier < 0.:
            raise ValueError('finite nonnegative multiplier required')
        # This is deliberately outside capture and on the replay stream.
        self.multiplier.fill_(multiplier)
        self.state_id, self.bound_update = state_id, dual.updates

    def terms(self, mean_kl):
        if self.state_id is None:
            raise RuntimeError('bind current dual state before capture/replay')
        return device_primal_dual_terms(mean_kl, self.multiplier, self.tolerance, self.rho)


def read_terms(terms):
    """Explicit synchronization boundary, required before any optimizer update.

    Do not call inside a CUDA graph. Nonfinite inputs must never silently become
    an accepted zero coefficient via the negative-KL branch.
    """
    if not bool(terms['finite']):
        raise FloatingPointError('nonfinite KL')
    return {name: bool(value) if name == 'feasible' else float(value)
            for name, value in terms.items() if name != 'finite'}


def scale_cotangent(gradient, terms, state_weight=1.):
    """Match production's FP32 scalar promotion and multiplication order.

    Production forms coefficient*state_weight in Python FP64, THEN converts
    that scalar to the gradient dtype. Do NOT cast coefficient before weighting
    or multiply FP64 gradients then cast. This implements
    grad*(coefficient*state_weight), not gradient scaling
    after the actor VJP; use only at the corresponding production boundary.
    """
    coefficient = (terms['kl_coefficient'] * state_weight).to(dtype=gradient.dtype)
    return gradient * coefficient
