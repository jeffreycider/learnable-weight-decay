"""Residual SOAP-Shampoo with the published PR395 zero-residual path.

The baseline raw direction is evaluated by upstream's direct polynomial action,
not explicit factor multiplication. Only additive residual terms use factors.
No outer polar is present. FP64 is for isolated derivative tests only.
"""
from dataclasses import dataclass
import math
import torch
from .soap_residual import FrozenMap


@dataclass(frozen=True)
class PublishedMap(FrozenMap):
    base_raw: torch.Tensor | None = None


def committed_pr395_proposal(record, hl, hr):
    r = record
    if r.base_raw is None:
        raise ValueError('published PR395 map requires upstream direct Shampoo direction')
    ql, qr = r.left_scale * hl, r.right_scale * hr
    # Expand only the residual. This keeps the upstream operation ordering at
    # H=0, while retaining nonzero action derivatives at zero initialization.
    update = (r.base_raw + (ql @ r.direction) @ r.right
              + (r.left @ r.direction) @ qr + (ql @ r.direction) @ qr)
    update = update * (math.sqrt(r.direction.shape[0]) / update.norm().clamp_min(1e-20))
    if r.spectral_scale is not None:
        rotated = (r.spectral_row.mT @ update @ r.spectral_col) * r.spectral_scale
        shaped = r.spectral_row @ rotated @ r.spectral_col.mT
        update = shaped * (update.norm() / shaped.norm().clamp_min(1e-30))
    p = r.weight if r.weight.dtype == torch.float64 else r.weight.float()
    coeff = (update * p).sum() / (p * p).sum().clamp_min(1e-12)
    radial = coeff * p
    scale = torch.where(coeff < 0, coeff.new_tensor(r.outward_scale), coeff.new_tensor(r.inward_scale))
    update = update - radial + scale * radial
    pnorm = p.norm().clamp_min(1e-8)
    if r.rowfloor:
        factor = (r.target_uw * p.norm(dim=1, keepdim=True).clamp_min(1e-8)
                  / update.norm(dim=1, keepdim=True).clamp_min(1e-8)).clamp_min(1).pow(r.rowfloor_rho)
        update = update * factor
    else:
        unorm = update.norm().clamp_min(1e-8)
        update = update * torch.where(unorm / pnorm < r.target_uw,
                                      r.target_uw * pnorm / unorm, torch.ones_like(pnorm))
    # Preserve upstream multiply/divide association and add(alpha=...), including
    # stored-weight casts. Multiplying lr into update first is not equivalent FP.
    radius = (pnorm + (-r.lr * (update * p).sum() / pnorm)).clamp_min(1e-8)
    proposal = torch.add(r.weight, update, alpha=-r.lr).to(r.weight.dtype)
    stats = proposal if proposal.dtype == torch.float64 else proposal.float()
    proposal = proposal * (radius / stats.norm().clamp_min(1e-8)).to(r.weight.dtype)
    if r.cwd > 0:
        mask = (update * p > 0).to(r.weight.dtype)
        proposal = proposal * (1 - r.lr * r.cwd * mask)
    return proposal
