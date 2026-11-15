"""First-order supervision from a frozen, explicitly supplied weight edit.

This optional diagnostic is not the preservation loss or a new edit policy.
Targets may be separately verified algebraic transformations. Matching their
factors individually would incorrectly penalize changes of low-rank basis;
we instead compare the represented edits in their combined thin spans.
No GPT evaluation, function-preservation claim, or trainer activation occurs
here. Callers must supply both edits in the same physical parameter units.
"""
import torch
from torch.autograd.function import once_differentiable

from .adaptive_action import AdaptiveAction, AdaptiveFactors


class _FactorDistance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, c, v, target_u, target_c, target_v):
        # A joint thin span avoids subtracting three nearly equal squared
        # norms. QR is only a numerical coordinate system for this evaluation;
        # backward below differentiates the original ambient squared distance.
        left = torch.linalg.qr(torch.cat((u, target_u), dim=1), mode='reduced').Q
        right = torch.linalg.qr(torch.cat((v, target_v), dim=1), mode='reduced').Q
        difference = ((left.T @ u) @ c @ (right.T @ v).T
                      - (left.T @ target_u) @ target_c @ (right.T @ target_v).T)
        ctx.save_for_backward(u, c, v, left, right, difference)
        return difference.square().sum()

    @staticmethod
    @once_differentiable
    def backward(ctx, cotangent):
        u, c, v, left, right, difference = ctx.saved_tensors
        # E = left @ difference @ right.T, never materialized.
        ev = left @ (difference @ (right.T @ v))
        etu = right @ (difference.T @ (left.T @ u))
        scale = 2 * cotangent
        return (scale * (ev @ c.T), scale * (u.T @ ev),
                scale * (etu @ c), None, None, None)


def matrix_edit_distance(predicted: AdaptiveFactors, target: AdaptiveFactors):
    """Squared Frobenius distance; frozen target; first-order gradients only.

    Ranks may differ. Both factors use the computational matrix orientation, not
    the embedding table's storage orientation. The temporary difference core is
    min(n, r_pred+r_target) by min(m, s_pred+s_target); for ordinary thin ranks
    this avoids an n-by-m edit. Full-span/wide factors can reach dense core size.
"""
    if not isinstance(predicted, AdaptiveFactors) or not isinstance(target, AdaptiveFactors):
        raise TypeError('adaptive factors required')
    if (predicted.U.shape[0] != target.U.shape[0]
            or predicted.V.shape[0] != target.V.shape[0]):
        raise ValueError('target and predicted matrix dimensions differ')
    tensors = (predicted.U, predicted.C, predicted.V, target.U, target.C, target.V)
    if any(t.dtype != predicted.U.dtype or t.device != predicted.U.device for t in tensors):
        raise ValueError('target and predicted precision/device must match')
    if predicted.U.dtype not in (torch.float32, torch.float64):
        raise ValueError('FP32 or FP64 factors required')
    if any(t.requires_grad for t in tensors[3:]):
        raise ValueError('edit targets must be frozen')
    return _FactorDistance.apply(*tensors)


def edit_target_terms(predicted: AdaptiveAction, target: AdaptiveAction):
    """Per-parameter squared edit errors, with no implicit weights or averaging.

Explicit terms let a diagnostic declare its normalization independently of
the user's equal-group percentage norm reward. This function does not change
that reward. Read-only parameters should be excluded from both actions.
"""
    if not isinstance(predicted, AdaptiveAction) or not isinstance(target, AdaptiveAction):
        raise TypeError('adaptive actions required')
    if predicted.matrices.keys() != target.matrices.keys() or predicted.vectors.keys() != target.vectors.keys():
        raise ValueError('target and predicted inventories differ')
    terms = {name: matrix_edit_distance(value, target.matrices[name])
             for name, value in predicted.matrices.items()}
    for name, value in predicted.vectors.items():
        teacher = target.vectors[name]
        if (value.shape != teacher.shape or value.dtype != teacher.dtype
                or value.device != teacher.device or teacher.requires_grad):
            raise ValueError('vector targets must match and be frozen')
        terms[name] = (value - teacher).square().sum()
    return terms
