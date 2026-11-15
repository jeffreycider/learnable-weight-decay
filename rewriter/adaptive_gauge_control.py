"""Algebra-only representability controls, never a learned update policy.

For X Y.T with k<=L, thin QR of L-column augmentations gives U,V and an
L-by-L core C. This exhibits membership in the adaptive action family without
an SVD or a dense projector. It does not prove that a shared actor learns the
coefficients, satisfies its radius, or preserves floating-point predictions.
"""
import torch

from .adaptive_action import AdaptiveFactors


def orthonormal_product(left, right, *, rank=None):
    """Represent left @ right.T with orthonormal adaptive rank-L factors.

    Rank-deficient and zero products are valid. Padding selects a deterministic
    orthonormal completion; it is a construction, not a differentiable rank
    selection procedure or a suggested training initializer. No global RNG is
    consumed. Input width is the known algebraic rank bound, not estimated rank.
    """
    if (not isinstance(left,torch.Tensor) or not isinstance(right,torch.Tensor)
            or left.ndim!=2 or right.ndim!=2 or left.shape[1]!=right.shape[1]
            or left.dtype!=right.dtype or left.device!=right.device
            or left.dtype not in (torch.float32,torch.float64)):
        raise ValueError('matching FP32/FP64 thin matrix factors required')
    k=left.shape[1];rank=k if rank is None else rank
    if type(rank) is not int or not 0<k<=rank<=min(left.shape[0],right.shape[0]):
        raise ValueError('require 0 < thin width <= rank <= physical dimensions')
    def qr(value):
        if rank>k:
            padding=value.new_zeros((len(value),rank-k))
            positions=torch.arange(rank-k,device=value.device)
            padding[positions,positions]=1
            value=torch.cat((value,padding),dim=1)
        q,r=torch.linalg.qr(value,mode='reduced')
        signs=torch.where(r.diagonal()<0,-torch.ones_like(r.diagonal()),torch.ones_like(r.diagonal()))
        return q*signs, signs[:,None]*r[:,:k]
    u,rx=qr(left);v,ry=qr(right)
    return AdaptiveFactors(u,rx@ry.T,v)
