"""Experimental dense-forward, thin-backward linear for adaptive U C V.T edits.

Forward uses the supplied realized stored weights verbatim. Backward computes
factor cotangents directly from activations and adjoints, avoiding dense dW.
This changes reduction order relative to dense autograd and is NOT enabled in
the trainer. GPU numerical, memory and runtime qualification is still required.
FP64 factors currently promote the whole activation/adjoint for their VJP;
large-vocabulary use needs a separately tested bounded-memory implementation.
"""
import torch
from torch.nn import functional as F
from torch.autograd.function import once_differentiable


class AdaptiveLinear(torch.autograd.Function):
    """First-order VJP for stored(W + scale U C V.T), including scale/bias.

Callers must construct realized_weight/bias from these exact factors and scale.
Those detached tensors specify the forward; their construction is deliberately
not retained. The scalar scale may carry a differentiable global guard graph.
No gradients are returned for the frozen realized tensors.
"""
    @staticmethod
    def forward(ctx, x, realized_weight, realized_bias, U, C, V, bias_edit, scale):
        if (x.ndim < 2 or realized_weight.ndim != 2 or U.ndim != 2
                or C.ndim != 2 or V.ndim != 2 or scale.numel() != 1):
            raise ValueError('linear input, matrix factors and scalar scale required')
        n, m = realized_weight.shape
        if (x.shape[-1] != m or U.shape != (n, C.shape[0])
                or V.shape != (m, C.shape[1])):
            raise ValueError('adaptive linear factor shapes differ')
        if (realized_weight.requires_grad or
                (realized_bias is not None and realized_bias.requires_grad)):
            raise ValueError('realized tensors must be detached')
        if (x.dtype not in (torch.float32, torch.float64)
                or realized_weight.dtype != x.dtype
                or U.dtype not in (torch.float32, torch.float64)
                or C.dtype != U.dtype or V.dtype != U.dtype or scale.dtype != U.dtype):
            raise ValueError('explicit matching FP32/FP64 precision required')
        if (realized_bias is None) != (bias_edit is None):
            raise ValueError('bias storage and edit must both be present or absent')
        if bias_edit is not None and (bias_edit.shape != (n,)
                or bias_edit.dtype != U.dtype or realized_bias.shape != (n,)
                or realized_bias.dtype != x.dtype):
            raise ValueError('bias shape or precision differs')
        ctx.save_for_backward(x, realized_weight, U, C, V, bias_edit, scale)
        return F.linear(x, realized_weight, realized_bias)

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        x, weight, U, C, V, bias_edit, scale = ctx.saved_tensors
        dx = dy @ weight if ctx.needs_input_grad[0] else None
        flat_x = x.reshape(-1, x.shape[-1]).to(U.dtype)
        flat_dy = dy.reshape(-1, dy.shape[-1]).to(U.dtype)
        xv = flat_x @ V
        yu = flat_dy @ U
        dU = scale * (flat_dy.T @ (xv @ C.T)) if ctx.needs_input_grad[3] else None
        dC = scale * (yu.T @ xv) if ctx.needs_input_grad[4] else None
        dV = scale * (flat_x.T @ (yu @ C)) if ctx.needs_input_grad[5] else None
        summed_dy = flat_dy.sum(0) if bias_edit is not None else None
        db = scale * summed_dy if ctx.needs_input_grad[6] else None
        ds = None
        if ctx.needs_input_grad[7]:
            ds = ((yu @ C) * xv).sum()
            if bias_edit is not None:
                ds = ds + (summed_dy * bias_edit).sum()
            ds = ds.reshape_as(scale)
        return dx, None, None, dU, dC, dV, db, ds


class AdaptiveEmbedding(torch.autograd.Function):
    """Lookup storage is vocabulary-by-width; U C V.T is width-by-vocabulary."""
    @staticmethod
    def forward(ctx, indices, realized_weight, U, C, V, scale):
        if realized_weight.requires_grad:
            raise ValueError('realized embedding must be detached')
        if (realized_weight.shape != (V.shape[0], U.shape[0])
                or C.shape != (U.shape[1], V.shape[1]) or scale.numel() != 1
                or U.dtype not in (torch.float32, torch.float64)
                or any(t.dtype != U.dtype for t in (C, V, scale))):
            raise ValueError('embedding orientation, factor precision or scale differs')
        ctx.save_for_backward(indices, U, C, V, scale)
        return F.embedding(indices, realized_weight)

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        indices, U, C, V, scale = ctx.saved_tensors
        ids = indices.flatten()
        g = dy.reshape(-1, dy.shape[-1]).to(U.dtype)
        selected = V[ids]
        gu = g @ U
        dU = scale * (g.T @ (selected @ C.T)) if ctx.needs_input_grad[2] else None
        dC = scale * (gu.T @ selected) if ctx.needs_input_grad[3] else None
        dV = None
        if ctx.needs_input_grad[4]:
            dV = torch.zeros_like(V).index_add_(0, ids, scale * (gu @ C))
        ds = ((gu @ C) * selected).sum().reshape_as(scale) if ctx.needs_input_grad[5] else None
        return None, None, dU, dC, dV, ds
