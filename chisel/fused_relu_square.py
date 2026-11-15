"""Optional FP32 ReLU-squared fusion; ported from the prior systems experiments."""
from __future__ import annotations
import torch
from torch.autograd.function import once_differentiable

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None

BLOCK = 1024


if triton is not None:
    @triton.jit
    def _relu_square_forward(X, Z, N, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < N
        x = tl.load(X + offsets, valid, other=0.)
        y = tl.where(x < 0., 0., x)          # NaN and -0.0 pass through like torch.relu
        tl.store(Z + offsets, y * y, valid)

    @triton.jit
    def _relu_square_backward(X, G, GX, N, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < N
        x = tl.load(X + offsets, valid, other=0.)
        g = tl.load(G + offsets, valid, other=0.)
        y = tl.where(x < 0., 0., x)
        gy = g * (2. * y)
        tl.store(GX + offsets, tl.where(y <= 0., 0., gy), valid)   # threshold_backward: keep unless result <= 0 (NaN kept)


def _torch_forward(x):
    y = torch.relu(x)
    return y * y


def _torch_backward(x, g):
    y = torch.relu(x)
    gy = g * (2. * y)
    return torch.where(y <= 0., torch.zeros((), dtype=gy.dtype, device=gy.device), gy)


class _FusedReluSquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if x.dtype != torch.float32:
            raise ValueError('fused relu-square targets the FP32 MLP activation; cast first')
        flat = x.contiguous().reshape(-1)
        use_triton = triton is not None and x.is_cuda
        if use_triton:
            z = torch.empty_like(flat)
            _relu_square_forward[(triton.cdiv(flat.numel(), BLOCK),)](flat, z, flat.numel(), BLOCK, enable_fp_fusion=False)
        else:
            z = _torch_forward(flat)
        ctx.save_for_backward(flat)
        ctx.use_triton = use_triton
        ctx.shape = x.shape
        return z.reshape(x.shape)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        flat, = ctx.saved_tensors
        g = grad_output.contiguous().reshape(-1).to(torch.float32)
        if ctx.use_triton:
            gx = torch.empty_like(flat)
            _relu_square_backward[(triton.cdiv(flat.numel(), BLOCK),)](flat, g, gx, flat.numel(), BLOCK, enable_fp_fusion=False)
        else:
            gx = _torch_backward(flat, g)
        return gx.reshape(ctx.shape)


def fused_relu_square(x):
    """Same values and gradients as ``x.relu().square()`` with one kernel each way."""
    return _FusedReluSquare.apply(x)


def reference_relu_square(x):
    return x.relu().square()
