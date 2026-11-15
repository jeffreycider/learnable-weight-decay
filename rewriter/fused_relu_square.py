"""Fused FP32 ``x.relu().square()`` forward/backward with autograd-exact rounding.

The NanoGPT MLP activation is ``x.relu().square()``. Eagerly that is two
forward kernels (relu writes y, square writes y*y) keeping y alive for
backward, and three backward kernels (``2*y`` from pow_backward of square,
the upstream multiply ``grad*(2*y)``, then ``threshold_backward`` of relu on the
saved result). On a 16x1024x3072 FP32 hidden (201 MB) that is ~5 full passes
per layer per microbatch. This module computes the same values in one kernel
each way, reproducing the eager operation sequence exactly:

  forward : y = where(x < 0, 0, x) ; z = y*y              (one FP32 rounding for z)
  backward: y = where(x < 0, 0, x) ; gy = g * (2*y) ; gx = where(y <= 0, 0, gy)

``2*y`` is exact in binary floating point and ``g*(2*y)`` is a single
rounding, identical to eager's ``grad * (2 * y.pow(1))``. Special values
follow eager exactly (measured on CPU): relu(NaN) = NaN, relu(-0.0) = -0.0,
relu(inf) = inf; the backward keeps the gradient wherever the saved relu
output is NOT <= 0 (``threshold_backward`` semantics), so NaN inputs give NaN
gradients and +-0 inputs give 0. Memory: eager saves two references to ONE
relu-output storage; the fused version saves the input of the same size, so
saved-tensor memory is unchanged (Astra review) - the only effect is fewer
elementwise passes. Kernels run with FP contraction disabled. Bit-exactness is
asserted (0 ULP) on CPU and in the CUDA test; no xfail. Honest expectation
from the owner's graph profile: about one percent of the 64-sequence replay.
Not wired anywhere; opt-in component for the owner's factorized/graph path.
"""
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
