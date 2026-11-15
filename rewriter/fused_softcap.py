"""Fused FP32 softcap ``15*x*(x.square()+225).rsqrt()`` with autograd-exact rounding.

Eager PyTorch evaluates the softcap as five elementwise kernels with three
full-vocabulary FP32 temporaries (x*x, +225, rsqrt, 15*x, product) and keeps
two of them (c = 15*x and r = rsqrt) alive for backward; backward is another
~seven elementwise kernels. On a 16x1024x50304 microbatch each temporary is
3.07 GiB. This module computes the same values in one pass each way while
reproducing the eager operation sequence exactly, so component outputs and
gradients are intended to be bit-identical (verified by torch.equal in the
tests; on CUDA the only possible deviation is the rsqrt implementation, which
the CUDA test measures in ULPs rather than assumes).

Forward (eager order):   a = x*x ; b = a + 225 ; r = rsqrt(b) ; c = 15*x ; y = c*r
Backward (autograd order for y = c*r):
   grad_c = g*r ; grad_r = g*c
   grad_x1 = grad_c*15                               (MulBackward of c = 15*x)
   grad_b  = (-0.5*grad_r) * ((r*r)*r)                (rsqrt_backward: -0.5*grad*result.pow(3), pow(3) = r*r*r)
   grad_x2 = grad_b * (2*x)                           (pow_backward for square: grad*(2*x.pow(1)))
   grad_x  = grad_x1 + grad_x2                        (two-term accumulation; addition is commutative)
Every operation is a single FP32 rounding; kernels run with FP contraction
disabled so no FMA merges a multiply and an add. Only x is saved for backward.
"""
from __future__ import annotations
import torch
from torch.autograd.function import once_differentiable

try:
    import triton
    import triton.language as tl
    import triton.language.extra.cuda.libdevice as libdevice
except ImportError:  # pragma: no cover
    triton = None

BLOCK = 1024


if triton is not None:
    @triton.jit
    def _softcap_forward(X, Y, N, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < N
        x = tl.load(X + offsets, valid, other=0.)
        a = x * x
        b = a + 225.
        r = libdevice.rsqrt(b)
        c = 15. * x
        tl.store(Y + offsets, c * r, valid)

    @triton.jit
    def _softcap_backward(X, G, GX, N, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < N
        x = tl.load(X + offsets, valid, other=0.)
        g = tl.load(G + offsets, valid, other=0.)
        a = x * x
        b = a + 225.
        r = libdevice.rsqrt(b)
        c = 15. * x
        grad_c = g * r
        grad_r = g * c
        grad_x1 = grad_c * 15.
        grad_b = (-0.5 * grad_r) * ((r * r) * r)
        grad_x2 = grad_b * (2. * x)
        tl.store(GX + offsets, grad_x1 + grad_x2, valid)


def _torch_forward(x):
    a = x * x
    b = a + 225.
    r = b.rsqrt()
    c = 15. * x
    return c * r


def _torch_backward(x, g):
    a = x * x
    b = a + 225.
    r = b.rsqrt()
    c = 15. * x
    grad_c = g * r
    grad_r = g * c
    grad_x1 = grad_c * 15.
    grad_b = (-0.5 * grad_r) * ((r * r) * r)
    grad_x2 = grad_b * (2. * x)
    return grad_x1 + grad_x2


class _FusedSoftcap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if x.dtype != torch.float32:
            raise ValueError('fused softcap is the FP32 head softcap; cast first')
        flat = x.contiguous().reshape(-1)
        use_triton = triton is not None and x.is_cuda
        if use_triton:
            y = torch.empty_like(flat)
            _softcap_forward[(triton.cdiv(flat.numel(), BLOCK),)](flat, y, flat.numel(), BLOCK, enable_fp_fusion=False)
        else:
            y = _torch_forward(flat)
        ctx.save_for_backward(flat)
        ctx.use_triton = use_triton
        ctx.shape = x.shape
        return y.reshape(x.shape)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        flat, = ctx.saved_tensors
        g = grad_output.contiguous().reshape(-1).to(torch.float32)
        if ctx.use_triton:
            gx = torch.empty_like(flat)
            _softcap_backward[(triton.cdiv(flat.numel(), BLOCK),)](flat, g, gx, flat.numel(), BLOCK, enable_fp_fusion=False)
        else:
            gx = _torch_backward(flat, g)
        return gx.reshape(ctx.shape)


def fused_softcap(x):
    """Same values and gradients as ``15*x*(x.square()+225).rsqrt()`` with one kernel each way."""
    return _FusedSoftcap.apply(x)


def reference_softcap(x):
    return 15 * x * (x.square() + 225).rsqrt()
