"""Explicit experimental FP32 input-gradient GEMM; never changes forward precision.

Importing this module does not import Triton or initialize CUDA. The opt-in is
captured during AnchoredLinear.forward, so a later backward uses the same choice.
This is a hybrid qualification primitive, not the persistent training backend.
"""
from contextlib import contextmanager
from contextvars import ContextVar

import torch


_backend = ContextVar('chisel_experimental_dx_backend', default='ieee')


def current_dx_backend():
    return _backend.get()


@contextmanager
def experimental_dx_backend(backend='tf32x3'):
    """Opt in for forwards in this context; default execution remains IEEE FP32."""
    if backend not in ('ieee', 'tf32x3', 'cublas3_large', 'cublas3_cached'):
        raise ValueError('dx backend must be ieee, tf32x3, cublas3_large or cublas3_cached')
    token = _backend.set(backend)
    try:
        yield
    finally:
        _backend.reset(token)


def compensated_dx(cotangent, weight):
    """Compute [tokens, out] @ [out, in] using three TF32 tensor-core products.

    Reject unsupported inputs rather than silently changing arithmetic. No
    conversion to BF16, split-K atomics, or approximation of the forward occurs.
    """
    if cotangent.ndim != 2 or weight.ndim != 2 or cotangent.shape[1] != weight.shape[0]:
        raise ValueError('dx requires compatible two-dimensional matrices')
    if cotangent.dtype != torch.float32 or weight.dtype != torch.float32:
        raise TypeError('compensated dx requires FP32 operands')
    if not cotangent.is_cuda or not weight.is_cuda or cotangent.device != weight.device:
        raise ValueError('compensated dx requires operands on the same CUDA device')
    if torch.cuda.get_device_capability(cotangent.device)[0] < 8:
        raise ValueError('compensated dx requires Ampere or newer')
    if min(*cotangent.shape, *weight.shape) < 1:
        raise ValueError('compensated dx requires nonempty dimensions')
    from rewriter.kernels.compensated_dx_triton import launch
    return launch(cotangent, weight)
