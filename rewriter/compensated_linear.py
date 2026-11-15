"""Experimental FP32-storage/output linear algebra using compensated TF32.

This changes forward arithmetic. It is not the strict IEEE baseline and is not
enabled by importing this module. Three tensor-core products approximate the
FP32 product; small finite edit responses require independent qualification.
"""
import torch
from functools import lru_cache
from .compensated_dx import compensated_dx


def bf16_six_product_linear(x,weight,bias=None,*,compensate=False):
    """Experimental six-product FP32 emulation; no single BF16 weight cast.

    Only isolated forward primitives/custom backwards may call this helper.
    Normal FP32 ranges are the qualification scope: BF16 subnormal handling
    and overflow during residual decomposition are not certified.
    """
    if x.ndim<2 or weight.ndim!=2 or x.shape[-1]!=weight.shape[1]:
        raise ValueError('linear operands have incompatible shapes')
    if x.dtype!=torch.float32 or weight.dtype!=torch.float32:
        raise TypeError('six-product emulation requires FP32 operands')
    if not x.is_cuda or x.device!=weight.device:
        raise ValueError('six-product emulation requires one CUDA device')
    if bias is not None and (bias.shape!=(weight.shape[0],) or bias.dtype!=x.dtype or bias.device!=x.device):
        raise ValueError('bias must match output dimension, dtype and device')
    if torch.is_grad_enabled() and any(v.requires_grad for v in (x,weight,bias) if v is not None):
        raise ValueError('forward primitive needs explicit custom backward or no_grad')
    from .kernels.compensated_bf16_triton import launch
    out=launch(x.reshape(-1,x.shape[-1]),weight.T,compensate=compensate)
    if bias is not None:out.add_(bias)
    return out.reshape(*x.shape[:-1],weight.shape[0])


def compensated_linear(x, weight, bias=None):
    """Explicit forward-only primitive for isolated compensated-GEMM trials."""
    if x.ndim < 2 or weight.ndim != 2 or x.shape[-1] != weight.shape[1]:
        raise ValueError('linear operands have incompatible shapes')
    if bias is not None and (bias.shape != (weight.shape[0],) or bias.dtype != weight.dtype
                             or bias.device != weight.device):
        raise ValueError('bias must match the output dimension, dtype and device')
    if torch.is_grad_enabled() and any(v.requires_grad for v in (x,weight,bias) if v is not None):
        raise ValueError('forward primitive needs explicit custom backward or no_grad')
    output=compensated_dx(x.reshape(-1,x.shape[-1]),weight.T)
    if bias is not None:output.add_(bias)
    return output.reshape(*x.shape[:-1],weight.shape[0])


@lru_cache(maxsize=1)
def _cublas():
    """Use the runner's existing cuBLAS; no CUDA toolkit/compiler is needed."""
    import ctypes
    from pathlib import Path
    library=Path(torch.__file__).resolve().parents[1]/'nvidia/cublas/lib/libcublas.so.12'
    if not library.is_file():raise RuntimeError('qualified cuBLAS12 runner library is missing')
    lib=ctypes.CDLL(str(library));pointer=ctypes.c_void_p;integer=ctypes.c_int
    for name in ('cublasGetPointerMode','cublasSetPointerMode'):
        if not hasattr(lib,name):setattr(lib,name,getattr(lib,name+'_v2'))
    for name in ('cublasGetMathMode','cublasGetPointerMode'):
        fn=getattr(lib,name);fn.argtypes=[pointer,ctypes.POINTER(integer)];fn.restype=integer
    for name in ('cublasSetMathMode','cublasSetPointerMode'):
        fn=getattr(lib,name);fn.argtypes=[pointer,integer];fn.restype=integer
    lib.cublasGemmEx.argtypes=[pointer,integer,integer,integer,integer,integer,
        pointer,pointer,integer,integer,pointer,integer,integer,pointer,pointer,
        integer,integer,integer,integer]
    lib.cublasGemmEx.restype=integer
    return lib


def _packed_mm(x,weight):
    import ctypes
    if (x.ndim!=2 or weight.ndim!=2 or x.shape[1]!=weight.shape[1]
        or x.dtype!=torch.float32 or weight.dtype!=torch.float32
        or not x.is_cuda or x.device!=weight.device or not x.is_contiguous() or not weight.is_contiguous()):
        raise ValueError('compatible contiguous CUDA FP32 matrices required')
    lib=_cublas()
    def check(status):
        if status:raise RuntimeError(f'cuBLAS compensated GEMM status {status}')
    with torch.cuda.device(x.device):
        handle=torch.cuda.current_blas_handle()
        old_math,old_pointer=ctypes.c_int(),ctypes.c_int()
        check(lib.cublasGetMathMode(handle,ctypes.byref(old_math)))
        check(lib.cublasGetPointerMode(handle,ctypes.byref(old_pointer)))
        out=torch.empty((x.shape[0],weight.shape[0]),dtype=x.dtype,device=x.device)
        alpha,beta=ctypes.c_float(1.),ctypes.c_float(0.)
        try:
            # CUDA12 header enums: TF32_TENSOR_OP_MATH=3, POINTER_MODE_HOST=0,
            # COMPUTE_32F_FAST_TF32=77, GEMM_DEFAULT_TENSOR_OP=99, R_32F=0.
            check(lib.cublasSetMathMode(handle,3));check(lib.cublasSetPointerMode(handle,0))
            check(lib.cublasGemmEx(handle,1,0,weight.shape[0],x.shape[0],x.shape[1],
                ctypes.byref(alpha),weight.data_ptr(),0,weight.shape[1],x.data_ptr(),0,x.shape[1],
                ctypes.byref(beta),out.data_ptr(),0,weight.shape[0],77,99))
        finally:
            # Preserve the exact teacher/default execution even after failure.
            math_status=lib.cublasSetMathMode(handle,old_math.value)
            pointer_status=lib.cublasSetPointerMode(handle,old_pointer.value)
            check(math_status);check(pointer_status)
    return out


def pack_compensated_weight(weight):
    from .kernels.compensated_pack_triton import pack
    return pack(weight,weight=True)


def packed_compensated_linear(x,weight,bias=None,*,packed_weight=None):
    """One cuBLAS TF32 GEMM over three packed terms, FP32 storage/output.

    Decomposition/reassembly costs must remain in any reported step timing.
    Supplying prepacked weights is only valid for those exact weight values.
    """
    if torch.is_grad_enabled() and any(v.requires_grad for v in (x,weight,bias) if v is not None):
        raise ValueError('forward primitive needs explicit custom backward or no_grad')
    from .kernels.compensated_pack_triton import pack
    packed_x=pack(x.reshape(-1,x.shape[-1]).contiguous())
    if packed_weight is None:packed_weight=pack_compensated_weight(weight)
    if packed_weight.shape!=(weight.shape[0],3*weight.shape[1]):raise ValueError('packed weight shape differs')
    out=_packed_mm(packed_x,packed_weight)
    if bias is not None:out.add_(bias)
    return out.reshape(*x.shape[:-1],weight.shape[0])


def cublas_three_product_linear(x,weight,bias=None,*,split_weight=None):
    """Three cuBLAS tensor-core products; useful when output is much smaller than K."""
    if torch.is_grad_enabled() and any(v.requires_grad for v in (x,weight,bias) if v is not None):
        raise ValueError('forward primitive needs explicit custom backward or no_grad')
    from .kernels.compensated_pack_triton import split
    high,low=split(x.reshape(-1,x.shape[-1]).contiguous())
    wh,wl=split(weight.contiguous()) if split_weight is None else split_weight
    out=_packed_mm(low,wh)
    out.add_(_packed_mm(high,wl));out.add_(_packed_mm(high,wh))
    if bias is not None:out.add_(bias)
    return out.reshape(*x.shape[:-1],weight.shape[0])
