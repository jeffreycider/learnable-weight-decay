"""Opt-in full-gradient FP32 GPT linear experiment, isolated from training defaults.

Forward storage/outputs remain FP32. The first experiment retains the ordinary
IEEE forward and replaces selected input/weight-gradient GEMMs with three TF32
products. Thin-factor VJP qualification does NOT certify these dense gradients
or their subsequent Muon update. The caller must qualify both independently.
"""
from contextlib import contextmanager
from types import MethodType
import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
from .compensated_dx import compensated_dx


@torch.library.custom_op('chisel_experimental::training_tf32x3',mutates_args=())
def training_tf32x3(a:torch.Tensor,b:torch.Tensor)->torch.Tensor:
    """Opaque GEMM boundary so AOTAutograd can compile the surrounding GPT."""
    return compensated_dx(a,b)


@training_tf32x3.register_fake
def _training_tf32x3_fake(a,b):
    return a.new_empty((a.shape[0],b.shape[1]))


@torch.library.custom_op('chisel_experimental::training_dw_splitk',mutates_args=())
def training_dw_splitk(a:torch.Tensor,b:torch.Tensor)->torch.Tensor:
    if a.shape[0]*b.shape[1]<=4_194_304 and a.shape[1]>=4096:
        if a.dtype!=torch.float32 or b.dtype!=torch.float32 or not a.is_cuda or a.device!=b.device:
            raise ValueError('split-K weight gradient requires compatible CUDA FP32 matrices')
        from .kernels.compensated_dw_triton import launch
        return launch(a,b,splits=8)
    return compensated_dx(a,b)


@training_dw_splitk.register_fake
def _training_dw_splitk_fake(a,b):
    return a.new_empty((a.shape[0],b.shape[1]))


class ExperimentalTrainingLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight,bias,dx_backend,dw_backend):
        if x.dtype!=torch.float32 or weight.dtype!=torch.float32:
            raise TypeError('ordinary-GPT GEMM experiment requires FP32 storage')
        if dx_backend not in ('ieee','tf32x3','cublas3_large') or dw_backend not in ('ieee','tf32x3','tf32x3_splitk'):
            raise ValueError('unknown experimental full-gradient GEMM backend')
        ctx.save_for_backward(x,weight)
        ctx.dx_backend=dx_backend;ctx.dw_backend=dw_backend;ctx.has_bias=bias is not None
        return F.linear(x,weight,bias)

    @staticmethod
    @once_differentiable
    def backward(ctx,gradient):
        x,weight=ctx.saved_tensors
        flat=x.reshape(-1,x.shape[-1]);g=gradient.reshape(-1,gradient.shape[-1])
        dx=dw=db=None
        if ctx.needs_input_grad[0]:
            if ctx.dx_backend=='tf32x3':dx=training_tf32x3(g,weight)
            elif ctx.dx_backend=='cublas3_large' and weight.shape[0]>=3072:
                from .compensated_linear import cublas_three_product_linear
                dx=cublas_three_product_linear(g,weight.T.contiguous())
            else:dx=g@weight
            dx=dx.reshape_as(x)
        if ctx.needs_input_grad[1]:
            if ctx.dw_backend=='tf32x3':dw=training_tf32x3(g.T,flat)
            elif ctx.dw_backend=='tf32x3_splitk':dw=training_dw_splitk(g.T,flat)
            else:dw=g.T@flat
        if ctx.has_bias and ctx.needs_input_grad[2]:db=g.sum(0)
        return dx,dw,db,None,None


@contextmanager
def experimental_training_gemms(model,*,dx_backend='ieee',dw_backend='tf32x3'):
    """Temporarily replace only this model's Linear forwards; never global hooks.

    Install before ordinary eager/graph-captured passes. Compiler specialization
    and any sensors must be independently qualified before production adoption.
    Parameters, optimizer state and module names are retained unchanged.
    """
    if dx_backend not in ('ieee','tf32x3','cublas3_large') or dw_backend not in ('ieee','tf32x3','tf32x3_splitk'):
        raise ValueError('unknown experimental full-gradient GEMM backend')
    changed=[]
    def forward(module,x):
        return ExperimentalTrainingLinear.apply(x,module.weight,module.bias,dx_backend,dw_backend)
    try:
        for module in model.modules():
            if isinstance(module,torch.nn.Linear):
                if hasattr(module,'_experimental_training_gemm_active'):
                    raise ValueError('nested full-gradient GEMM installation is unsupported')
                old=module.__dict__.get('forward')
                changed.append((module,old));module._experimental_training_gemm_active=True
                module.forward=MethodType(forward,module)
        yield model
    finally:
        for module,old in reversed(changed):
            if old is None:del module.forward
            else:module.forward=old
            del module._experimental_training_gemm_active
