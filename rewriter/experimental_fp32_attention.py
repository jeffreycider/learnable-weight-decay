"""Explicit FP32-output compensated attention experiment; never enabled by default."""
import torch
from contextlib import contextmanager


@torch.library.custom_op('chisel_experimental::fp32_attention_forward',mutates_args=())
def _forward(q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,scale:float)->tuple[torch.Tensor,torch.Tensor]:
    from .kernels.fp32_attention_triton import forward
    return forward(q,k,v,scale)


@_forward.register_fake
def _fake_forward(q,k,v,scale):
    return torch.empty_like(q,memory_format=torch.contiguous_format),q.new_empty(q.shape[:-1])


@torch.library.custom_op('chisel_experimental::fp32_attention_backward',mutates_args=())
def _backward(q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,o:torch.Tensor,l:torch.Tensor,g:torch.Tensor,scale:float)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    from .kernels.fp32_attention_triton import backward
    return backward(q,k,v,o,l,g,scale)


@_backward.register_fake
def _fake_backward(q,k,v,o,l,g,scale):
    return tuple(torch.empty_like(x,memory_format=torch.contiguous_format) for x in (q,k,v))


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,scale):
        output,normalizer=_forward(q,k,v,scale);ctx.save_for_backward(q,k,v,output,normalizer);ctx.scale=scale
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx,gradient):
        return (*_backward(*ctx.saved_tensors,gradient,ctx.scale),None)


def fp32_attention(q,k,v,*,scale=.12):
    if q.shape!=k.shape or q.shape!=v.shape or q.ndim!=4 or q.shape[-1] not in (64,128):
        raise ValueError('matching B,H,N,D tensors with D64/128 required')
    if any(x.dtype!=torch.float32 or not x.is_cuda or x.device!=q.device for x in (q,k,v)):
        raise ValueError('same-device CUDA FP32 attention inputs required')
    if scale<=0:raise ValueError('positive attention scale required')
    return _Attention.apply(q,k,v,float(scale))


class _FunctionalProxy:
    def __init__(self,original,attention):self.original=original;self.attention=attention
    def __getattr__(self,name):return getattr(self.original,name)
    def scaled_dot_product_attention(self,q,k,v,attn_mask=None,dropout_p=0.,is_causal=False,*,scale=None,enable_gqa=False):
        if attn_mask is not None or dropout_p!=0 or not is_causal or enable_gqa:
            raise ValueError('experimental attention supports causal dropout-free equal-head GPT only')
        return self.attention(q,k,v,scale=q.shape[-1]**-.5 if scale is None else scale)


@contextmanager
def experimental_attention_context(attention=fp32_attention):
    """Isolated-process override of two experimental GPT equation modules.

    Does not patch torch.nn.functional or the baseline module. The caller must
    keep other threads from evaluating these same equation modules concurrently.
    """
    from . import gpt_precision,factorized_fp32
    modules=(gpt_precision,factorized_fp32);originals=[m.F for m in modules]
    if any(isinstance(f,_FunctionalProxy) for f in originals):raise ValueError('nested attention override')
    try:
        for module,original in zip(modules,originals):module.F=_FunctionalProxy(original,attention)
        yield
    finally:
        for module,original in zip(modules,originals):module.F=original
