import torch
import triton
import triton.language as tl


@triton.jit
def _pack(X,Y,K:tl.constexpr,COUNT:tl.constexpr,WEIGHT:tl.constexpr,BLOCK:tl.constexpr):
    index=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    value=tl.load(X+index,index<COUNT,other=0.)
    high=tl.inline_asm_elementwise('cvt.rn.tf32.f32 $0, $1;',constraints='=f,f',args=[value],dtype=tl.float32,is_pure=True,pack=1)
    residual=value-high
    low=tl.inline_asm_elementwise('cvt.rn.tf32.f32 $0, $1;',constraints='=f,f',args=[residual],dtype=tl.float32,is_pure=True,pack=1)
    base=(index//K)*(3*K)+index%K
    # Cross terms accumulate first; the high-high term is last.
    tl.store(Y+base,high if WEIGHT else low,index<COUNT)
    tl.store(Y+base+K,low if WEIGHT else high,index<COUNT)
    tl.store(Y+base+2*K,high,index<COUNT)


def pack(x,*,weight=False):
    if x.ndim!=2 or x.dtype!=torch.float32 or not x.is_cuda or not x.is_contiguous():
        raise ValueError('packing requires contiguous two-dimensional CUDA FP32 input')
    out=torch.empty((x.shape[0],3*x.shape[1]),dtype=x.dtype,device=x.device)
    _pack[(triton.cdiv(x.numel(),1024),)](x,out,x.shape[1],x.numel(),weight,1024,enable_fp_fusion=False)
    return out


@triton.jit
def _split(X,H,L,COUNT:tl.constexpr,BLOCK:tl.constexpr):
    index=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    value=tl.load(X+index,index<COUNT,other=0.)
    high=tl.inline_asm_elementwise('cvt.rn.tf32.f32 $0, $1;',constraints='=f,f',args=[value],dtype=tl.float32,is_pure=True,pack=1)
    residual=value-high
    low=tl.inline_asm_elementwise('cvt.rn.tf32.f32 $0, $1;',constraints='=f,f',args=[residual],dtype=tl.float32,is_pure=True,pack=1)
    tl.store(H+index,high,index<COUNT);tl.store(L+index,low,index<COUNT)


def split(x):
    if x.ndim!=2 or x.dtype!=torch.float32 or not x.is_cuda or not x.is_contiguous():
        raise ValueError('splitting requires contiguous CUDA FP32 matrix')
    high,low=torch.empty_like(x),torch.empty_like(x)
    _split[(triton.cdiv(x.numel(),1024),)](x,high,low,x.numel(),1024,enable_fp_fusion=False)
    return high,low
