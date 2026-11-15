"""Deterministic split-K FP32 weight-gradient experiment, no atomic reductions.

Small output matrices with long token reductions underfill the GPU in the
ordinary three-product GEMM. Independent K partitions expose more parallelism;
their FP32 partial outputs are reduced in FP64, then stored in FP32. This changes
summation order and needs dense-gradient/Muon/functional qualification.
"""
import torch
import triton
import triton.language as tl


@triton.autotune(configs=[
    triton.Config({'BM':64,'BN':64,'BK':32},num_warps=4,num_stages=3),
    triton.Config({'BM':128,'BN':64,'BK':32},num_warps=4,num_stages=3),
    triton.Config({'BM':64,'BN':128,'BK':32},num_warps=4,num_stages=3),
    triton.Config({'BM':128,'BN':128,'BK':32},num_warps=8,num_stages=3),
],key=['M','N','K','AS0','AS1','BS0','BS1','SPLITS'])
@triton.jit
def _partial(A,B,P,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
             AS0:tl.constexpr,AS1:tl.constexpr,BS0:tl.constexpr,BS1:tl.constexpr,
             SPLITS:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    tile=tl.program_id(0);part=tl.program_id(1);nn=tl.cdiv(N,BN)
    rows=(tile//nn)*BM+tl.arange(0,BM);cols=(tile%nn)*BN+tl.arange(0,BN)
    chunk=tl.cdiv(K,SPLITS*BK)*BK;ks=tl.arange(0,BK)
    total=tl.zeros((BM,BN),tl.float32)
    for block in range(tl.cdiv(chunk,BK)):
        k=part*chunk+block*BK+ks
        a=tl.load(A+rows[:,None]*AS0+k[None,:]*AS1,(rows[:,None]<M)&(k[None,:]<K),other=0.)
        b=tl.load(B+k[:,None]*BS0+cols[None,:]*BS1,(k[:,None]<K)&(cols[None,:]<N),other=0.)
        total=tl.dot(a,b,total,input_precision='tf32x3')
    tl.store(P+part*M*N+rows[:,None]*N+cols[None,:],total,(rows[:,None]<M)&(cols[None,:]<N))


@triton.jit
def _reduce(P,C,COUNT:tl.constexpr,SPLITS:tl.constexpr,BLOCK:tl.constexpr):
    offsets=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    parts=tl.arange(0,SPLITS)
    value=tl.load(P+parts[:,None]*COUNT+offsets[None,:],offsets[None,:]<COUNT,other=0.).to(tl.float64)
    result=tl.sum(value,axis=0).to(tl.float32)
    tl.store(C+offsets,result,offsets<COUNT)


def launch(a,b,*,splits=8):
    m,k=a.shape;n=b.shape[1]
    partial=torch.empty((splits,m,n),dtype=torch.float32,device=a.device)
    result=torch.empty((m,n),dtype=torch.float32,device=a.device)
    with torch.cuda.device(a.device):
        _partial[lambda meta:(triton.cdiv(m,meta['BM'])*triton.cdiv(n,meta['BN']),splits)](
            a,b,partial,m,n,k,*a.stride(),*b.stride(),splits,enable_fp_fusion=False)
        _reduce[(triton.cdiv(m*n,256),)](partial,result,m*n,splits,256,enable_fp_fusion=False)
    return result
