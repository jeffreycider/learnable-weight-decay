"""Experimental FP32 GEMM via six BF16 tensor-core products.

Each FP32 operand is decomposed into three BF16 residual components. Six
products retain all terms through second order; third/fourth-order products
are omitted. This is approximate FP32 emulation, never a strict IEEE backend.
The leading and correction accumulators stay separate until the final sum.
"""
import torch
import triton
import triton.language as tl


@triton.autotune(configs=[
    triton.Config({'BM':64,'BN':64,'BK':32},num_warps=4,num_stages=3),
    triton.Config({'BM':128,'BN':64,'BK':32},num_warps=4,num_stages=3),
    triton.Config({'BM':64,'BN':128,'BK':32},num_warps=4,num_stages=3),
    triton.Config({'BM':64,'BN':64,'BK':64},num_warps=4,num_stages=3),
],key=['M','N','K','AS0','AS1','BS0','BS1','COMPENSATE'])
@triton.jit
def _gemm(A,B,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
          AS0:tl.constexpr,AS1:tl.constexpr,BS0:tl.constexpr,BS1:tl.constexpr,
          COMPENSATE:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pid=tl.program_id(0);nm=tl.cdiv(M,BM);nn=tl.cdiv(N,BN)
    group=pid//(8*nn);first=group*8;gm=tl.minimum(nm-first,8)
    pm=first+(pid%(8*nn))%gm;pn=(pid%(8*nn))//gm
    rows=pm*BM+tl.arange(0,BM);cols=pn*BN+tl.arange(0,BN)
    ks=tl.arange(0,BK)
    main=tl.zeros((BM,BN),tl.float32);correction=tl.zeros((BM,BN),tl.float32)
    rounding=tl.zeros((BM,BN),tl.float32)
    for block in range(tl.cdiv(K,BK)):
        k=block*BK+ks
        a=tl.load(A+rows[:,None]*AS0+k[None,:]*AS1,(rows[:,None]<M)&(k[None,:]<K),other=0.)
        b=tl.load(B+k[:,None]*BS0+cols[None,:]*BS1,(k[:,None]<K)&(cols[None,:]<N),other=0.)
        a0=a.to(tl.bfloat16);b0=b.to(tl.bfloat16)
        ar=a-a0.to(tl.float32);br=b-b0.to(tl.float32)
        a1=ar.to(tl.bfloat16);b1=br.to(tl.bfloat16)
        a2=(ar-a1.to(tl.float32)).to(tl.bfloat16)
        b2=(br-b1.to(tl.float32)).to(tl.bfloat16)
        correction=tl.dot(a2,b0,correction)
        correction=tl.dot(a1,b1,correction)
        correction=tl.dot(a0,b2,correction)
        correction=tl.dot(a1,b0,correction)
        correction=tl.dot(a0,b1,correction)
        if COMPENSATE:
            partial=tl.dot(a0,b0)
            y=partial-rounding
            updated=main+y
            rounding=(updated-main)-y
            main=updated
        else:
            main=tl.dot(a0,b0,main)
    tl.store(C+rows[:,None]*N+cols[None,:],main+correction,(rows[:,None]<M)&(cols[None,:]<N))


def launch(a,b,*,compensate=False):
    m,k=a.shape;n=b.shape[1]
    result=torch.empty((m,n),device=a.device,dtype=torch.float32)
    with torch.cuda.device(a.device):
        _gemm[lambda meta:(triton.cdiv(m,meta['BM'])*triton.cdiv(n,meta['BN']),)](
            a,b,result,m,n,k,*a.stride(),*b.stride(),compensate,enable_fp_fusion=False)
    return result
