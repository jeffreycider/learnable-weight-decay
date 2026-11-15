"""Experimental causal FP32 attention: compensated tensor-core dot products.

Inputs, probabilities, accumulators and outputs are FP32. TF32x3 products and
online softmax change arithmetic; this is not IEEE math-SDPA equivalence. No
atomics or BF16/FP16 operand casts. Each backward output tile has one writer.
"""
import torch
import triton
import triton.language as tl

_CONFIG=(32,32,4)


def configure_tiles(m,n,warps):
    global _CONFIG
    if m not in (32,64,128) or n not in (32,64,128) or warps not in (4,8):raise ValueError('unsupported attention tile')
    _CONFIG=(m,n,warps)


@triton.jit
def _forward(Q,K,V,O,L,N:tl.constexpr,D:tl.constexpr,SQ:tl.constexpr,
             BM:tl.constexpr,BN:tl.constexpr):
    m=tl.program_id(0)*BM+tl.arange(0,BM);h=tl.program_id(1)
    d=tl.arange(0,D);n=tl.arange(0,BN)
    q=tl.load(Q+h*N*D+m[:,None]*D+d[None,:],m[:,None]<N,0)*SQ
    maximum=tl.full((BM,),-float('inf'),tl.float32)
    denominator=tl.zeros((BM,),tl.float32);result=tl.zeros((BM,D),tl.float32)
    for start in range(0,tl.minimum((tl.program_id(0)+1)*BM,N),BN):
        cols=start+n
        k=tl.load(K+h*N*D+cols[None,:]*D+d[:,None],cols[None,:]<N,0)*SQ
        scores=tl.dot(q,k,input_precision='tf32x3')
        scores=tl.where((m[:,None]>=cols[None,:])&(cols[None,:]<N),scores,-float('inf'))
        next_max=tl.maximum(maximum,tl.max(scores,1))
        alpha=tl.exp(maximum-next_max);probability=tl.exp(scores-next_max[:,None])
        value=tl.load(V+h*N*D+cols[:,None]*D+d[None,:],cols[:,None]<N,0)
        result=result*alpha[:,None]+tl.dot(probability,value,input_precision='tf32x3')
        denominator=denominator*alpha+tl.sum(probability,1);maximum=next_max
    result=result/denominator[:,None]
    tl.store(O+h*N*D+m[:,None]*D+d[None,:],result,m[:,None]<N)
    tl.store(L+h*N+m,maximum+tl.log(denominator),m<N)


@triton.jit
def _delta(O,G,Delta,N:tl.constexpr,D:tl.constexpr,BM:tl.constexpr):
    m=tl.program_id(0)*BM+tl.arange(0,BM);h=tl.program_id(1);d=tl.arange(0,D)
    o=tl.load(O+h*N*D+m[:,None]*D+d[None,:],m[:,None]<N,0)
    g=tl.load(G+h*N*D+m[:,None]*D+d[None,:],m[:,None]<N,0)
    tl.store(Delta+h*N+m,tl.sum(o*g,1),m<N)


@triton.jit
def _dq(Q,K,V,G,L,Delta,DQ,N:tl.constexpr,D:tl.constexpr,SQ:tl.constexpr,
        BM:tl.constexpr,BN:tl.constexpr):
    m=tl.program_id(0)*BM+tl.arange(0,BM);h=tl.program_id(1);d=tl.arange(0,D);n=tl.arange(0,BN)
    q=tl.load(Q+h*N*D+m[:,None]*D+d[None,:],m[:,None]<N,0)*SQ
    g=tl.load(G+h*N*D+m[:,None]*D+d[None,:],m[:,None]<N,0)
    normalizer=tl.load(L+h*N+m,m<N,0);delta=tl.load(Delta+h*N+m,m<N,0)
    result=tl.zeros((BM,D),tl.float32)
    for start in range(0,tl.minimum((tl.program_id(0)+1)*BM,N),BN):
        cols=start+n
        k=tl.load(K+h*N*D+cols[:,None]*D+d[None,:],cols[:,None]<N,0)*SQ
        v=tl.load(V+h*N*D+cols[None,:]*D+d[:,None],cols[None,:]<N,0)
        scores=tl.dot(q,tl.trans(k),input_precision='tf32x3')
        prob=tl.exp(scores-normalizer[:,None])
        prob=tl.where((m[:,None]>=cols[None,:])&(cols[None,:]<N),prob,0.)
        dp=tl.dot(g,v,input_precision='tf32x3')
        ds=prob*(dp-delta[:,None])
        result+=tl.dot(ds,k,input_precision='tf32x3')
    tl.store(DQ+h*N*D+m[:,None]*D+d[None,:],result*SQ,m[:,None]<N)


@triton.jit
def _dkv(Q,K,V,G,L,Delta,DK,DV,N:tl.constexpr,D:tl.constexpr,SQ:tl.constexpr,
         BM:tl.constexpr,BN:tl.constexpr):
    cols=tl.program_id(0)*BN+tl.arange(0,BN);h=tl.program_id(1);d=tl.arange(0,D);mm=tl.arange(0,BM)
    k=tl.load(K+h*N*D+cols[:,None]*D+d[None,:],cols[:,None]<N,0)*SQ
    v=tl.load(V+h*N*D+cols[:,None]*D+d[None,:],cols[:,None]<N,0)
    dk=tl.zeros((BN,D),tl.float32);dv=tl.zeros((BN,D),tl.float32)
    for start in range((tl.program_id(0)*BN//BM)*BM,N,BM):
        m=start+mm
        q=tl.load(Q+h*N*D+m[:,None]*D+d[None,:],m[:,None]<N,0)*SQ
        g=tl.load(G+h*N*D+m[:,None]*D+d[None,:],m[:,None]<N,0)
        normalizer=tl.load(L+h*N+m,m<N,0);delta=tl.load(Delta+h*N+m,m<N,0)
        scores=tl.dot(q,tl.trans(k),input_precision='tf32x3')
        prob=tl.exp(scores-normalizer[:,None])
        prob=tl.where((m[:,None]>=cols[None,:])&(m[:,None]<N)&(cols[None,:]<N),prob,0.)
        dv+=tl.dot(tl.trans(prob),g,input_precision='tf32x3')
        dp=tl.dot(g,tl.trans(v),input_precision='tf32x3')
        ds=prob*(dp-delta[:,None])
        dk+=tl.dot(tl.trans(ds),q,input_precision='tf32x3')
    tl.store(DK+h*N*D+cols[:,None]*D+d[None,:],dk*SQ,cols[:,None]<N)
    tl.store(DV+h*N*D+cols[:,None]*D+d[None,:],dv,cols[:,None]<N)


def forward(q,k,v,scale):
    q,k,v=(x.contiguous() for x in (q,k,v));b,h,n,d=q.shape
    output=torch.empty_like(q);normalizer=torch.empty((b,h,n),device=q.device,dtype=torch.float32)
    bm,bn,warps=_CONFIG
    _forward[(triton.cdiv(n,bm),b*h)](q,k,v,output,normalizer,n,d,scale**.5,bm,bn,num_warps=warps,num_stages=2,enable_fp_fusion=False)
    return output,normalizer


def backward(q,k,v,output,normalizer,gradient,scale):
    q,k,v,gradient=(x.contiguous() for x in (q,k,v,gradient));b,h,n,d=q.shape
    dq,dk,dv=(torch.empty_like(q) for _ in range(3));delta=torch.empty_like(normalizer)
    bm,bn,warps=_CONFIG
    _delta[(triton.cdiv(n,32),b*h)](output,gradient,delta,n,d,32,num_warps=4,enable_fp_fusion=False)
    _dq[(triton.cdiv(n,bm),b*h)](q,k,v,gradient,normalizer,delta,dq,n,d,scale**.5,bm,bn,num_warps=warps,num_stages=2,enable_fp_fusion=False)
    _dkv[(triton.cdiv(n,bn),b*h)](q,k,v,gradient,normalizer,delta,dk,dv,n,d,scale**.5,bm,bn,num_warps=warps,num_stages=2,enable_fp_fusion=False)
    return dq,dk,dv
