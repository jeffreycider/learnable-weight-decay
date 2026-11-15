"""First-order direct-factor kernels used by the optional offline panel backend.

Forward uses an authoritative precomposed stored dense weight. Caller guarantees
it equals cast(base + scale * U @ C @ V.T). The backward implements the cast STE
in real arithmetic, reassociating contractions to avoid constructing dense dW.
It is NOT bitwise the existing dense FP32 dW followed by FP64 projection.
"""
import torch
from torch.autograd.function import once_differentiable


def _validate(stored_weight,u,c,v,scale,chunk_tokens):
    if type(chunk_tokens) is not int or chunk_tokens<1:
        raise ValueError('chunk_tokens must be a positive integer')
    if stored_weight.requires_grad:
        raise ValueError('stored weight must be detached; only factor gradients are supported')
    if scale.numel()!=1 or u.dtype!=c.dtype or u.dtype!=v.dtype or scale.dtype!=u.dtype:
        raise ValueError('scalar scale and uniform factor precision required')
    if stored_weight.shape!=(u.shape[0],v.shape[0]) or c.shape!=(u.shape[1],v.shape[1]):
        raise ValueError('factor shapes do not match stored weight')


class _FactorLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, stored_weight, u, c, v, scale, bias, chunk_tokens):
        _validate(stored_weight,u,c,v,scale,chunk_tokens)
        ctx.chunk_tokens=chunk_tokens
        ctx.save_for_backward(x,stored_weight,u,c,v,scale)
        ctx.bias_dtype=None if bias is None else bias.dtype
        return torch.nn.functional.linear(x,stored_weight,None if bias is None else bias.to(x.dtype))

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        x,w,u,c,v,scale=ctx.saved_tensors
        # Input propagation uses the exact rounded weight from forward.
        dx=dy@w if ctx.needs_input_grad[0] else None
        xf=x.reshape(-1,x.shape[-1])
        df=dy.reshape(-1,dy.shape[-1])
        gu=torch.zeros_like(u);gc=torch.zeros_like(c);gv=torch.zeros_like(v);gs=torch.zeros_like(scale)
        for start in range(0,len(xf),ctx.chunk_tokens):
            xc=xf[start:start+ctx.chunk_tokens].to(u.dtype)
            dc=df[start:start+ctx.chunk_tokens].to(u.dtype)
            xv=xc@v;du=dc@u
            gu.add_((dc.T@xv)@c.T*scale)
            core_credit=du.T@xv
            gc.add_(core_credit*scale)
            gv.add_((xc.T@du)@c*scale)
            # <dL/dC before scale,C> equals the scale VJP. Reuse the
            # small core contraction instead of another token-by-rank GEMM.
            gs.add_((core_credit*c).sum().reshape_as(scale))
        gb=None if ctx.bias_dtype is None else dy.reshape(-1,dy.shape[-1]).sum(0).to(ctx.bias_dtype)
        return dx,None,gu,gc,gv,gs,gb,None


def factor_linear(x,stored_weight,u,c,v,scale,bias=None,*,chunk_tokens=128):
    """First-order factor VJP with the exact supplied dense-weight forward.

Scale may carry a differentiable group/global guard graph. Its returned scalar
cotangent must propagate through that graph, in addition to the direct factor
cotangents. No gradients for the base/stored weight, no higher-order derivatives.
Token chunking bounds FP64 x/dy promotion, not total memory. Contraction and
chunk reduction orders differ from dense FP32 dW; gradients are not bit-exact.
"""
    return _FactorLinear.apply(x,stored_weight,u,c,v,scale,bias,chunk_tokens)


class _FactorEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx,ids,stored_weight,u,c,v,scale,chunk_tokens):
        # Chisel computational embedding is [dimension, vocabulary], whereas
        # torch embedding stores [vocabulary, dimension].
        _validate(stored_weight.T,u,c,v,scale,chunk_tokens)
        ctx.save_for_backward(ids,u,c,v,scale)
        ctx.chunk_tokens=chunk_tokens
        return torch.nn.functional.embedding(ids,stored_weight)

    @staticmethod
    @once_differentiable
    def backward(ctx,dy):
        ids,u,c,v,scale=ctx.saved_tensors
        ids=ids.reshape(-1);df=dy.reshape(-1,dy.shape[-1])
        gu=torch.zeros_like(u);gc=torch.zeros_like(c);gv=torch.zeros_like(v);gs=torch.zeros_like(scale)
        for start in range(0,len(ids),ctx.chunk_tokens):
            selected=ids[start:start+ctx.chunk_tokens]
            dc=df[start:start+ctx.chunk_tokens].to(u.dtype)
            vi=v[selected];du=dc@u
            gu.add_((dc.T@vi)@c.T*scale)
            core_credit=du.T@vi
            gc.add_(core_credit*scale)
            gv.index_add_(0,selected,(du@c)*scale)
            gs.add_((core_credit*c).sum().reshape_as(scale))
        return None,None,gu,gc,gv,gs,None


def factor_embedding(ids,stored_weight,u,c,v,scale,*,chunk_tokens=128):
    """Exact gather forward, direct factor VJP in Chisel embedding orientation.

Stored weight is cast(base + scale * (U @ C @ V.T).T), with U rows
representing dimensions and V rows representing vocabulary entries. Repeated
IDs sum into V credit. No padding_idx, max_norm or frequency scaling semantics
are supported. Factors/scale may be shared across calls; autograd sums credit.

Only token chunks are promoted to factor precision. Output/input adjoints and
factor gradients still exist in full; no peak-memory improvement is claimed.
Chunking, FP64 contraction and repeated-index reductions change floating-point
accumulation order versus a dense FP32 weight gradient. First derivatives only.
"""
    return _FactorEmbedding.apply(ids,stored_weight,u,c,v,scale,chunk_tokens)
