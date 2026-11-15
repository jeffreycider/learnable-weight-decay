"""Dense FP32 forward with exact anchored-factor first-order backward.

The stored edited parameter is a frozen value. Its derivative is supplied by the
thin action coordinates, using the same cast derivative as the dense reference.
No dense parameter gradient is constructed. The strict-forward path supports the
explicit offline graph option; compensated GEMMs remain experimental opt-ins.
"""
import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
from rewriter.compensated_dx import current_dx_backend, compensated_dx


class AnchoredLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, p, q, a, b, vector, scale, forward_backend='ieee',dx_weight_high=None,dx_weight_low=None,backward_backend=None):
        ctx.save_for_backward(x, weight, p, q, a, b, vector, scale,dx_weight_high,dx_weight_low)
        ctx.dx_backend = current_dx_backend() if backward_backend is None else backward_backend
        if ctx.dx_backend not in ('ieee','tf32x3','cublas3_large','cublas3_cached'):raise ValueError('unknown explicit dx backend')
        ctx.dx_split_weight=None
        if ctx.dx_backend=='cublas3_cached' and weight.shape[0]>=3072:
            if dx_weight_high is None or dx_weight_low is None:raise ValueError('cached dx requires current realized weight decomposition')
            if any(v.shape!=weight.T.shape or v.dtype!=weight.dtype or v.device!=weight.device or not v.is_contiguous() for v in (dx_weight_high,dx_weight_low)):
                raise ValueError('cached dx decomposition schema differs')
            ctx.dx_split_weight=(dx_weight_high,dx_weight_low)
        if forward_backend=='tf32x3':
            from .compensated_linear import compensated_linear
            return compensated_linear(x,weight,bias)
        if forward_backend=='packed3':
            from .compensated_linear import packed_compensated_linear
            return packed_compensated_linear(x,weight,bias)
        if forward_backend!='ieee':raise ValueError('unknown linear forward backend')
        return F.linear(x, weight, bias)

    @staticmethod
    @once_differentiable
    def backward(ctx, cotangent):
        x,weight,p,q,a,b,vector,scale,_,_=ctx.saved_tensors
        flat=x.reshape(-1,x.shape[-1]);g=cotangent.reshape(-1,cotangent.shape[-1])
        dp=g.T@(flat@b);dq=flat.T@(g@a);dv=g.sum(0)
        ds=(dp*p).sum()+(dq*q).sum()+(dv*vector).sum()
        dx = None
        if ctx.needs_input_grad[0]:
            if ctx.dx_backend == 'tf32x3':
                dx = compensated_dx(g, weight)
            elif ctx.dx_backend in ('cublas3_large','cublas3_cached') and weight.shape[0] >= 3072:
                # Actual-shape measurements favor only the long-K head/MLP-in
                # paths. Transposition and decomposition remain in step cost.
                from .compensated_linear import cublas_three_product_linear
                dx = (cublas_three_product_linear(g,weight.T,split_weight=ctx.dx_split_weight)
                      if ctx.dx_split_weight is not None else cublas_three_product_linear(g,weight.T.contiguous()))
            else:
                dx = g @ weight
            dx = dx.reshape_as(x)
        gradients=(dx,None,None,dp*scale,dq*scale,None,None,dv*scale,ds)
        return gradients+(None,)*(len(ctx.needs_input_grad)-len(gradients))


class AnchoredEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, weight, p, q, a, b, scale):
        ctx.save_for_backward(indices,p,q,a,b,scale)
        return F.embedding(indices,weight)

    @staticmethod
    @once_differentiable
    def backward(ctx,cotangent):
        indices,p,q,a,b,scale=ctx.saved_tensors
        ids=indices.flatten();g=cotangent.reshape(-1,cotangent.shape[-1])
        dp=g.T@b[ids];dq=torch.zeros_like(q).index_add_(0,ids,g@a)
        ds=(dp*p).sum()+(dq*q).sum()
        return None,None,dp*scale,dq*scale,None,None,ds
