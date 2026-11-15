"""Experimental tiled version of ranking_loss.topk_margin_loss; never enabled implicitly.

Teacher top-k uses torch.topk, preserving the reference's handling of teacher
scores/ties. The outsider scan avoids vocabulary-sized adjusted/scatter buffers.
Backward shares the outsider maximum cotangent across every exact tie, matching
amax (not argmax). GPU arithmetic is explicitly FP32 or FP64 inside tiles,
with contraction disabled and an FP32 student cotangent. CPU uses
the dense semantic reference. Only first derivatives are implemented.
"""
import torch
from torch.autograd.function import once_differentiable
from .ranking_loss import topk_margin_loss
try:
    import triton
    import triton.language as tl
except ImportError:
    triton=None

if triton is not None:
    @triton.jit
    def _forward(T,S,Values,Indices,Loss,Hinges,Member,Outside,Counts,V:tl.constexpr,K:tl.constexpr,A:tl.constexpr,B:tl.constexpr,BK:tl.constexpr,FP64:tl.constexpr):
        dtype=tl.float64 if FP64 else tl.float32
        row=tl.program_id(0);j=tl.arange(0,BK)
        ids=tl.load(Indices+row*K+j,j<K,other=0)
        vals=tl.load(Values+row*K+j,j<K,other=0.).to(dtype)
        next_ids=tl.load(Indices+row*K+j+1,j<K-1,other=0)
        next_vals=tl.load(Values+row*K+j+1,j<K-1,other=0.).to(dtype)
        edited=tl.load(S+row*V+ids).to(dtype)
        edited_next=tl.load(S+row*V+next_ids).to(dtype)
        h=tl.where(j<K-1,tl.maximum(A*(vals-next_vals)-(edited-edited_next),0.),0.)
        tl.store(Hinges+row*K+j,h,j<K)
        boundary_id=tl.load(Indices+row*K+K-1);boundary_t=tl.load(Values+row*K+K-1).to(dtype)
        boundary=tl.load(S+row*V+boundary_id).to(dtype)-A*boundary_t
        columns=tl.arange(0,B);best=tl.full((B,),float('-inf'),dtype);count=tl.zeros((B,),tl.int32)
        for start in range(0,V,B):
            c=start+columns;valid=c<V;inside=tl.full((B,),False,tl.int1)
            for index in tl.static_range(K):
                top=tl.load(Indices+row*K+index);inside=inside|(c==top)
            t=tl.load(T+row*V+c,valid,other=0.).to(dtype);s=tl.load(S+row*V+c,valid,other=0.).to(dtype)
            adjusted=s-A*t;eligible=valid&(~inside);z=tl.where(eligible,adjusted,float('-inf'))
            count=tl.where(z>best,eligible.to(tl.int32),count+(eligible&(z==best)).to(tl.int32))
            best=tl.maximum(best,z)
        maximum=tl.max(best,0);ties=tl.sum(tl.where(best==maximum,count,0),0)
        member=tl.maximum(maximum-boundary,0.)
        tl.store(Outside+row,maximum);tl.store(Counts+row,ties);tl.store(Member+row,member)
        tl.store(Loss+row,(tl.sum(h*h,0)+member*member)/K)

    @triton.jit
    def _backward(T,S,Indices,Hinges,Member,Outside,Counts,Cotangent,Gradient,V:tl.constexpr,K:tl.constexpr,A:tl.constexpr,B:tl.constexpr,FP64:tl.constexpr):
        dtype=tl.float64 if FP64 else tl.float32
        row=tl.program_id(0);columns=tl.arange(0,B);scale=tl.load(Cotangent+row)/K
        member=tl.load(Member+row);maximum=tl.load(Outside+row);ties=tl.load(Counts+row)
        membership_scale=(2.*member)*scale
        for start in range(0,V,B):
            c=start+columns;valid=c<V;value=tl.zeros((B,),dtype);inside=tl.full((B,),False,tl.int1)
            for index in tl.static_range(K):
                top=tl.load(Indices+row*K+index);inside=inside|(c==top)
                current=tl.load(Hinges+row*K+index)
                previous=0.
                if index>0:previous=tl.load(Hinges+row*K+index-1)
                g=(2.*previous)*scale-(2.*current)*scale
                if index==K-1:g=g-membership_scale
                value=tl.where(c==top,g,value)
            t=tl.load(T+row*V+c,valid,other=0.).to(dtype);s=tl.load(S+row*V+c,valid,other=0.).to(dtype)
            value=tl.where((~inside)&((s-A*t)==maximum),membership_scale/ties,value)
            tl.store(Gradient+row*V+c,value,valid)

class _TiledRank(torch.autograd.Function):
    @staticmethod
    def forward(ctx,teacher,student,k,alpha,fp64,return_topk):
        t=teacher.detach().reshape(-1,teacher.shape[-1]).contiguous();s=student.reshape_as(t).contiguous()
        values,indices=t.topk(k,dim=-1,sorted=True);n,v=t.shape
        dtype=torch.float64 if fp64 else s.dtype
        loss=torch.empty(n,device=s.device,dtype=dtype);hinges=torch.empty((n,k),device=s.device,dtype=dtype)
        member=torch.empty_like(loss);outside=torch.empty_like(loss);counts=torch.empty(n,device=s.device,dtype=torch.int32)
        _forward[(n,)](t,s,values,indices,loss,hinges,member,outside,counts,v,k,alpha,1024,triton.next_power_of_2(k),fp64,num_warps=4,enable_fp_fusion=False)
        ctx.save_for_backward(t,s,indices,hinges,member,outside,counts);ctx.k=k;ctx.alpha=alpha;ctx.shape=student.shape;ctx.fp64=fp64
        if return_topk:
            ctx.mark_non_differentiable(values,indices)
            return loss.reshape(student.shape[:-1]),values,indices
        return loss.reshape(student.shape[:-1])
    @staticmethod
    @once_differentiable
    def backward(ctx,cotangent,*unused):
        t,s,indices,hinges,member,outside,counts=ctx.saved_tensors;grad=torch.empty_like(s)
        scale=cotangent.contiguous().reshape(-1)
        _backward[(s.shape[0],)](t,s,indices,hinges,member,outside,counts,scale,grad,s.shape[1],ctx.k,ctx.alpha,1024,ctx.fp64,num_warps=4,enable_fp_fusion=False)
        return None,grad.reshape(ctx.shape),None,None,None,None

def tiled_topk_margin_loss(teacher,student,*,k=10,margin_fraction=.5,arithmetic="fp32",return_teacher_topk=False):
    """Per-token squared-logit penalty, same formula as the dense reference.

    This standalone primitive still materializes its student-logit gradient;
    a combined KL/ranking epilogue is a separate optimization. No objective
    weights or scientific thresholds are chosen by this helper. arithmetic="fp64"
    promotes inside the tile and returns FP64 losses, casting the final student
    cotangent once to FP32. It accepts FP32 teacher logits or FP64 log-probs;
    additive row constants cancel mathematically but need numerical qualification.
    return_teacher_topk optionally returns (loss, (values, indices)) from the SAME
    detached teacher selection for read-only diagnostics; it is not a cache.
    """
    if teacher.shape!=student.shape or teacher.ndim<2:raise ValueError('matching token/vocabulary shapes required')
    if not 1<=k<teacher.shape[-1]:raise ValueError('k must be positive and smaller than the vocabulary')
    if not 0<margin_fraction<=1:raise ValueError('margin_fraction must lie in (0, 1]')
    if arithmetic not in ("fp32","fp64"):raise ValueError('arithmetic must be fp32 or fp64')
    if teacher.device!=student.device:raise ValueError('teacher and student device must match')
    if not student.is_cuda:
        t,s=(teacher.double(),student.double()) if arithmetic=="fp64" else (teacher,student)
        loss=topk_margin_loss(t,s,k=k,margin_fraction=margin_fraction)
        if return_teacher_topk:return loss,t.detach().reshape(-1,t.shape[-1]).topk(k,dim=-1,sorted=True)
        return loss
    if student.dtype!=torch.float32 or teacher.dtype not in (torch.float32,torch.float64):raise ValueError('CUDA tiled ranking requires FP32 student and FP32/FP64 teacher')
    if arithmetic=="fp32" and teacher.dtype!=torch.float32:raise ValueError('FP64 teacher requires FP64 arithmetic')
    if triton is None:raise RuntimeError('Triton is required for CUDA tiled ranking')
    if student.numel()==0:raise ValueError('nonempty tokens required')
    result=_TiledRank.apply(teacher,student,k,float(margin_fraction),arithmetic=="fp64",return_teacher_topk)
    return (result[0],(result[1],result[2])) if return_teacher_topk else result
