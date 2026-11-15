"""Experimental FP64 relative-gap ranking, separate from the absolute pilot.

One vocabulary scan selects the worst fractional outsider and accumulates its
inverse-gap cotangent. Exact ties are classes, never epsilon-clamped gaps. The
boundary selection keeps torch.topk's exact cutoff subgradient. No trainer or
production dispatch uses this primitive until separate numerical qualification.
"""
import torch
from torch.autograd.function import once_differentiable
from . import tiled_tie_ranking as absolute
triton=absolute.triton
if triton is not None:
 import triton.language as tl
 from .tiled_tie_ranking import _class_min,_class_max,_boundary_scores,_selected_backward

 @triton.jit
 def _relative_forward(T,S,Values,Indices,BoundaryMax,Cutoff,Loss,Hinges,Member,
                       Outside,OutsideCount,OutsideInverse,BoundaryCount,BoundaryHinge,
                       V:tl.constexpr,K:tl.constexpr,A:tl.constexpr,B:tl.constexpr,BK:tl.constexpr):
  row=tl.program_id(0);j=tl.arange(0,BK)
  keys=tl.load(Values+row*K+j,j<K,other=float('-inf')).to(tl.float64)
  ids=tl.load(Indices+row*K+j,j<K,other=0);edited=tl.load(S+row*V+ids).to(tl.float64)
  smallest=_class_min(keys,tl.where(j<K,edited,float('inf')),BK)
  largest=_class_max(keys,tl.where(j<K,edited,float('-inf')),BK)
  threshold=tl.load(Values+row*K+K-1).to(tl.float64);boundary_max=tl.load(BoundaryMax+row)
  largest=tl.where(keys==threshold,boundary_max,largest)
  nxt=tl.minimum(j+1,BK-1);next_key=tl.gather(keys,nxt,0);next_max=tl.gather(largest,nxt,0)
  strict=(j<K-1)&(keys>next_key);gap=tl.where(strict,keys-next_key,1.)
  hinge=tl.where(strict,tl.maximum(A-(smallest-next_max)/gap,0.),0.)
  weighted=hinge/gap
  tl.store(Hinges+row*K+j,weighted,j<K)
  last_hinge=tl.sum(tl.where((j<K-1)&(next_key==threshold),weighted,0.),0)
  columns=tl.arange(0,B);best=tl.full((B,),float('-inf'),tl.float64)
  count=tl.zeros((B,),tl.int32);inverse=tl.zeros((B,),tl.float64);boundary_count=tl.zeros((B,),tl.int32)
  cutoff=tl.load(Cutoff+row)
  for start in range(0,V,B):
   c=start+columns;t=tl.load(T+row*V+c,c<V,other=float('inf')).to(tl.float64);s=tl.load(S+row*V+c,c<V,other=0.).to(tl.float64)
   eligible=(c<V)&(t<threshold);g=tl.where(eligible,threshold-t,1.)
   z=tl.where(eligible,A-(cutoff-s)/g,float('-inf'))
   new=z>best;equal=eligible&(z==best)
   inverse=tl.where(new,1./g,inverse+tl.where(equal,1./g,0.))
   count=tl.where(new,eligible.to(tl.int32),count+equal.to(tl.int32));best=tl.maximum(best,z)
   boundary_count+=((c<V)&(t==threshold)&(s==boundary_max)).to(tl.int32)
  maximum=tl.max(best,0);ties=tl.sum(tl.where(best==maximum,count,0),0)
  inverse_sum=tl.sum(tl.where(best==maximum,inverse,0.),0)
  member=tl.maximum(maximum,0.)
  tl.store(Outside+row,maximum);tl.store(OutsideCount+row,ties);tl.store(OutsideInverse+row,inverse_sum/tl.maximum(ties,1))
  tl.store(BoundaryCount+row,tl.sum(boundary_count,0));tl.store(BoundaryHinge+row,last_hinge);tl.store(Member+row,member)
  tl.store(Loss+row,(tl.sum(hinge*hinge,0)+member*member)/K)

 @triton.jit
 def _relative_dense_backward(T,S,Values,BoundaryMax,Cutoff,CutoffIndex,Member,Outside,
                              OutsideCount,OutsideInverse,BoundaryCount,BoundaryHinge,Cotangent,Gradient,
                              V:tl.constexpr,K:tl.constexpr,A:tl.constexpr,B:tl.constexpr):
  row=tl.program_id(0);columns=tl.arange(0,B)
  threshold=tl.load(Values+row*K+K-1).to(tl.float64);boundary_max=tl.load(BoundaryMax+row)
  cutoff=tl.load(Cutoff+row);cutoff_index=tl.load(CutoffIndex+row);maximum=tl.load(Outside+row)
  scale=tl.load(Cotangent+row).to(tl.float64)/K;member=(2.*tl.load(Member+row))*scale
  outside_grad=member/tl.maximum(tl.load(OutsideCount+row),1);cutoff_grad=member*tl.load(OutsideInverse+row)
  boundary_grad=((2.*tl.load(BoundaryHinge+row))*scale)/tl.maximum(tl.load(BoundaryCount+row),1)
  for start in range(0,V,B):
   c=start+columns;t=tl.load(T+row*V+c,c<V,other=float('inf')).to(tl.float64);s=tl.load(S+row*V+c,c<V,other=0.).to(tl.float64)
   eligible=(c<V)&(t<threshold);gap=tl.where(eligible,threshold-t,1.)
   z=A-(cutoff-s)/gap
   gradient=tl.where(eligible&(z==maximum),outside_grad/gap,0.)
   gradient+=tl.where((t==threshold)&(s==boundary_max),boundary_grad,0.)
   gradient-=tl.where(c==cutoff_index,cutoff_grad,0.)
   tl.store(Gradient+row*V+c,gradient,c<V)


def _cpu_rows(t,s,values,indices,alpha,cutoff,boundary_max):
 n,k=indices.shape;values=values.double();edited=s.double().gather(1,indices)
 hinges=torch.zeros((n,k),dtype=torch.float64);weighted=torch.zeros_like(hinges)
 for row in range(n):
  for j in range(k-1):
   gap=values[row,j]-values[row,j+1]
   if gap>0:
    upper=edited[row,values[row]==values[row,j]].min()
    lower=boundary_max[row] if values[row,j+1]==values[row,-1] else edited[row,values[row]==values[row,j+1]].max()
    hinges[row,j]=(alpha-(upper-lower)/gap).clamp_min(0.);weighted[row,j]=hinges[row,j]/gap
 threshold=values[:,-1:];lower=t<threshold;gap=torch.where(lower,threshold-t.double(),1.)
 z=alpha-(cutoff[:,None]-s.double())/gap;outside=torch.where(lower,z,-torch.inf).amax(1)
 winners=lower&(z==outside[:,None]);outside_count=winners.sum(1);outside_inverse=torch.where(winners,1./gap,0.).sum(1)/outside_count.clamp_min(1)
 boundary_count=((t==threshold)&(s==boundary_max[:,None])).sum(1)
 boundary_hinge=(weighted[:,:-1]*(values[:,1:]==threshold)).sum(1);member=outside.clamp_min(0.)
 return (hinges.square().sum(1)+member.square())/k,weighted,member,outside,outside_count,outside_inverse,boundary_count,boundary_hinge


class _RelativeRank(torch.autograd.Function):
 @staticmethod
 def forward(ctx,teacher,student,k,alpha,return_topk):
  t=teacher.detach().reshape(-1,teacher.shape[-1]).contiguous();s=student.reshape_as(t).contiguous();values,indices=t.topk(k,dim=-1,sorted=True);n,v=t.shape
  if s.is_cuda:
   scores=torch.empty((n,v),device=s.device,dtype=torch.float64);_boundary_scores[(n,triton.cdiv(v,1024))](t,s,values,scores,v,k,1024)
  else:scores=torch.where(t==values[:,-1:],s.double(),-torch.inf)
  boundary_values,boundary_indices=scores.topk(k,dim=-1,sorted=True);remaining=k-(values>values[:,-1:]).sum(1,keepdim=True)
  cutoff=boundary_values.gather(1,remaining-1).flatten().contiguous();cutoff_index=boundary_indices.gather(1,remaining-1).flatten().contiguous();boundary_max=boundary_values[:,0].contiguous()
  del scores,boundary_values,boundary_indices
  if s.is_cuda:
   loss=torch.empty(n,dtype=torch.float64,device=s.device);hinges=torch.empty((n,k),dtype=torch.float64,device=s.device)
   member,outside,outside_inverse,boundary_hinge=(torch.empty_like(loss) for _ in range(4));outside_count=torch.empty(n,dtype=torch.int32,device=s.device);boundary_count=torch.empty_like(outside_count)
   _relative_forward[(n,)](t,s,values,indices,boundary_max,cutoff,loss,hinges,member,outside,outside_count,outside_inverse,boundary_count,boundary_hinge,v,k,alpha,1024,triton.next_power_of_2(k),num_warps=4,enable_fp_fusion=False)
  else:loss,hinges,member,outside,outside_count,outside_inverse,boundary_count,boundary_hinge=_cpu_rows(t,s,values,indices,alpha,cutoff,boundary_max)
  ctx.save_for_backward(t,s,values,indices,boundary_max,cutoff,cutoff_index,hinges,member,outside,outside_count,outside_inverse,boundary_count,boundary_hinge);ctx.k,ctx.alpha,ctx.shape=k,alpha,student.shape
  if return_topk:
   ctx.mark_non_differentiable(values,indices);return loss.reshape(student.shape[:-1]),values,indices
  return loss.reshape(student.shape[:-1])
 @staticmethod
 @once_differentiable
 def backward(ctx,cotangent,*unused):
  t,s,values,indices,boundary_max,cutoff,cutoff_index,hinges,member,outside,outside_count,outside_inverse,boundary_count,boundary_hinge=ctx.saved_tensors;cotangent=cotangent.contiguous().flatten()
  if s.is_cuda:
   gradient=torch.empty_like(s)
   _relative_dense_backward[(len(s),)](t,s,values,boundary_max,cutoff,cutoff_index,member,outside,outside_count,outside_inverse,boundary_count,boundary_hinge,cotangent,gradient,s.shape[1],ctx.k,ctx.alpha,1024,num_warps=4,enable_fp_fusion=False)
   _selected_backward[(len(s),)](s,values,indices,hinges,cotangent,gradient,s.shape[1],ctx.k,triton.next_power_of_2(ctx.k),num_warps=4,enable_fp_fusion=False)
  else:
   scale=cotangent.double()/ctx.k;membership=2.*member*scale;threshold=values[:,-1:].double();lower=t<threshold;gap=torch.where(lower,threshold-t.double(),1.)
   z=ctx.alpha-(cutoff[:,None]-s.double())/gap
   gradient=torch.where(lower&(z==outside[:,None]),(membership/outside_count.clamp_min(1))[:,None]/gap,0.)
   gradient+=torch.where((t==threshold)&(s==boundary_max[:,None]),(2.*boundary_hinge*scale/boundary_count.clamp_min(1))[:,None],0.)
   gradient.scatter_add_(1,cutoff_index[:,None],-(membership*outside_inverse)[:,None])
   for row in range(len(s)):
    for j in range(ctx.k-1):
     if values[row,j]<=values[row,j+1]:continue
     high=indices[row,values[row]==values[row,j]];high=high[s[row,high]==s[row,high].min()];g=2.*hinges[row,j]*scale[row];gradient[row,high]-=g/len(high)
     if values[row,j+1]>values[row,-1]:
      low=indices[row,values[row]==values[row,j+1]];low=low[s[row,low]==s[row,low].max()];gradient[row,low]+=g/len(low)
   gradient=gradient.to(s.dtype)
  return None,gradient.reshape(ctx.shape),None,None,None


def tiled_relative_topk_margin_loss(teacher,student,*,k=100,margin_fraction=.5,return_teacher_topk=False):
 """First derivatives; no gap floor, approximation, or teacher gradients."""
 if teacher.shape!=student.shape or teacher.ndim<2 or not student.numel():raise ValueError('matching nonempty token/vocabulary shapes required')
 if type(k) is not int or not 1<=k<=student.shape[-1] or not 0<margin_fraction<=1:raise ValueError('valid ranking size and positive margin fraction required')
 if teacher.device!=student.device or not teacher.is_floating_point() or not student.is_floating_point():raise ValueError('floating teacher/student on same device required')
 if student.is_cuda:
  if triton is None:raise RuntimeError('Triton required for CUDA ranking')
  if student.dtype!=torch.float32 or teacher.dtype not in (torch.float32,torch.float64):raise ValueError('CUDA requires FP32 student and FP32/FP64 teacher')
 out=_RelativeRank.apply(teacher,student,k,float(margin_fraction),return_teacher_topk)
 return (out[0],(out[1],out[2])) if return_teacher_topk else out
