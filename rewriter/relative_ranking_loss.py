"""Experimental relative teacher-gap constraints; not the production loss default.

Absolute squared-logit hinges make a rank reversal across a tiny original gap
cheap. Here each strict relation pays for the fractional margin violation. No
numerical floor reclassifies a nonzero teacher gap as a tie. Exact ties retain
the existing interchangeable-class semantics. This is a dense reference for a
separate objective experiment, not a qualified fast backend.
"""
import torch


def relative_tie_aware_topk_margin_loss(teacher,student,*,k=100,margin_fraction=.5,return_teacher_topk=False):
    """Per-context mean squared fractional margin violations.

    For an original positive gap g and edited gap d, the hinge is
    max(0, margin_fraction - d/g)^2. Ordering uses adjacent exact teacher
    classes. Membership uses the worst fractional violation among *all* lower
    classes, which need not be the outsider with the largest absolute hinge.
    Teacher tensors are detached; division uses FP64 for small strict gaps.
    Optional teacher top-k output reuses the exact sorted tuple for diagnostics.
    Gradients can be large for tiny gaps; training must explicitly qualify its
    optimizer/scaling and cannot silently reuse absolute-loss coefficients.
    """
    if teacher.shape!=student.shape or teacher.ndim<2 or not 1<=k<=teacher.shape[-1]:
        raise ValueError('matching context/vocabulary tensors and 1 <= k <= vocabulary required')
    if not 0<margin_fraction<=1:raise ValueError('margin fraction must lie in (0,1]')
    target=teacher.detach().double();s=student.double()
    values,indices=target.topk(k,dim=-1,sorted=True);selected=s.gather(-1,indices)
    threshold=values[...,-1:];boundary_class=target==threshold
    boundary_scores=torch.where(boundary_class,s,-torch.inf)
    remaining=k-(values>threshold).sum(-1,keepdim=True)
    cutoff=boundary_scores.topk(k,dim=-1,sorted=True).values.gather(-1,remaining-1).squeeze(-1)
    same_class=values.unsqueeze(-1)==values.unsqueeze(-2)
    class_min=torch.where(same_class,selected.unsqueeze(-2),torch.inf).amin(-1)
    class_max=torch.where(same_class,selected.unsqueeze(-2),-torch.inf).amax(-1)
    class_max=torch.where(values==threshold,boundary_scores.amax(-1,keepdim=True),class_max)
    gaps=values[...,:-1]-values[...,1:];strict=gaps>0
    # Replacing only exact zero denominators by one is for finite inactive
    # arithmetic, never an epsilon tolerance or an approximate tie decision.
    denominator=torch.where(strict,gaps,1.)
    internal_hinge=(margin_fraction-(class_min[...,:-1]-class_max[...,1:])/denominator).relu()
    internal=torch.where(strict,internal_hinge.square(),0.).sum(-1)
    lower=target<threshold;outside_gaps=threshold-target
    relative=margin_fraction-(cutoff.unsqueeze(-1)-s)/torch.where(lower,outside_gaps,1.)
    membership=torch.where(lower,relative,-torch.inf).amax(-1).relu().square()
    loss=(internal+membership)/k
    return (loss,(values,indices)) if return_teacher_topk else loss
