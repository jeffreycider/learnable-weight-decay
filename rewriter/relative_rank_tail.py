"""Default-off dense reference: worst relative ranking violation per microbatch.

No production dispatch calls this module. The inputs are existing model logits;
this reduction needs no additional model forward or backward. Unlike a mean
across relations/tokens, a single violation is not divided by their counts.
"""
import torch


def maximum_relative_rank_loss(teacher, student, *, k=100, margin_fraction=.5):
    """Return one maximum squared fractional hinge per context, in FP64.

    Each positive teacher class gap g requires edited gap >= alpha*g. Ordering
    uses the smallest score of the higher class and largest score of the lower
    class. Membership uses the R-th best member of the whole boundary tie class
    against every strictly lower outsider. Original exact ties are interchangeable.

    All terms are nonnegative, so this value is zero iff every such margin is
    satisfied. With alpha>0, zero implies no strict top-k order/membership change.
    The converse for unchanged ranks is not claimed: a positive but too-small
    edited gap still incurs a hinge. No epsilon turns near ties into exact ties.
    """
    if teacher.shape != student.shape or teacher.ndim < 2 or not 1 <= k <= teacher.shape[-1]:
        raise ValueError('matching context/vocabulary tensors and valid k required')
    if not 0 < margin_fraction <= 1:
        raise ValueError('margin fraction must lie in (0,1]')
    target=teacher.detach().double(); s=student.double()
    values,indices=target.topk(k,dim=-1,sorted=True);selected=s.gather(-1,indices)
    threshold=values[...,-1:];boundary_class=target==threshold
    boundary_scores=torch.where(boundary_class,s,-torch.inf)
    remaining=k-(values>threshold).sum(-1,keepdim=True)
    cutoff=boundary_scores.topk(k,dim=-1,sorted=True).values.gather(-1,remaining-1).squeeze(-1)
    same_class=values.unsqueeze(-1)==values.unsqueeze(-2)
    class_min=torch.where(same_class,selected.unsqueeze(-2),torch.inf).amin(-1)
    class_max=torch.where(same_class,selected.unsqueeze(-2),-torch.inf).amax(-1)
    class_max=torch.where(values==threshold,boundary_scores.amax(-1,keepdim=True),class_max)
    if k > 1:
        gaps=values[...,:-1]-values[...,1:];strict=gaps>0
        hinge=(margin_fraction-(class_min[...,:-1]-class_max[...,1:])/torch.where(strict,gaps,1.)).relu()
        ordering=torch.where(strict,hinge.square(),0.).amax(-1)
    else:
        ordering=s.sum(-1)*0.
    lower=target<threshold;gaps=threshold-target
    hinge=margin_fraction-(cutoff.unsqueeze(-1)-s)/torch.where(lower,gaps,1.)
    membership=torch.where(lower,hinge,-torch.inf).amax(-1).relu().square()
    return torch.maximum(ordering,membership)


def reduce_micro_maximum(per_context_loss, *, micro_tokens=16*1024, normalization='mean_micro'):
    """Reduce complete fixed-size microbatches, returning scalar and maxima.

    mean_micro = sum(maxima)/number_of_micros; sum_micro = sum(maxima).
    Both are zero iff all supplied margins hold. Their scale depends on the
    declared partition; they are NOT equivalent to the whole-panel maximum.
    Uneven final chunks are rejected rather than silently changing their weight.
    A production implementation can backprop each micromaximum immediately with
    its fixed 1/M (or 1) weight, retaining one model backward per microbatch.
    """
    if type(micro_tokens) is not int or micro_tokens < 1 or not per_context_loss.numel() or per_context_loss.numel()%micro_tokens:
        raise ValueError('complete nonempty fixed-size microbatches required')
    if normalization not in ('mean_micro','sum_micro'):
        raise ValueError('explicit mean_micro or sum_micro normalization required')
    maxima=per_context_loss.reshape(-1,micro_tokens).amax(-1)
    return (maxima.mean() if normalization=='mean_micro' else maxima.sum()),maxima
