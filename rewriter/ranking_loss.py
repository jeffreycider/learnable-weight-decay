"""Dense reference for the top-k margin term in tokenwise-v1 training.

Teacher logits are immutable targets. Adjacent inequalities preserve ordering
within the teacher top-k; one maximum checks membership against every outsider.
Positive teacher gaps and zero loss imply unchanged top-k order/membership.
Exact teacher ties do not define a unique order. This dense reference is for
correctness; a tiled implementation should avoid the vocabulary-sized temporary.
"""
import torch


def topk_margin_loss(teacher, student, *, k=10, margin_fraction=0.5):
    """Return one squared-hinge penalty per token, in squared-logit units.

Require edited margins to retain at least ``margin_fraction`` of the original
margin. Unlike a zero-margin hinge, this pushes back before the order flips.
The membership term uses the worst violation, not a sum over the vocabulary.
"""
    if teacher.shape != student.shape or teacher.ndim < 2:
        raise ValueError('matching token/vocabulary shapes required')
    if not 1 <= k < teacher.shape[-1]:
        raise ValueError('k must be positive and smaller than the vocabulary')
    if not 0 < margin_fraction <= 1:
        raise ValueError('margin_fraction must lie in (0, 1]')
    target = teacher.detach()
    values, indices = target.topk(k, dim=-1, sorted=True)
    edited = student.gather(-1, indices)
    original_gaps = values[..., :-1] - values[..., 1:]
    edited_gaps = edited[..., :-1] - edited[..., 1:]
    internal = (margin_fraction * original_gaps - edited_gaps).relu().square().sum(-1)

    # For every outsider j: s_k - s_j >= fraction * (t_k - t_j).
    adjusted = student - margin_fraction * target
    outside = adjusted.scatter(-1, indices, -torch.inf).amax(-1)
    boundary = edited[..., -1] - margin_fraction * values[..., -1]
    membership = (outside - boundary).relu().square()
    return (internal + membership) / k


def tie_aware_topk_margin_loss(teacher, student, *, k=100, margin_fraction=0.5):
    """Strict teacher-rank margins, with exact teacher ties interchangeable.

    Each adjacent teacher score class uses its minimum student score against
    the next class's maximum. At the teacher boundary, choose the best remaining
    slots from the *whole* tied class, including tokens outside teacher.topk.
    A student token from a lower teacher class must remain below that cutoff.
    This reference uses O(k^2) group masks; production should use segmented scans.
    The old topk_margin_loss arithmetic is intentionally unchanged.
    """
    if teacher.shape!=student.shape or teacher.ndim<2 or not 1<=k<=teacher.shape[-1]:
        raise ValueError('matching tensors and 1 <= k <= vocabulary required')
    if not 0<margin_fraction<=1:raise ValueError('margin fraction must lie in (0,1]')
    target=teacher.detach();values,indices=target.topk(k,dim=-1,sorted=True)
    selected=student.gather(-1,indices);threshold=values[...,-1:]
    boundary_class=target==threshold
    boundary_scores=torch.where(boundary_class,student,-torch.inf)
    boundary_max=boundary_scores.amax(-1,keepdim=True)
    remaining=k-(values>threshold).sum(-1,keepdim=True)
    cutoff=boundary_scores.topk(k,dim=-1,sorted=True).values.gather(-1,remaining-1).squeeze(-1)
    same_class=values.unsqueeze(-1)==values.unsqueeze(-2)
    class_min=torch.where(same_class,selected.unsqueeze(-2),torch.inf).amin(-1)
    class_max=torch.where(same_class,selected.unsqueeze(-2),-torch.inf).amax(-1)
    class_max=torch.where(values==threshold,boundary_max,class_max)
    gaps=values[...,:-1]-values[...,1:]
    constraints=(margin_fraction*gaps-(class_min[...,:-1]-class_max[...,1:])).relu().square()
    internal=torch.where(gaps>0,constraints,0.).sum(-1)
    adjusted=student-margin_fraction*target
    outside=torch.where(target<threshold,adjusted,-torch.inf).amax(-1)
    membership=(outside-(cutoff-margin_fraction*threshold.squeeze(-1))).relu().square()
    return (internal+membership)/k
