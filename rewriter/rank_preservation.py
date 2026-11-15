"""Top-k preservation with original exact ties treated as one rank.

The student's selected top-k must contain every teacher token strictly above
the teacher boundary, contain no token below it, and respect every strict order
among its selected tokens. New student ties between distinct teacher ranks are
failures. At a tied teacher boundary, any members of that tied class may fill
the remaining slots. No numerical tolerance is silently added to equality.
"""
import torch


@torch.no_grad()
def topk_preservation_flags(teacher, student, *, k=100, teacher_topk=None, student_topk=None):
    """Return per-token booleans; no extra model execution or autograd graph.

    Optional topk tuples are (sorted values, indices) from these exact tensors.
    The dense outsider reduction is a correctness reference; fused kernels can
    produce the same maximum while traversing logits for the training loss.
    """
    if teacher.shape!=student.shape or teacher.ndim<2 or not 1<=k<=teacher.shape[-1]:
        raise ValueError('matching token/vocabulary tensors and 1 <= k <= vocabulary required')
    tv,ti=teacher.topk(k,dim=-1,sorted=True) if teacher_topk is None else teacher_topk
    sv,si=student.topk(k,dim=-1,sorted=True) if student_topk is None else student_topk
    if any(x.shape!=teacher.shape[:-1]+(k,) for x in (tv,ti,sv,si)):
        raise ValueError('top-k arrays have wrong shape')
    boundary=tv[...,-1:]
    teacher_at_student=teacher.gather(-1,si)
    sorted_student=si.sort(-1).values.contiguous()
    positions=torch.searchsorted(sorted_student,ti.contiguous()).clamp_max(k-1)
    included=sorted_student.gather(-1,positions)==ti
    missing_strict=((tv>boundary)&~included).any(-1)
    selected_below=(teacher_at_student<boundary).any(-1)
    lower_max=torch.where(teacher<boundary,student,-torch.inf).amax(-1)
    # This also rejects a new tie across the top-k boundary, even if this
    # particular topk implementation happened to choose the correct member.
    boundary_crossing=lower_max>=sv[...,-1]
    reversed_order=(teacher_at_student[...,:-1]<teacher_at_student[...,1:]).any(-1)
    new_tie=((teacher_at_student[...,:-1]>teacher_at_student[...,1:])&
             (sv[...,:-1]==sv[...,1:])).any(-1)
    invalid=~(teacher.isfinite().all(-1)&student.isfinite().all(-1))
    order=reversed_order|new_tie|invalid
    membership=missing_strict|selected_below|boundary_crossing|invalid
    teacher_argmax=teacher.argmax(-1);student_argmax=student.argmax(-1)
    return dict(literal_argmax_changed=teacher_argmax!=student_argmax,
        literal_order_changed=(ti!=si).any(-1),
        literal_membership_changed=(ti.sort(-1).values!=sorted_student).any(-1),
        tie_aware_argmax_changed=(teacher.gather(-1,student_argmax.unsqueeze(-1)).squeeze(-1)<tv[...,0])|invalid,
        strict_order_changed=order,strict_membership_changed=membership,
        strict_topk_changed=order|membership)
