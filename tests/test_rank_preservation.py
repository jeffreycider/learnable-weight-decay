import itertools

import torch

from rewriter.rank_preservation import topk_preservation_flags
from rewriter.ranking_loss import tie_aware_topk_margin_loss,topk_margin_loss


def oracle(teacher,student,k):
    chosen=sorted(range(len(student)),key=lambda i:student[i],reverse=True)[:k]
    threshold=sorted(teacher,reverse=True)[k-1]
    if any(teacher[i]>threshold and i not in chosen for i in range(len(teacher))):return True
    if any(teacher[i]<threshold for i in chosen):return True
    # Preserve every strict relation relevant to selected ranks; original
    # boundary ties that were not selected can be exchanged freely.
    for i in chosen:
        for j in chosen:
            if teacher[i]>teacher[j] and student[i]<=student[j]:return True
        for j in range(len(student)):
            if teacher[j]<threshold and student[i]<=student[j]:return True
    return False


def test_all_small_orderings_match_direct_pairwise_oracle():
    teachers=[(3.,3.,2.,1.),(3.,2.,2.,1.),(2.,2.,2.,1.),(3.,2.,1.,0.)]
    students=list(itertools.product((0.,1.,2.,3.),repeat=4))
    for k in (1,2,3,4):
        for teacher in teachers:
            t=torch.tensor([teacher]*len(students));s=torch.tensor(students)
            actual=topk_preservation_flags(t,s,k=k)['strict_topk_changed'].tolist()
            # torch.topk tie order can differ from Python; failure status is
            # invariant because newly tied distinct ranks are always rejected.
            expected=[oracle(teacher,student,k) for student in students]
            assert actual==expected


def test_original_ties_can_reorder_but_new_ties_are_not_preservation():
    t=torch.tensor([[4.,3.,3.,0.]])
    s=torch.tensor([[4.,2.,3.5,0.]])
    assert not topk_preservation_flags(t,s,k=2)['strict_topk_changed'].item()
    # Collapse distinct original ranks into the same student score.
    assert topk_preservation_flags(t,torch.tensor([[3.,3.,2.,0.]]),k=2)['strict_topk_changed'].item()


def test_supplied_topk_is_equivalent_and_nonfinite_rows_fail():
    t=torch.tensor([[4.,3.,2.,1.]])
    s=torch.tensor([[4.,3.1,2.,1.]])
    a=topk_preservation_flags(t,s,k=3)
    b=topk_preservation_flags(t,s,k=3,teacher_topk=t.topk(3),student_topk=s.topk(3))
    assert all(torch.equal(a[key],b[key]) for key in a)
    s[0,3]=torch.nan
    assert topk_preservation_flags(t,s,k=3)['strict_topk_changed'].item()


def test_loss_preserves_tie_classes_not_arbitrary_adjacent_representatives():
    t=torch.tensor([[3.,3.,2.,0.]],dtype=torch.float64)
    # Ignoring only zero-gap adjacent pairs would miss token0 falling below2.
    bad=torch.tensor([[1.,4.,3.,0.]],dtype=torch.float64,requires_grad=True)
    loss=tie_aware_topk_margin_loss(t,bad,k=3)
    assert loss.item()>0
    loss.sum().backward();assert bad.grad.isfinite().all()
    good=torch.tensor([[4.,3.5,2.,0.]],dtype=torch.float64)
    assert tie_aware_topk_margin_loss(t,good,k=3).item()==0


def test_boundary_tie_members_can_exchange_and_all_tied_teacher_has_zero_rank_loss():
    t=torch.tensor([[4.,3.,3.,0.]],dtype=torch.float64)
    # Whichever boundary token teacher.topk chooses, the other may replace it.
    for s in ([4.,-2.,3.,0.],[4.,3.,-2.,0.]):
        assert tie_aware_topk_margin_loss(t,torch.tensor([s],dtype=torch.float64),k=2).item()==0
    t=torch.ones(2,5,dtype=torch.float64)
    s=torch.arange(10,dtype=torch.float64).reshape(2,5).requires_grad_()
    loss=tie_aware_topk_margin_loss(t,s,k=3)
    assert torch.equal(loss,torch.zeros_like(loss))
    loss.sum().backward();assert torch.equal(s.grad,torch.zeros_like(s))


def test_distinct_teacher_ranks_match_old_loss_and_full_classification_gradcheck():
    t=torch.tensor([[4.,3.,2.,0.]],dtype=torch.float64)
    s=torch.tensor([[3.1,3.2,1.,2.]],dtype=torch.float64,requires_grad=True)
    assert torch.equal(tie_aware_topk_margin_loss(t,s,k=3),topk_margin_loss(t,s,k=3))
    assert torch.autograd.gradcheck(lambda x:tie_aware_topk_margin_loss(t,x,k=4),s)
