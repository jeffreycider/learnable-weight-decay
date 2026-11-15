import itertools
import torch
from rewriter.relative_ranking_loss import relative_tie_aware_topk_margin_loss as relative
from rewriter.ranking_loss import tie_aware_topk_margin_loss as absolute
from rewriter.rank_preservation import topk_preservation_flags

def test_microgap_flip_is_not_made_cheap_by_logit_scale():
    teacher=torch.tensor([[3.,2.,1.,0.]],dtype=torch.float64)
    student=torch.tensor([[3.,1.,2.,0.]],dtype=torch.float64)
    base=relative(teacher,student,k=3)
    for scale in (1.,1e-4,1e-8):
        torch.testing.assert_close(relative(teacher*scale,student*scale,k=3),base)
        torch.testing.assert_close(absolute(teacher*scale,student*scale,k=3),absolute(teacher,student,k=3)*scale**2)
    assert base.item()>0

def test_zero_loss_never_hides_a_strict_rank_change_small_exhaustive():
    teachers=[(3.,3.,2.,1.),(3.,2.,2.,1.),(2.,2.,2.,1.),(3.,2.,1.,0.)]
    students=list(itertools.product((0.,1.,2.,3.),repeat=4))
    for k in range(1,5):
        for row in teachers:
            t=torch.tensor([row]*len(students),dtype=torch.float64);s=torch.tensor(students,dtype=torch.float64)
            loss=relative(t,s,k=k);changed=topk_preservation_flags(t,s,k=k)['strict_topk_changed']
            assert not ((loss==0)&changed).any()

def test_exact_ties_and_full_classification_no_invalid_gradients():
    t=torch.ones(2,5,dtype=torch.float64);s=torch.arange(10,dtype=torch.float64).reshape(2,5).requires_grad_()
    loss=relative(t,s,k=5);assert torch.equal(loss,torch.zeros_like(loss))
    loss.sum().backward();assert torch.equal(s.grad,torch.zeros_like(s))
    t=torch.tensor([[4.,3.,3.,0.]],dtype=torch.float64)
    for values in ([4.,-2.,3.,0.],[4.,3.,-2.,0.]):
        assert relative(t,torch.tensor([values],dtype=torch.float64),k=2).item()==0

def test_gradcheck_teacher_detached_and_scale_covariant_cotangent():
    t=torch.tensor([[4.,3.,2.,0.]],dtype=torch.float64,requires_grad=True)
    s=torch.tensor([[3.1,3.2,1.,2.]],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda x:relative(t,x,k=3),s)
    assert torch.autograd.gradcheck(lambda x:relative(t,x,k=4),s)
    grad=torch.autograd.grad(relative(t,s,k=3).sum(),s)[0]
    scale=1e-7;small=(s.detach()*scale).requires_grad_()
    small_grad=torch.autograd.grad(relative(t*scale,small,k=3).sum(),small)[0]
    torch.testing.assert_close(small_grad*scale,grad);assert t.grad is None

def test_membership_uses_worst_relative_outsider():
    # Teacher boundary1: outsider1 gap0.001 has relative violation1.5;
    # outsider2 gap1 has0.6, despite its much larger absolute violation.
    t=torch.tensor([[1.,.999,0.]],dtype=torch.float64)
    s=torch.tensor([[1.,1.001,1.1]],dtype=torch.float64,requires_grad=True)
    loss=relative(t,s,k=1);torch.testing.assert_close(loss,torch.tensor([2.25],dtype=torch.float64))
    loss.backward();assert s.grad[0,1]>0 and s.grad[0,2]==0
