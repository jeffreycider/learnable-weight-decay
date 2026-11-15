import itertools
import pytest
import torch
from rewriter.relative_rank_tail import maximum_relative_rank_loss,reduce_micro_maximum
from rewriter.rank_preservation import topk_preservation_flags
from rewriter.relative_ranking_loss import relative_tie_aware_topk_margin_loss


def test_zero_loss_implies_no_strict_changes_with_exact_teacher_ties():
    for teacher in ([3.,2.,1.,0.],[3.,3.,2.,0.],[3.,2.,2.,0.],[2.,2.,2.,2.]):
        t=torch.tensor([teacher],dtype=torch.double)
        for values in itertools.permutations([0.,1.,2.,3.]):
            s=torch.tensor([values],dtype=torch.double)
            for k in (1,2,3,4):
                loss=maximum_relative_rank_loss(t,s,k=k)
                all_margins=relative_tie_aware_topk_margin_loss(t,s,k=k)
                assert (loss.item()==0)==(all_margins.item()==0)
                if loss.item()==0:
                    assert not topk_preservation_flags(t,s,k=k)['strict_topk_changed'].any()
    t=torch.tensor([[3.,2.,2.,0.]],dtype=torch.double)
    s=torch.tensor([[3.,1.,2.,0.]],dtype=torch.double)
    assert maximum_relative_rank_loss(t,s,k=2).item()==0  # boundary tie exchange


def test_one_reversed_pair_is_not_diluted_by_relations_or_tokens():
    t=torch.tensor([[3.,2.,1.,0.]],dtype=torch.double).repeat(1024,1)
    s=t.clone();s[513,1]=.9;s.requires_grad_()
    per=maximum_relative_rank_loss(t,s,k=3)
    loss,maxima=reduce_micro_maximum(per,micro_tokens=1024)
    assert loss.item()==pytest.approx(.36)
    assert (per>0).sum()==1 and maxima.shape==(1,)
    loss.backward();assert torch.isfinite(s.grad).all() and (s.grad.abs().sum(-1)>0).sum()==1


def test_membership_checks_worst_fractional_outsider_and_full_vocabulary():
    t=torch.tensor([[4.,3.,2.,0.]],dtype=torch.double)
    s=torch.tensor([[4.,3.,2.8,2.5]],dtype=torch.double)
    # boundary3 versus outsider2 has hinge .3; outsider0 has hinge1/3.
    assert maximum_relative_rank_loss(t,s,k=2).item()==pytest.approx((.5-.5/3)**2)
    assert maximum_relative_rank_loss(t,t,k=4).item()==0
    t=torch.tensor([[3.,3.,2.]],dtype=torch.double);s=torch.tensor([[1.,4.,3.]],dtype=torch.double)
    assert maximum_relative_rank_loss(t,s,k=3).item()>0


def test_gradcheck_teacher_stop_gradient_and_scale_invariance():
    t=torch.tensor([[3.,2.,1.,0.]],dtype=torch.double,requires_grad=True)
    s=torch.tensor([[3.1,.85,1.1,-.1]],dtype=torch.double,requires_grad=True)
    assert torch.autograd.gradcheck(lambda x:maximum_relative_rank_loss(t,x,k=3),(s,))
    a=maximum_relative_rank_loss(t,s,k=3);b=maximum_relative_rank_loss(t*1e-8,s*1e-8,k=3)
    torch.testing.assert_close(a,b)
    gs,gt=torch.autograd.grad(a.sum(),(s,t),allow_unused=True);assert gt is None and torch.isfinite(gs).all()


def test_micro_partition_and_normalization_are_explicit():
    values=torch.tensor([1.,0.,4.,0.],dtype=torch.double,requires_grad=True)
    mean,_=reduce_micro_maximum(values,micro_tokens=2,normalization='mean_micro')
    total,_=reduce_micro_maximum(values,micro_tokens=2,normalization='sum_micro')
    whole,_=reduce_micro_maximum(values,micro_tokens=4)
    assert (mean.item(),total.item(),whole.item())==(2.5,5.,4.)
    assert torch.autograd.gradcheck(lambda x:reduce_micro_maximum(x,micro_tokens=2)[0],(values,))
    assert reduce_micro_maximum(torch.zeros(4),micro_tokens=2)[0]==0
    with pytest.raises(ValueError):reduce_micro_maximum(torch.ones(3),micro_tokens=2)
    with pytest.raises(ValueError):reduce_micro_maximum(values,normalization='implicit')
