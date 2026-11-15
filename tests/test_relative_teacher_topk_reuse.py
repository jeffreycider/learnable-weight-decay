"""Exact old/new reference comparisons; no GPT evaluation."""
from pathlib import Path
import pytest
import torch
import rewriter.tokenwise_reference as current
import rewriter.relative_ranking_loss as relative

OLD=Path('runs/teacher-relative-topk-cpu-20261112/sources/rewriter')

def old_functions():
    rank_ns=dict(relative.__dict__);exec((OLD/'relative_ranking_loss.py').read_text(),rank_ns)
    panel_ns=dict(current.__dict__)
    source=(OLD/'tokenwise_reference.py').read_text().replace(
        'from .relative_ranking_loss import relative_tie_aware_topk_margin_loss',
        'relative_tie_aware_topk_margin_loss=OLD_RANK')
    panel_ns['OLD_RANK']=rank_ns['relative_tie_aware_topk_margin_loss'];exec(source,panel_ns)
    return rank_ns['relative_tie_aware_topk_margin_loss'],panel_ns['logit_terms_and_cotangent']


@pytest.mark.parametrize('k',[1,4,9])
def test_optional_tuple_preserves_scalar_grad_and_exact_boundary(k):
    torch.set_num_threads(1);old,_=old_functions()
    teacher=torch.tensor([[5.,5.,4.,3.,3.,3.,2.,1.,0.],[9.,8.,7.,6.,5.,4.,3.,2.,1.]],dtype=torch.float64)
    torch.manual_seed(18);student=torch.randn_like(teacher,requires_grad=True)
    expected=old(teacher,student,k=k)
    actual,topk=relative.relative_tie_aware_topk_margin_loss(teacher,student,k=k,return_teacher_topk=True)
    default=relative.relative_tie_aware_topk_margin_loss(teacher,student,k=k)
    assert torch.equal(expected,actual) and torch.equal(actual,default)
    assert all(torch.equal(x,y) for x,y in zip(topk,teacher.topk(k,dim=-1,sorted=True)))
    assert all(not x.requires_grad for x in topk)
    assert torch.equal(torch.autograd.grad(expected.sum(),student)[0],torch.autograd.grad(actual.sum(),student)[0])


@pytest.mark.parametrize('k',[1,4,9])
@pytest.mark.parametrize('backward,balance,weight',[(True,False,100.),(False,False,100.),(True,True,100.),(False,True,0.),(True,False,0.),(False,False,0.)])
def test_panel_bitexact_multichunk_noncontiguous_ties_disabled_and_balance(k,backward,balance,weight):
    torch.set_num_threads(1);_,old=old_functions();torch.manual_seed(22)
    teacher=torch.randn(2,9,5,dtype=torch.float64).transpose(1,2)
    teacher[...,0]=2.;teacher[...,1]=2.;teacher[...,2:6]=1.
    teacher=teacher.log_softmax(-1).transpose(0,1)
    student=(teacher.float()+torch.randn_like(teacher.float())*.1).transpose(0,1).contiguous().transpose(0,1)
    assert not teacher.is_contiguous() and not student.is_contiguous()
    config=current.TokenwiseConfig(mean_tolerance=1e-10,tail_tolerance=1e-10,scale=1e-8,rho=.01,ranking_k=k,ranking_tie_policy='exact_teacher_ties',ranking_loss_mode='relative_gap',ranking_weight=weight)
    kwargs=dict(config=config,multiplier=.1,panel_tokens=20,chunk_tokens=3,backward=backward,balance=balance)
    expected=old(student,teacher,**kwargs);actual=current.logit_terms_and_cotangent(student,teacher,**kwargs)
    for a,b in zip(expected,actual):
        if a is None:assert b is None
        else:assert torch.equal(a,b)


@pytest.mark.parametrize('ranking_weight,balance',[(100.,False),(0.,False),(0.,True)])
def test_exactly_one_teacher_topk_per_chunk(monkeypatch,ranking_weight,balance):
    torch.set_num_threads(1);teacher=torch.randn(7,9,dtype=torch.float64).log_softmax(-1)
    student=teacher.float()+.1*torch.randn(7,9);calls=[];original=torch.Tensor.topk
    def counted(tensor,*args,**kwargs):
        if tensor.untyped_storage().data_ptr()==teacher.untyped_storage().data_ptr():calls.append(tuple(tensor.shape))
        return original(tensor,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'topk',counted)
    config=current.TokenwiseConfig(mean_tolerance=1e-10,tail_tolerance=1e-10,scale=1e-8,rho=.01,ranking_k=4,ranking_tie_policy='exact_teacher_ties',ranking_loss_mode='relative_gap',ranking_weight=ranking_weight)
    current.logit_terms_and_cotangent(student,teacher,config=config,multiplier=.1,panel_tokens=7,chunk_tokens=3,backward=True,balance=balance)
    assert calls==[(3,9),(3,9),(1,9)]
