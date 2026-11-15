import dataclasses
import pytest
import torch
from rewriter.tokenwise_reference import TokenwiseConfig,TokenwiseDual,logit_terms_and_cotangent
from rewriter.relative_ranking_loss import relative_tie_aware_topk_margin_loss


def config(**kwargs):
    return TokenwiseConfig(1e-10,1e-10,1e-8,.01,ranking_weight=.1,ranking_k=3,ranking_tie_policy='exact_teacher_ties',**kwargs)


def test_relative_configuration_is_explicit_and_old_dual_canonicalizes_absolute():
    old=TokenwiseDual([1],config=config(),dual_lr=.01);state=old.state_dict();state['config'].pop('ranking_loss_mode');old.load_state_dict(state)
    new=TokenwiseDual([1],config=config(ranking_loss_mode='relative_gap'),dual_lr=.01)
    with pytest.raises(ValueError,match='configuration'):new.load_state_dict(state)
    for backend in ('tiled_fp64_v1','tiled_tie_fp64_v1'):
        with pytest.raises(ValueError,match='explicit backend'):config(ranking_loss_mode='relative_gap',loss_backend=backend)
    with pytest.raises(ValueError,match='requires relative_gap'):config(loss_backend='tiled_relative_fp64_v1')


def test_relative_reference_matches_direct_loss_gradient_without_extra_student_backward():
    teacher=torch.tensor([[2.,1.,.9,-1.],[1.,.4,.3,-2.]],dtype=torch.float64).log_softmax(-1)
    student=torch.tensor([[1.,1.5,.8,-1.],[.6,.7,.3,-2.]],dtype=torch.float32)
    c=config(ranking_loss_mode='relative_gap');gradient,parts,kl=logit_terms_and_cotangent(student,teacher,config=c,multiplier=.02,panel_tokens=2,chunk_tokens=1)
    s=student.clone().requires_grad_(True);div=(teacher.exp()*(teacher-s.double().log_softmax(-1))).sum(-1)
    rank=relative_tie_aware_topk_margin_loss(teacher,s.double(),k=3,margin_fraction=.5)
    loss=(.02*(div-c.mean_tolerance)/c.scale+.5*c.rho*((div-c.tail_tolerance)/c.scale).relu().square()+c.ranking_weight*rank).mean()
    expected=torch.autograd.grad(loss,s)[0]
    torch.testing.assert_close(gradient,expected,rtol=2e-6,atol=1e-3)
    torch.testing.assert_close(parts[2],c.ranking_weight*rank.sum(),rtol=1e-12,atol=1e-12)
