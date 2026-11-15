from types import SimpleNamespace
from dataclasses import asdict
import pytest
import torch
from rewriter.tokenwise_reference import TokenwiseConfig
from rewriter.tokenwise_tiled_relative import logit_terms_and_cotangent
from rewriter.relative_ranking_loss import relative_tie_aware_topk_margin_loss
from rewriter.token_preservation import token_preservation_terms
from rewriter.rank_preservation import topk_preservation_flags


def config():
    values=asdict(TokenwiseConfig(1e-10,1e-10,1e-8,.01,ranking_weight=3e-9,ranking_k=100,ranking_tie_policy='exact_teacher_ties'))
    values.update(ranking_loss_mode='relative_gap',loss_backend='tiled_relative_fp64_v1')
    return SimpleNamespace(**values)


def reference(student,teacher,cfg,multiplier,panel):
    s=student.detach().clone().requires_grad_();lp=s.double().log_softmax(-1)
    kl=(teacher.exp()*(teacher-lp)).sum(-1)
    rank=relative_tie_aware_topk_margin_loss(teacher,s.double(),k=min(cfg.ranking_k,s.shape[-1]),margin_fraction=cfg.margin_fraction)
    terms=token_preservation_terms(kl,multiplier=multiplier,mean_tolerance=cfg.mean_tolerance,tail_tolerance=cfg.tail_tolerance,scale=cfg.scale,rho=cfg.rho,ranking_loss=rank,ranking_weight=cfg.ranking_weight)
    tail=torch.autograd.grad(terms['tail_quadratic'].sum()/panel,s,retain_graph=True)[0]
    rank_grad=torch.autograd.grad(rank.sum()/panel,s,retain_graph=True)[0]
    grad=torch.autograd.grad(terms['total'].sum()/panel,s)[0]
    return grad,terms,kl,tail,rank_grad


@pytest.mark.parametrize('vocabulary',[10,137])
@pytest.mark.parametrize('multiplier',[0.,.03])
def test_combined_cotangent_components_and_balance_match_dense(vocabulary,multiplier):
    torch.manual_seed(45);t=torch.randn(7,vocabulary).double().log_softmax(-1);student=(t+torch.randn_like(t)*2e-5).float();cfg=config();panel=19
    g,sums,kl=logit_terms_and_cotangent(student,t,config=cfg,multiplier=multiplier,panel_tokens=panel,backward=True,balance=True)
    expected,terms,raw,tail,rank=reference(student,t,cfg,multiplier,panel)
    torch.testing.assert_close(g,expected,rtol=4e-7,atol=1e-8)
    torch.testing.assert_close(kl,raw,rtol=1e-6,atol=2e-15)
    torch.testing.assert_close(sums[:3],torch.stack([terms[n].sum() for n in ['dual_linear','tail_quadratic','ranking']]),rtol=2e-6,atol=1e-10)
    torch.testing.assert_close(sums[11:13],torch.stack([tail.double().square().sum(),rank.double().square().sum()]),rtol=3e-6,atol=1e-14)
    flags=topk_preservation_flags(t,student,k=min(100,vocabulary))
    assert sums[13:17].tolist()==[int(flags[n].sum()) for n in ['tie_aware_argmax_changed','strict_order_changed','strict_membership_changed','strict_topk_changed']]


def test_uneven_microbatch_normalization_is_one_aggregate_cotangent():
    torch.manual_seed(71);t=torch.randn(11,137).double().log_softmax(-1);s=(t+torch.randn_like(t)*1e-4).float();cfg=config()
    full=logit_terms_and_cotangent(s,t,config=cfg,multiplier=.02,panel_tokens=11,backward=True)
    parts=[logit_terms_and_cotangent(s[a:b],t[a:b],config=cfg,multiplier=.02,panel_tokens=11,backward=True) for a,b in [(0,3),(3,11)]]
    torch.testing.assert_close(torch.cat([p[0] for p in parts]),full[0],atol=0,rtol=0)
    torch.testing.assert_close(sum(p[1] for p in parts),full[1],atol=1e-10,rtol=1e-12)


def test_relative_mode_and_tie_semantics_are_explicit_not_default():
    t=torch.zeros(2,10).double().log_softmax(-1);s=t.float();cfg=config()
    cfg.ranking_loss_mode='absolute'
    with pytest.raises(ValueError,match='relative_gap'):logit_terms_and_cotangent(s,t,config=cfg,multiplier=0.,panel_tokens=2)
    cfg.ranking_loss_mode='relative_gap';cfg.ranking_tie_policy='legacy'
    with pytest.raises(ValueError,match='exact_teacher_ties'):logit_terms_and_cotangent(s,t,config=cfg,multiplier=0.,panel_tokens=2)
