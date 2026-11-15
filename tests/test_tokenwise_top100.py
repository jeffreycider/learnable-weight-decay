from dataclasses import asdict, replace
import pytest
import torch
from rewriter.tokenwise_reference import TokenwiseConfig,TokenwiseDual,logit_terms_and_cotangent,TOKEN_SUM_FIELDS
from rewriter.ranking_loss import tie_aware_topk_margin_loss
from rewriter.token_preservation import token_preservation_terms
from rewriter.rank_preservation import topk_preservation_flags

@pytest.mark.parametrize('vocab,k,chunk',[(10,100,1),(137,100,3),(137,100,128)])
def test_new_topk_matches_dense_objective_and_strict_counts(vocab,k,chunk):
    torch.manual_seed(51)
    t=torch.randn(2,4,vocab).double().log_softmax(-1)
    s=(t.float()+.15*torch.randn(2,4,vocab)).requires_grad_()
    c=TokenwiseConfig(.001,.002,.01,.03,.4,ranking_k=k,ranking_tie_policy='exact_teacher_ties')
    g,parts,raw=logit_terms_and_cotangent(s,t,config=c,multiplier=.7,panel_tokens=16,chunk_tokens=chunk)
    kl=(t.exp()*(t-s.double().log_softmax(-1))).sum(-1)
    rank=tie_aware_topk_margin_loss(t,s.double(),k=min(k,vocab))
    terms=token_preservation_terms(kl,multiplier=.7,mean_tolerance=c.mean_tolerance,tail_tolerance=c.tail_tolerance,scale=c.scale,rho=c.rho,ranking_loss=rank,ranking_weight=c.ranking_weight)
    expected=torch.autograd.grad(terms['total'].sum()/16,s)[0]
    torch.testing.assert_close(g,expected,rtol=2e-6,atol=2e-7)
    flags=topk_preservation_flags(t,s,k=min(k,vocab))
    for i,name in enumerate(('tie_aware_argmax_changed','strict_order_changed','strict_membership_changed','strict_topk_changed'),13):
        assert parts[i]==flags[name].sum()


def test_exact_ties_do_not_create_strict_failure_or_margin_loss():
    t=torch.tensor([[3.,3.,2.,2.,1.]],dtype=torch.float64).log_softmax(-1)
    s=torch.tensor([[4.,3.9,2.8,2.9,1.]],dtype=torch.float32)
    c=TokenwiseConfig(1.,1.,1.,.01,1.,ranking_k=3,ranking_tie_policy='exact_teacher_ties')
    _,parts,_=logit_terms_and_cotangent(s,t,config=c,multiplier=0.,panel_tokens=1,backward=False)
    assert parts[2]==0 and parts[16]==0


def test_old_saved_config_canonicalizes_only_to_old_scope():
    c=TokenwiseConfig(1e-8,1e-8,1e-8,.01)
    d=TokenwiseDual([1],config=c,dual_lr=.1);state=d.state_dict()
    state['config'].pop('ranking_k');state['config'].pop('ranking_tie_policy')
    d.load_state_dict(state)
    new=replace(c,ranking_k=100,ranking_tie_policy='exact_teacher_ties')
    with pytest.raises(ValueError,match='resume'):TokenwiseDual([1],config=new,dual_lr=.1).load_state_dict(state)


def test_unqualified_backend_cannot_silently_change_scope():
    c=TokenwiseConfig(1e-8,1e-8,1e-8,.01,loss_backend='tiled_fp64_v1',ranking_k=100,ranking_tie_policy='exact_teacher_ties')
    with pytest.raises(ValueError,match='legacy top10'):
        logit_terms_and_cotangent(torch.zeros(1,137),torch.zeros(1,137,dtype=torch.float64),config=c,multiplier=0,panel_tokens=1)
