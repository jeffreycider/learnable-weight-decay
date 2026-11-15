from dataclasses import replace
import pytest
import torch
from rewriter.tokenwise_reference import TokenwiseConfig,logit_terms_and_cotangent
from rewriter.tokenwise_micro_max import worst_context_loss
from rewriter.relative_rank_tail import maximum_relative_rank_loss
from rewriter.token_preservation import token_preservation_terms


def config():
    return TokenwiseConfig(1e-10,1e-10,1e-8,.01,ranking_weight=100.,ranking_k=100,ranking_tie_policy='exact_teacher_ties',ranking_loss_mode='relative_gap_micro_max',loss_backend='reference_micro_max_v1')


def dense(s,t,c,panel,multiplier):
    s=s.detach().clone().requires_grad_();lp=s.double().log_softmax(-1)
    kl=(t.exp()*(t-lp)).sum(-1)
    terms=token_preservation_terms(kl,multiplier=multiplier,mean_tolerance=c.mean_tolerance,tail_tolerance=c.tail_tolerance,scale=c.scale,rho=c.rho)
    worst=maximum_relative_rank_loss(t,s,k=min(c.ranking_k,s.shape[-1]),margin_fraction=c.margin_fraction).flatten().max(dim=0).values
    loss=terms['total'].sum()/panel+c.ranking_weight*worst*s.shape[0]/panel
    return torch.autograd.grad(loss,s)[0],worst.detach(),kl.detach()


@pytest.mark.parametrize('vocab',[10,137])
def test_complete_logit_cotangent_matches_dense_one_max_subgradient(vocab):
    torch.manual_seed(21);t=torch.randn(9,vocab).double().log_softmax(-1);s=(t+torch.randn_like(t)*2e-4).float();c=config();diag={}
    g,parts,kl=logit_terms_and_cotangent(s,t,config=c,multiplier=.3,panel_tokens=18,chunk_tokens=3,diagnostics=diag)
    expected,maximum,raw=dense(s,t,c,18,.3)
    torch.testing.assert_close(g,expected,rtol=3e-6,atol=2e-4)
    torch.testing.assert_close(kl,raw,rtol=1e-5,atol=2e-15)
    torch.testing.assert_close(parts[2],maximum*100*9)
    assert diag['micro_weight']==.5 and diag['unweighted_max_relative_rank']==maximum.item()
    read,readparts,_=logit_terms_and_cotangent(s,t,config=c,multiplier=.3,panel_tokens=18,chunk_tokens=5,backward=False)
    assert read is None;torch.testing.assert_close(readparts,parts,rtol=1e-14,atol=1e-14)


def test_exact_worst_context_tie_chooses_first_row_and_chunk_invariance():
    t=torch.tensor([[3.,2.,1.,0.]],dtype=torch.double).log_softmax(-1).repeat(7,1)
    s=torch.tensor([[3.,.9,1.,0.]],dtype=torch.float32).repeat(7,1)
    results=[]
    for chunk in (1,3,7):
        loss,selected,index=worst_context_loss(s,t,k=3,margin_fraction=.5,chunk_tokens=chunk)
        results.append((loss.detach(),torch.autograd.grad(loss,selected)[0],index))
        assert index.item()==0
    for item in results[1:]:
        for a,b in zip(results[0],item):torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_positive_tiny_gaps_are_not_reclassified_as_ties():
    t=torch.tensor([[3e-12,2e-12,1e-12,0.]],dtype=torch.double).log_softmax(-1)
    s=torch.tensor([[3e-12,.9e-12,1e-12,0.]],dtype=torch.float32)
    loss,row,index=worst_context_loss(s,t,k=3,margin_fraction=.5,chunk_tokens=1)
    gradient=torch.autograd.grad(loss,row)[0]
    assert loss>.3 and torch.isfinite(gradient).all() and gradient.abs().max()>1e11
    ties=torch.zeros(1,4,dtype=torch.double).log_softmax(-1)
    loss,row,_=worst_context_loss(s,ties,k=4,margin_fraction=.5)
    assert loss==0 and torch.count_nonzero(torch.autograd.grad(loss,row)[0])==0


def test_explicit_mean_micro_normalization_and_nondivisible_panel_rejected():
    c=config();torch.manual_seed(22);t=torch.randn(8,10).double().log_softmax(-1);s=(t+1e-3*torch.randn_like(t)).float()
    micros=[logit_terms_and_cotangent(s[a:a+4],t[a:a+4],config=c,multiplier=0.,panel_tokens=8) for a in (0,4)]
    expected=100*sum(maximum_relative_rank_loss(t[a:a+4],s[a:a+4],k=10).max() for a in (0,4))/2
    torch.testing.assert_close(sum(x[1][2] for x in micros)/8,expected)
    with pytest.raises(ValueError,match='complete microbatches'):logit_terms_and_cotangent(s,t,config=c,multiplier=0.,panel_tokens=9)
    with pytest.raises(ValueError,match='micro-max'):replace(c,loss_backend='reference')


def test_selected_rank_cotangent_finite_difference():
    t=torch.tensor([[3.,2.,1.,0.],[4.,3.,1.,0.]],dtype=torch.double).log_softmax(-1)
    s=torch.tensor([[3.1,.8,1.2,-.2],[4.1,3.1,1.1,.1]],dtype=torch.float32)
    loss,row,index=worst_context_loss(s,t,k=3,margin_fraction=.5)
    grad=torch.autograd.grad(loss,row)[0];eps=1e-3
    for j in range(4):
        a=s.clone();b=s.clone();a[index,j]+=eps;b[index,j]-=eps
        finite=(worst_context_loss(a,t,k=3,margin_fraction=.5,backward=False)[0]-worst_context_loss(b,t,k=3,margin_fraction=.5,backward=False)[0])/(2*eps)
        assert float(grad[0,j])==pytest.approx(float(finite),rel=2e-4,abs=2e-4)
