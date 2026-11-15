from dataclasses import replace
import pytest
import torch
from rewriter.graph_tokenwise import device_tokenwise_terms
from rewriter.tokenwise_reference import TokenwiseConfig
from rewriter.token_preservation import token_preservation_terms

@pytest.mark.parametrize('multiplier',[0.,.37])
def test_captured_row_algebra_matches_reference_exactly(multiplier):
 cfg=TokenwiseConfig(1e-10,1e-10,1e-8,.01,ranking_weight=20435615.123355106)
 kl=torch.tensor([-1e-16,0.,1e-11,1e-10,3e-9],dtype=torch.float64,requires_grad=True)
 rank=torch.tensor([0.,1e-15,1e-12,0.,3e-12],dtype=torch.float64,requires_grad=True)
 parts=device_tokenwise_terms(kl,rank,cfg,torch.tensor(multiplier,dtype=torch.float64))
 ref=token_preservation_terms(kl,multiplier=multiplier,mean_tolerance=cfg.mean_tolerance,tail_tolerance=cfg.tail_tolerance,scale=cfg.scale,rho=cfg.rho,ranking_loss=rank,ranking_weight=cfg.ranking_weight)
 for got,key in zip(parts,('dual_linear','tail_quadratic','ranking')):torch.testing.assert_close(got,ref[key],rtol=0,atol=0)
 g=torch.autograd.grad(sum(parts).sum(),(kl,rank),retain_graph=True);r=torch.autograd.grad(ref['total'].sum(),(kl,rank))
 for a,b in zip(g,r):torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_objective_change_revokes_graph_binding():
 from rewriter.graph_tokenwise import TokenwiseThinGradientGraph
 from rewriter.tokenwise_reference import TokenwiseDual
 cfg=TokenwiseConfig(1e-10,1e-10,1e-8,.01)
 graph=object.__new__(TokenwiseThinGradientGraph);graph.token_config=cfg;graph.binding_valid=True
 graph._check_token_config(TokenwiseDual([6],config=cfg,dual_lr=.01))
 with pytest.raises(ValueError):graph._check_token_config(TokenwiseDual([6],config=replace(cfg,scale=1e-7),dual_lr=.01))
 assert not graph.binding_valid
