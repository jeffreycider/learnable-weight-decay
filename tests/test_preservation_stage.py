"""The repair stage changes optimization pressure, never its edit or readout."""
from types import SimpleNamespace
import pytest
import torch
from tests.test_transformer_action import fixture, FactorActor
from rewriter.function_preservation import FunctionalPreservation
from rewriter.grouped_norm import GroupedNormObjective
from scripts.train_multistate_chisel import build_proposal


def proposal(coefficient):
    parameters,snapshot=fixture(torch.float64)
    actor=FactorActor()
    engine=SimpleNamespace(parameters=parameters,
        transpose_storage={n:n=='embed.weight' for n,p in parameters.items() if p.ndim==2})
    engine.edited_parameters=lambda action,scale:FunctionalPreservation.edited_parameters(engine,action,scale)
    grouped=GroupedNormObjective(parameters,zero_reference_policy='freeze')
    return actor,parameters,build_proposal(actor,snapshot,engine,grouped,radius_fraction=10.,shrinkage_coefficient=coefficient)


def test_preservation_stage_keeps_nonzero_edits_and_actual_group_readout():
    _,base,normal=proposal(1.)
    _,_,repair=proposal(0.)
    for name in base:torch.testing.assert_close(normal[0][name],repair[0][name],rtol=0,atol=0)
    assert repair[2]['realized_edit_norm']>0
    assert repair[2]['group_l2_percentage_reduction']==normal[2]['group_l2_percentage_reduction']
    assert repair[2]['unweighted_norm_term']==normal[2]['norm_term']
    assert repair[1].item()==repair[2]['norm_term']==0
    norm_grad=torch.autograd.grad(repair[1],repair[4][1],allow_unused=True)
    assert all(g is None or not bool(g.any()) for g in norm_grad)


@pytest.mark.parametrize('coefficient',[0.,.2,1.])
def test_preservation_cotangent_stays_connected_independently_of_norm_reward(coefficient):
    actor,base,(edited,regularizer,stats,_,credit)=proposal(coefficient)
    preservation=sum((edited[n]-base[n]).square().sum() for n in edited)
    factors=credit[1]
    expected=torch.autograd.grad(preservation,factors,retain_graph=True,allow_unused=True)
    total=torch.autograd.grad(regularizer+preservation,factors,retain_graph=True,allow_unused=True)
    norm=torch.autograd.grad(regularizer,factors,allow_unused=True)
    for a,b,c in zip(total,expected,norm):
        if b is not None:torch.testing.assert_close(a,b+(0 if c is None else c),rtol=1e-10,atol=1e-10)
    torch.autograd.backward(credit[0],[torch.zeros_like(x) if g is None else g for x,g in zip(factors,total)])
    assert torch.isfinite(actor.x.grad).all() and actor.x.grad.norm()>0


@pytest.mark.parametrize('coefficient',[-1.,float('nan'),float('inf')])
def test_bad_stage_coefficient_rejected(coefficient):
    with pytest.raises(ValueError,match='shrinkage coefficient'):proposal(coefficient)
