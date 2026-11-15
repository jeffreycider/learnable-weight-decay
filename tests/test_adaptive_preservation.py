from dataclasses import replace

import pytest
import torch

from rewriter.adaptive_action import AdaptiveAction, AdaptiveFactors
from rewriter.adaptive_preservation import build_adaptive_proposal, detached_adaptive_credit, group_rms_scales
from rewriter.controller import ControllerSnapshot
from rewriter.grouped_norm import GroupedNormObjective
from rewriter.tokenwise_reference import TokenwiseConfig, TokenwiseDual, tokenwise_panel
from tests.test_factorized_fp32 import setup


def case():
    torch.set_num_threads(1)
    engine, template, _, inputs = setup()
    engine.microbatch_sequences = 1
    shared = torch.tensor(.2, requires_grad=True)
    private = []
    matrices = {}
    for n,f in template.matrices.items():
        u = f.anchors.A.detach().clone().requires_grad_()
        v = f.anchors.B.detach().clone().requires_grad_()
        private.extend((u,v))
        matrices[n] = (u,v)
    def actor(_):
        factors = {}
        for n,(u,v) in matrices.items():
            a = torch.linalg.qr(u, mode='reduced').Q
            b = torch.linalg.qr(v, mode='reduced').Q
            c = torch.eye(2)*(.01*shared.sin())
            factors[n] = AdaptiveFactors(a,c,b)
        return AdaptiveAction(factors,{n:v.detach()*shared for n,v in template.vectors.items()})
    norm = sum(p.square().sum() for p in engine.parameters.values()).sqrt()
    snapshot = ControllerSnapshot({}, {}, torch.zeros(1,4), norm, norm*.01)
    grouped = GroupedNormObjective(engine.parameters, zero_reference_policy='freeze')
    return engine, actor, shared, private, snapshot, grouped, inputs


@pytest.mark.parametrize('radius', [1e-6, 1.])
def test_two_stage_credit_matches_direct_gpt_norm_and_guard_derivatives(radius):
    results, gradients = [], []
    config = TokenwiseConfig(mean_tolerance=1e-10, tail_tolerance=1e-10, scale=1e-5,
                            rho=.01, ranking_weight=.01, ranking_k=10,
                            ranking_tie_policy='exact_teacher_ties', ranking_loss_mode='relative_gap')
    for detach in (False, True):
        engine, actor, shared, private, snapshot, grouped, inputs = case()
        teachers = [engine.cache_teacher(x[None], data_role='synthetic')[0] for x in inputs]
        dual = TokenwiseDual([3], config=config, dual_lr=.01)
        dual.multipliers[3] = .1
        parameters, regularizer, stats, editable, credit = build_adaptive_proposal(
            actor, snapshot, engine, grouped, radius_fraction=radius, detach_credit=detach)
        assert stats['radius_saturated'] == (radius<1e-5)
        for n,p in parameters.items():
            if n not in editable:assert p is engine.parameters[n]
        results.append(tokenwise_panel(engine, parameters, regularizer, inputs, config=config,
                                      dual=dual, state_id=3, actor_credit=credit,
                                      teacher_provider=lambda a,b:teachers[a]))
        gradients.append([None if p.grad is None else p.grad.clone() for p in (shared,*private)])
        assert engine.compute.student_forwards == 2 and engine.compute.student_backwards == 2
        assert all(p.grad is None and not p.requires_grad for p in engine.parameters.values())
    assert results[0] == results[1]
    for actual, expected in zip(*gradients):
        assert (actual is None) == (expected is None)
        if actual is not None:torch.testing.assert_close(actual,expected,rtol=3e-5,atol=2e-6)
    assert any(g is not None and g.norm()>0 for g in gradients[1][1:])


def test_detached_credit_owns_all_factor_storage_and_groups_scale_only_once():
    engine, actor, _, _, snapshot, grouped, _ = case()
    action = actor(snapshot)
    detached, (originals, leaves) = detached_adaptive_credit(action)
    assert len(originals) == 3*len(action.matrices)+len(action.vectors)
    for original,leaf in zip(originals,leaves):
        assert original.data_ptr() != leaf.data_ptr()
        assert leaf.is_leaf and leaf.requires_grad == original.requires_grad
        torch.testing.assert_close(original,leaf,rtol=0,atol=0)
    scales = group_rms_scales(grouped)
    for group,names in grouped.groups.items():
        expected = (grouped.reference_squared[group]/sum(engine.parameters[n].numel() for n in names)).sqrt().float()
        for n in names:torch.testing.assert_close(scales[n], expected)
    with pytest.raises(TypeError, match='fixed-anchor'):
        build_adaptive_proposal(lambda _:setup()[1], snapshot, engine, grouped, radius_fraction=.01)
