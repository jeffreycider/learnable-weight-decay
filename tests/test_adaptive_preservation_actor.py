"""Independent full actor -> realized tiny GPT -> factor-credit regression."""
import pytest
import torch

from rewriter.adaptive_coordinate_chisel import AdaptiveCoordinateChisel
from rewriter.adaptive_preservation import build_adaptive_proposal
from rewriter.algebra import Anchors
from rewriter.controller import (ControllerSnapshot, MatrixObservation, VectorObservation,
                                 coordinate_features, vector_features)
from rewriter.grouped_norm import GroupedNormObjective
from rewriter.tokenwise_reference import TokenwiseConfig, TokenwiseDual, tokenwise_panel
from tests.test_factorized_fp32 import setup


def actual_actor_case():
    torch.set_num_threads(1)
    engine, _, _, inputs = setup()
    engine.microbatch_sequences = 1
    # A deliberately zero reference group must get no write; other positive
    # groups still exercise both embedding orientation and all vector roles.
    engine.parameters['blocks.0.attn.proj.weight'].zero_()
    grouped = GroupedNormObjective(engine.parameters, zero_reference_policy='freeze')
    generator = torch.Generator().manual_seed(91)
    matrices, vectors = {}, {}
    for name, parameter in engine.parameters.items():
        weight = parameter.T if engine.transpose_storage.get(name, False) else parameter
        if weight.ndim == 2:
            anchors = Anchors.random(*weight.shape, 2, generator=generator)
            wb, wta = weight @ anchors.B, weight.T @ anchors.A
            matrices[name] = MatrixObservation(anchors,
                coordinate_features([wb, wb*.1, wb*.2, wb*.3], anchors.A, torch.ones(len(weight)), [0.]*10),
                coordinate_features([wta, wta*.1, wta*.2, wta*.3], anchors.B, torch.ones(weight.shape[1]), [0.]*10),
                (wb.clone(), wta.clone()))
        else:
            vectors[name] = VectorObservation(vector_features([weight, weight*.1, weight*.2, weight*.3], [0.]*10), weight.clone())
    norm = sum(p.square().sum() for p in engine.parameters.values()).sqrt()
    snapshot = ControllerSnapshot(matrices, vectors, torch.randn(2, 12, generator=generator), norm, norm*.01)
    torch.manual_seed(133)
    actor = AdaptiveCoordinateChisel(rank=2, trajectory_dim=12, model_dim=128, num_layers=1, latent_tokens=2)
    private = actor.make_private(snapshot, state_key=('test',0,6), seed=77)
    with torch.no_grad():
        for head in actor.output_heads:
            head.weight.normal_(std=.001)
        actor.blocks[0].attn.proj.weight.normal_(std=.001)
    return engine, actor, private, snapshot, grouped, inputs


@pytest.mark.parametrize('radius', [1e-6, 1.])
def test_real_adaptive_actor_tiny_gpt_detached_credit_matches_direct(radius):
    outputs, gradients = [], []
    for detached in (False, True):
        engine, actor, private, snapshot, grouped, inputs = actual_actor_case()
        cached = [engine.cache_teacher(x[None], data_role='synthetic')[0] for x in inputs]
        config = TokenwiseConfig(mean_tolerance=1e-10, tail_tolerance=1e-10, scale=1e-5,
            rho=.01, ranking_weight=.01, ranking_k=10,
            ranking_tie_policy='exact_teacher_ties', ranking_loss_mode='relative_gap')
        dual = TokenwiseDual([6], config=config, dual_lr=.01)
        dual.multipliers[6] = .1
        parameters, regularizer, stats, editable, credit = build_adaptive_proposal(
            lambda snap:actor(snap,private), snapshot, engine, grouped,
            radius_fraction=radius, detach_credit=detached)
        assert stats['radius_saturated'] == (radius < 1e-5)
        assert 'blocks.0.attn.proj.weight' in grouped.frozen_parameter_names
        for name in set(engine.parameters)-editable:
            assert parameters[name] is engine.parameters[name]
        # Exercise the actual stored embedding orientation, not only a square
        # action: the snapshot/action view is transpose of storage.
        assert parameters['embed.weight'].shape == (32,128)
        outputs.append(tokenwise_panel(engine, parameters, regularizer, inputs,
            config=config, dual=dual, state_id=6, actor_credit=credit,
            teacher_provider=lambda a,b:cached[a]))
        gradients.append({prefix+name:None if p.grad is None else p.grad.detach().clone()
                          for prefix,module in [('shared.',actor),('private.',private)]
                          for name,p in module.named_parameters()})
        assert engine.compute.student_forwards == engine.compute.student_backwards == 2
        assert all(not p.requires_grad and p.grad is None for p in engine.parameters.values())
    assert outputs[0] == outputs[1]
    for name in gradients[0]:
        a,b = gradients[0][name],gradients[1][name]
        assert (a is None) == (b is None), name
        if a is not None:
            torch.testing.assert_close(a,b,rtol=2e-4,atol=2e-5,msg=name)
    assert gradients[1]['shared.matrix_output.weight'].norm()>0
    assert gradients[1]['private.matrix_rows.0.raw'].norm()>0
    assert gradients[1]['private.matrix_columns.0.raw'].norm()>0


def test_zero_reference_matrix_has_no_detached_write_credit():
    engine, actor, private, snapshot, grouped, _ = actual_actor_case()
    _, regularizer, _, _, credit = build_adaptive_proposal(
        lambda snap:actor(snap,private), snapshot, engine, grouped,
        radius_fraction=.01)
    regularizer.backward()
    name = 'blocks.0.attn.proj.weight'
    matrix_index = list(snapshot.matrices).index(name)
    # Its maps may still learn through globally useful READ observations, but
    # an inactive WRITE must not manufacture zero cotangents/Adam clock ticks.
    frozen_leaves = credit[1][3*matrix_index:3*matrix_index+3]
    assert all(leaf.grad is None for leaf in frozen_leaves)
