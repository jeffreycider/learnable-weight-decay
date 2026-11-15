import copy

import pytest
import torch

from rewriter.gradient_components import GradientComponents


def test_decomposition_includes_state_weight_and_excludes_accumulated_gradients():
    p = torch.nn.Parameter(torch.tensor([1., 2.], dtype=torch.float64))
    unused = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
    p.grad = torch.tensor([7., -2.], dtype=torch.float64)
    before = p.grad.clone()
    norm = p.square().sum()
    preservation = -3 * p[0] + p[1]
    observer = GradientComponents(norm, dict(active=[p], unused=[unused]), weight=.25)
    assert torch.equal(p.grad, before) and unused.grad is None
    (.25 * (norm + preservation)).backward()
    combined = p.grad.clone()
    result = observer.finish()['roles']
    assert torch.equal(p.grad, combined) and unused.grad is None
    assert result['active']['norm_gradient_l2'] == pytest.approx(1.25 ** .5)
    assert result['active']['preservation_gradient_l2'] == pytest.approx(.625 ** .5)
    assert result['active']['combined_gradient_l2'] == pytest.approx(1.625 ** .5)
    assert result['active']['norm_preservation_dot'] == pytest.approx(-.125)
    assert result['unused']['norm_preservation_cosine'] is None
    with pytest.raises(RuntimeError, match='one-shot'):
        observer.finish()


def test_rejects_overlapping_roles_and_parameter_changes():
    p = torch.nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError, match='disjoint'):
        GradientComponents(p.square().sum(), dict(a=[p], b=[p]))
    observer = GradientComponents(p.square().sum(), dict(a=[p]))
    with torch.no_grad():
        p.add_(1)
    with pytest.raises(RuntimeError, match='changed'):
        observer.finish()


def test_detached_factor_bridge_measures_chisel_gradient_without_populating_leaves():
    p = torch.nn.Parameter(torch.tensor([2.], dtype=torch.float64))
    action = 3 * p
    leaf = action.detach().clone().requires_grad_()
    norm = leaf.square().sum()
    observer = GradientComponents(norm, dict(actor=[p]), weight=.5,
                                  actor_credit=([action], [leaf]))
    assert p.grad is None and leaf.grad is None
    # Norm contributes .5 * 2 * 6 * 3 = 18; preservation contributes -3.
    (.5 * (norm - 2 * leaf.sum())).backward()
    action.backward(leaf.grad)
    result = observer.finish()['roles']['actor']
    assert result['norm_gradient_l2'] == 18
    assert result['preservation_gradient_l2'] == 3
    assert result['combined_gradient_l2'] == 15
    assert result['norm_preservation_cosine'] == -1


@pytest.mark.parametrize('backend', ['reference', 'direct_streamed'])
def test_observer_keeps_tiny_gpt_loss_gradients_rng_and_compute_unchanged(backend):
    from tests.test_adaptive_preservation_actor import actual_actor_case
    from rewriter.adaptive_coordinate_chisel import AdaptiveCoordinateChisel
    from rewriter.edit_scope import editable_names, editable_weight_norm_sq
    from rewriter.tokenwise_reference import TokenwiseConfig, TokenwiseDual
    from scripts.diagnose_cpu_norm_preservation import LocalCPUContexts
    torch.set_num_threads(1)
    outcomes = []
    for enabled in (False, True):
        engine, _, _, snapshot, grouped, inputs = actual_actor_case()
        inputs = inputs[:1, :4]
        torch.manual_seed(659)
        actor = AdaptiveCoordinateChisel(rank=2, trajectory_dim=12, model_dim=128,
            num_layers=1, latent_tokens=2, fresh_weight_core=True,
            factor_precision='fp64', residual_precision='fp32')
        private = actor.make_private(snapshot, state_key=('test', 0, 6), seed=77)
        with torch.no_grad():
            for head in actor.output_heads:
                head.weight.normal_(std=.001)
        teacher = engine.cache_teacher(inputs, data_role='synthetic')[0]
        editable = editable_names(engine.parameters, 'grouped') - grouped.frozen_parameter_names
        context = dict(engine=engine, domain='gpt', local_id=6, snapshot=snapshot,
                       grouped=grouped, inputs=inputs, ids=[1], teacher=lambda a,b: teacher,
                       norm_squared=editable_weight_norm_sq(engine.parameters, editable))
        config = TokenwiseConfig(mean_tolerance=1e-10, tail_tolerance=1e-10,
            scale=1e-5, rho=.01, ranking_weight=.01, ranking_k=10,
            ranking_tie_policy='exact_teacher_ties', ranking_loss_mode='relative_gap')
        dual = TokenwiseDual([6], config=config, dual_lr=.01)
        dual.multipliers[6] = .1
        contexts = LocalCPUContexts(backend)
        # Simulate an earlier panel's accumulated shared gradient.
        for p in actor.parameters():
            p.grad = torch.full_like(p, .01)
        row = contexts.measure(context, actor, private, config, dual, learning=True,
                               weight=.5, gradient_components=enabled)
        components = row.pop('gradient_components', None)
        if enabled:
            assert components['extra_gpt_forwards'] == components['extra_gpt_backwards'] == 0
            assert components['roles']['heads']['norm_gradient_l2'] > 0
            assert components['roles']['heads']['preservation_gradient_l2'] > 0
        outcomes.append((row, copy.deepcopy(contexts.compute), torch.get_rng_state(),
                         [None if p.grad is None else p.grad.clone()
                          for module in (actor, private) for p in module.parameters()]))
    a, b = outcomes
    assert a[:2] == b[:2]
    assert a[1]['student_forwards'] == a[1]['student_backwards'] == 1
    assert torch.equal(a[2], b[2])
    for x, y in zip(a[3], b[3]):
        assert (x is None) == (y is None)
        if x is not None:
            assert torch.equal(x, y)
