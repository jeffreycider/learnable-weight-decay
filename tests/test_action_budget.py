"""Guard algebra and exact delayed credit; no scored GPT evaluations."""
from dataclasses import replace

import pytest
import torch

from rewriter.action_budget import base_motion_radius, restrict_global_scale
from rewriter.lag_one import LagOneHypergradient
from rewriter.transformer_action import prepare_transformer_action
from test_transformer_action import FactorActor, fixture


def test_disabled_is_exact_identity_of_existing_graph():
    scale = torch.tensor(.4, requires_grad=True)
    assert restrict_global_scale(None, scale, None) is scale


def test_radius_is_immutable_and_shrinks_with_actual_motion():
    motion = torch.tensor(4.)
    radius = base_motion_radius(motion, fraction=.25)
    motion.zero_()
    assert radius.item() == 1.
    assert base_motion_radius(motion, fraction=.25).item() == 0.


@pytest.mark.parametrize('radius_value', [0., .1, 100.])
def test_global_bound_and_finite_gradient_at_zero_action(radius_value):
    action = torch.zeros(7, dtype=torch.float64, requires_grad=True)
    scale = restrict_global_scale(action.square().sum(), action.new_tensor(1.),
                                  action.new_tensor(radius_value))
    gradient, = torch.autograd.grad((scale * action).sum(), action)
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient, torch.full_like(action, 0. if radius_value == 0 else 1.))


@pytest.mark.parametrize('existing', [.01, 1.])
def test_global_action_preserves_cancellation_and_obeys_stricter_bound(existing):
    action = torch.tensor([3., -3., 4., -4.], dtype=torch.float64, requires_grad=True)
    radius = action.new_tensor(2.)
    scale = restrict_global_scale(action.square().sum(), action.new_tensor(existing), radius)
    committed = scale * action
    assert float(committed.detach().norm()) <= 2.
    assert float(scale.detach()) <= existing
    assert committed.sum().item() == 0.  # One scalar preserves cross-layer cancellation.
    torch.autograd.gradcheck(lambda x: restrict_global_scale(x.square().sum(), x.new_tensor(existing),
                                                            radius) * x, (action,))


@pytest.mark.parametrize('fraction', [.0001, 10.])
def test_budgeted_delayed_vjp_equals_dense_rewrite_and_finite_difference(fraction):
    parameters, snapshot = fixture(torch.float64)
    actor = FactorActor()
    prepared = prepare_transformer_action(actor, snapshot, parameters, radius_fraction=10.)
    radius = base_motion_radius(torch.tensor(1., dtype=torch.float64), fraction=fraction)
    scale = restrict_global_scale(prepared.action.norm_sq(), prepared.scale, radius)
    prepared = replace(prepared, scale=scale)
    proposal = prepared.materialize()
    targets = {name: torch.sin(p) for name, p in parameters.items()}
    def objective(values):
        return sum(.5 * (p-targets[name]).square().sum() for name, p in values.items())
    expected, = torch.autograd.grad(objective(proposal), actor.x, retain_graph=True)
    live = {name: p.detach().clone().requires_grad_() for name, p in proposal.items()}
    grads = torch.autograd.grad(objective(live), tuple(live.values()))
    projected = prepared.project_next_gradients(dict(zip(live, grads)),
                        global_token_count=1, loss_reduction='global_mean')
    lag = LagOneHypergradient()
    prepared.record_executed(lag, action_step=999, actor=actor)
    observed = lag.consume(projected, gradient_step=1000)
    torch.testing.assert_close(observed.gradients['x'], expected, rtol=1e-9, atol=1e-11)
    actual_norm = sum((proposal[n]-p).square().sum() for n,p in parameters.items()).sqrt()
    assert actual_norm <= radius * (1+1e-10)
    initial = actor.x.detach().clone()
    finite=[]
    for i in range(len(initial)):
        losses=[]
        for sign in (-1, 1):
            with torch.no_grad():
                actor.x.copy_(initial); actor.x[i].add_(sign*1e-5)
            p = prepare_transformer_action(actor, snapshot, parameters, radius_fraction=10.)
            p = replace(p, scale=restrict_global_scale(p.action.norm_sq(), p.scale, radius))
            losses.append(float(objective(p.materialize()).detach()))
        finite.append((losses[1]-losses[0])/2e-5)
    torch.testing.assert_close(torch.tensor(finite, dtype=torch.float64), expected, rtol=2e-4, atol=2e-8)


def test_reused_radius_buffer_cannot_change_pending_gradient():
    x = torch.tensor([2., 3.], requires_grad=True)
    radius = torch.tensor(1.)
    scale = restrict_global_scale(x.square().sum(), x.new_tensor(1.), radius)
    expected, = torch.autograd.grad(scale*x[0], x, retain_graph=True)
    radius.zero_()
    actual, = torch.autograd.grad(scale*x[0], x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('bad', [-1., float('nan'), float('inf')])
def test_invalid_motion_rejected(bad):
    with pytest.raises(ValueError):
        base_motion_radius(torch.tensor(bad), fraction=.25)


def test_no_higher_order_base_information():
    with pytest.raises(ValueError):
        base_motion_radius(torch.tensor(1., requires_grad=True), fraction=.25)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_zero_norm_floor_is_evaluated_in_at_least_fp32(dtype):
    squared = torch.zeros((), dtype=dtype, requires_grad=True)
    scale = restrict_global_scale(squared, squared.new_tensor(1.), squared.new_tensor(1.))
    gradient, = torch.autograd.grad(scale, squared)
    assert scale.item() == 1. and gradient.item() == 0.
    assert scale.dtype == torch.float32


def test_stricter_existing_scale_keeps_its_derivative():
    x = torch.tensor(.1, requires_grad=True)
    scale = restrict_global_scale(x.new_tensor(10.), x.square(), x.new_tensor(100.))
    derivative, = torch.autograd.grad(scale, x)
    torch.testing.assert_close(derivative, 2*x)
