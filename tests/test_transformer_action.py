from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rewriter.algebra import Anchors, Factors, RawAction
from rewriter.controller import (ControllerSnapshot, MatrixObservation, VectorObservation,
                                 coordinate_features, vector_features)
from rewriter.function_preservation import FunctionalPreservation
from rewriter.grouped_norm import GroupedNormObjective
from rewriter.lag_one import LagOneHypergradient
from rewriter.transformer_action import prepare_transformer_action
from rewriter.transformer_chisel import TransformerChisel
from scripts.train_multistate_chisel import build_proposal


def fixture(dtype=torch.float32):
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(811)
    shapes = {'embed.weight': (7, 4), 'proj.weight': (7, 4), 'proj.bias': (7,)}
    for role, shape in [('attn.q', (4, 4)), ('attn.k', (4, 4)), ('attn.v', (4, 4)),
                        ('attn.proj', (4, 4)), ('mlp.fc', (6, 4)), ('mlp.proj', (4, 6))]:
        shapes[f'blocks.0.{role}.weight'] = shape
        shapes[f'blocks.0.{role}.bias'] = (shape[0],)
    shapes['norm1.gains'] = (4,)
    parameters = {n: torch.randn(s, generator=generator, dtype=dtype) for n, s in shapes.items()}
    parameters['proj.bias'].zero_()  # Explicit zero-reference group.
    matrices, vectors = {}, {}
    for name, p in parameters.items():
        w = p.T if name == 'embed.weight' else p
        if p.ndim == 2:
            a = Anchors.random(*w.shape, 2, generator=generator, dtype=dtype)
            wb, wta = w @ a.B, w.T @ a.A
            rows = coordinate_features([wb] * 4, a.A, torch.ones(len(w)), [0.] * 10)
            cols = coordinate_features([wta] * 4, a.B, torch.ones(w.shape[1]), [0.] * 10)
            matrices[name] = MatrixObservation(a, rows, cols, (wb.clone(), wta.clone()))
        else:
            vectors[name] = VectorObservation(vector_features([w] * 4, [0.] * 10), w.clone())
    norm = sum(p.square().sum() for p in parameters.values()).sqrt()
    snapshot = ControllerSnapshot(matrices, vectors, torch.randn(2, 12, generator=generator), norm, norm * .01)
    return parameters, snapshot


class FactorActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.x = nn.Parameter(torch.tensor([.2, -.15, .3], dtype=torch.float64))

    def forward(self, snapshot):
        matrices, vectors = {}, {}
        for name, obs in snapshot.matrices.items():
            a = obs.anchors
            p = torch.arange(a.A.numel(), dtype=self.x.dtype).reshape_as(a.A) / 10
            q = torch.arange(a.B.numel(), dtype=self.x.dtype).reshape_as(a.B).flip(0) / 11
            matrices[name] = Factors.canonical(p * self.x[0] + p.flip(0) * self.x[1].square(),
                                                q * self.x[2] + q.flip(0) * self.x[0], a)
        for name, obs in snapshot.vectors.items():
            v = torch.linspace(-1, 1, len(obs.weight), dtype=self.x.dtype)
            vectors[name] = v * self.x[1] + v.square() * self.x[2]
        return RawAction(matrices, vectors)


@pytest.mark.parametrize('radius_fraction', [.0001, 10.])
def test_actual_transformer_proposal_matches_offline_and_keeps_actor_graph(radius_fraction):
    parameters, snapshot = fixture()
    torch.manual_seed(29)
    actor = TransformerChisel(rank=2, trajectory_dim=12, model_dim=128, num_layers=1, patch_size=4)
    with torch.no_grad():
        for head in actor.output_heads:
            head.weight.normal_(std=.02)
    engine = SimpleNamespace(parameters=parameters,
        transpose_storage={n: n == 'embed.weight' for n, p in parameters.items() if p.ndim == 2})
    engine.edited_parameters = lambda action, scale: FunctionalPreservation.edited_parameters(engine, action, scale)
    grouped = GroupedNormObjective(parameters, zero_reference_policy='freeze')
    offline, _, stats, _, _ = build_proposal(actor, snapshot, engine, grouped, radius_fraction=radius_fraction)
    prepared = prepare_transformer_action(actor, snapshot, parameters, radius_fraction=radius_fraction)
    actual = prepared.materialize()
    assert float(prepared.scale.detach()) == stats['global_scale']
    for name in parameters:
        torch.testing.assert_close(actual[name], offline[name], rtol=0, atol=0)
    assert actual['norm1.gains'] is parameters['norm1.gains']
    assert actual['proj.bias'] is parameters['proj.bias']
    assert 'head.bias' in prepared.zero_reference_groups
    gradient = torch.autograd.grad(sum(p.square().sum() for p in actual.values()), actor.matrix_output.weight)[0]
    assert gradient.norm() > 0  # No disconnected factor leaves.


@pytest.mark.parametrize('radius_fraction', [.0001, 10.])
def test_lag_one_vjp_matches_dense_next_loss_and_finite_difference(radius_fraction):
    parameters, snapshot = fixture(torch.float64)
    actor = FactorActor()
    prepared = prepare_transformer_action(actor, snapshot, parameters, radius_fraction=radius_fraction)
    proposal = prepared.materialize()
    assert (float(prepared.scale.detach()) < 1) == (radius_fraction < 1)
    targets = {n: torch.sin(p) for n, p in parameters.items()}
    def objective(values):
        return sum(.5 * (p - targets[n]).square().sum() for n, p in values.items())
    expected, = torch.autograd.grad(objective(proposal), (actor.x,), retain_graph=True)
    live = {n: p.detach().clone().requires_grad_() for n, p in proposal.items()}
    ordinary = torch.autograd.grad(objective(live), tuple(live.values()))
    gradients = {n: g * 8192 for n, g in zip(live, ordinary)}
    projected = prepared.project_next_gradients(gradients, global_token_count=8192)
    normalized = prepared.project_next_gradients(dict(zip(live, ordinary)),
        global_token_count=8192, loss_reduction='global_mean')
    for name in projected['matrices']:
        for a, b in zip(projected['matrices'][name], normalized['matrices'][name]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    bridge = LagOneHypergradient()
    prepared.record_executed(bridge, action_step=999, actor=actor)
    result = bridge.consume(projected, gradient_step=1000)
    torch.testing.assert_close(result.gradients['x'], expected, rtol=2e-10, atol=1e-11)
    # Non-square lookup orientation and editable vector edits are both exercised.
    assert prepared.action.matrices['embed.weight'].P.shape == (4, 2)
    assert prepared.action.vectors['blocks.0.attn.q.bias'].requires_grad
    initial = actor.x.detach().clone()
    finite = []
    for i in range(3):
        values = []
        for sign in (-1, 1):
            with torch.no_grad():
                actor.x.copy_(initial); actor.x[i].add_(sign * 1e-6)
            action = prepare_transformer_action(actor, snapshot, parameters, radius_fraction=radius_fraction)
            values.append(float(objective(action.materialize()).detach()))
        finite.append((values[1] - values[0]) / 2e-6)
    torch.testing.assert_close(torch.tensor(finite, dtype=torch.float64), expected, rtol=2e-5, atol=2e-8)


def test_anchor_refresh_uses_owned_old_anchors_and_mutations_fail_closed():
    parameters, snapshot = fixture(torch.float64)
    actor = FactorActor()
    prepared = prepare_transformer_action(actor, snapshot, parameters)
    before = {n: p.detach().clone() for n, p in prepared.materialize().items()}
    snapshot.matrices['embed.weight'].anchors.A.add_(20)
    for name, p in prepared.materialize().items():
        torch.testing.assert_close(p, before[name], rtol=0, atol=0)
    bridge = LagOneHypergradient()
    prepared.record_executed(bridge, action_step=31, actor=actor)
    gradients = {n: torch.ones_like(p) for n, p in parameters.items()}
    projected = prepared.project_next_gradients(gradients, global_token_count=1)
    prepared.snapshot.matrices['embed.weight'].anchors.A.add_(1)
    with pytest.raises(RuntimeError, match='mutated'):
        bridge.consume(projected, gradient_step=32)
    parameters, snapshot = fixture(torch.float64)
    prepared = prepare_transformer_action(actor, snapshot, parameters)
    with torch.no_grad(): actor.x.add_(1)
    with pytest.raises(RuntimeError, match='actor .* mutated'):
        prepared.record_executed(LagOneHypergradient(), action_step=1, actor=actor)


def test_readonly_scalar_zero_groups_and_gradient_units_validation():
    parameters, snapshot = fixture(torch.float64)
    parameters['temperature'] = torch.tensor(-0., dtype=torch.float64)
    actor = FactorActor()
    prepared = prepare_transformer_action(actor, snapshot, parameters)
    assert prepared.materialize()['temperature'] is parameters['temperature']
    with pytest.raises(ValueError, match='global_token_count'):
        prepared.project_next_gradients(parameters, global_token_count=0)
    with pytest.raises(ValueError, match='loss_reduction'):
        prepared.project_next_gradients(parameters, global_token_count=1, loss_reduction='ddp_mean')
    for name in prepared.editable:
        parameters[name].zero_()
    zero = prepare_transformer_action(actor, snapshot, parameters)
    assert len(zero.zero_reference_groups) == 15 and not zero.editable
    assert float(zero.scale) == 1 and float(zero.snapshot.radius) == 0
    assert all(zero.materialize()[n] is p for n, p in parameters.items())
    bridge = LagOneHypergradient();zero.record_executed(bridge, action_step=0, actor=actor)
    result = bridge.consume(zero.project_next_gradients({}, global_token_count=1), gradient_step=1)
    assert result.gradients['x'] is None


def test_wrong_anchor_bank_or_embedding_orientation_is_rejected():
    parameters, snapshot = fixture(torch.float64)
    actor = FactorActor()
    def swapped_anchors(observation):
        raw = actor(observation)
        old = raw.matrices['embed.weight']
        raw.matrices['embed.weight'] = Factors(old.P, old.Q,
            Anchors(old.anchors.A.clone(), old.anchors.B.clone()))
        return raw
    with pytest.raises(ValueError, match='action-time anchors'):
        prepare_transformer_action(actor, snapshot, parameters, actor_forward=swapped_anchors)
    with pytest.raises(ValueError, match='orientation mismatch'):
        prepare_transformer_action(actor, snapshot, parameters, embedding_names=())
