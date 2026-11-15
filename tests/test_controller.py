import dataclasses
import torch
import pytest
from rewriter.algebra import Anchors
from rewriter.controller import (GlobalRewriter, ControllerSnapshot, MatrixObservation,
                                VectorObservation, coordinate_features, vector_features)


def make_snapshot(rank=2):
    gen = torch.Generator().manual_seed(123)
    matrices = {}
    for name, (n, m) in {'first': (7, 5), 'second': (6, 9)}.items():
        a = Anchors.random(n, m, rank, generator=gen)
        banks = [torch.randn(n, m, generator=gen) for _ in range(4)]
        rows = coordinate_features([x @ a.B for x in banks], a.A, torch.ones(n), [0.] * 10)
        cols = coordinate_features([x.T @ a.A for x in banks], a.B, torch.ones(m), [0.] * 10)
        matrices[name] = MatrixObservation(a, rows, cols, (banks[0] @ a.B, banks[0].T @ a.A))
    vbanks = [torch.randn(4, generator=gen) for _ in range(4)]
    vectors = {'gain': VectorObservation(vector_features(vbanks, [0.] * 10), vbanks[0].clone())}
    return ControllerSnapshot(matrices, vectors, torch.randn(3, 12, generator=gen), torch.tensor(10.), torch.tensor(.1))


def small_controller():
    torch.manual_seed(456)
    return GlobalRewriter(rank=2, trajectory_dim=12, coordinate_width=8, global_width=16,
                          pool_queries=2, latents=4, blocks=1, heads=2, ff_width=32)


def test_zero_heads_exact_identity_and_nonzero_action_derivative():
    model = small_controller()
    s = make_snapshot()
    action = model(s)
    assert action.norm_sq() == 0
    loss = action.contraction(s.weights, s.vector_weights)
    loss.backward()
    assert model.row_decoder[-1].weight.grad.abs().sum() > 0
    assert model.column_decoder[-1].weight.grad.abs().sum() > 0
    assert model.vector_decoder[-1].weight.grad.abs().sum() > 0


def test_global_information_affects_distant_matrix_and_vector():
    model = small_controller()
    with torch.no_grad():
        for decoder in [model.row_decoder, model.column_decoder, model.vector_decoder]:
            decoder[-1].weight.normal_(std=.2)
    snapshot = make_snapshot()
    first = model(snapshot)
    changed = dict(snapshot.matrices)
    obs = changed['first']
    changed['first'] = dataclasses.replace(obs, rows=obs.rows + 10., columns=obs.columns + 10.)
    second = model(dataclasses.replace(snapshot, matrices=changed))
    assert (first.matrices['second'].Q - second.matrices['second'].Q).abs().max() > 1e-6
    assert (first.vectors['gain'] - second.vectors['gain']).abs().max() > 1e-6
    # Own local coordinates of the affected layer remain exactly unchanged.
    assert changed['second'] is snapshot.matrices['second']


def test_trajectory_bank_affects_actions_and_no_recurrent_state():
    model = small_controller()
    with torch.no_grad():
        model.column_decoder[-1].weight.normal_()
    snapshot = make_snapshot()
    a = model(snapshot)
    b = model(dataclasses.replace(snapshot, trajectories=snapshot.trajectories + 5.))
    c = model(snapshot)
    assert (a.matrices['second'].Q - b.matrices['second'].Q).abs().max() > 1e-6
    torch.testing.assert_close(a.matrices['second'].Q, c.matrices['second'].Q, atol=0, rtol=0)
    assert all(getattr(module, 'dropout', 0) == 0 for module in model.modules()
               if isinstance(module, torch.nn.MultiheadAttention))


def test_feature_helpers_detach_clone_and_retain_magnitudes():
    banks = [torch.randn(5, 2, requires_grad=True) for _ in range(4)]
    anchor = torch.randn(5, 2)
    rms = torch.ones(5)
    metadata = torch.zeros(10)
    result = coordinate_features(banks, anchor, rms, metadata)
    saved = result.clone()
    assert result.shape == (5, 25)
    assert not result.requires_grad
    with torch.no_grad():
        banks[0].zero_(); anchor.zero_(); rms.zero_(); metadata.fill_(9.)
    torch.testing.assert_close(result, saved)
    vbanks = [torch.randn(5, requires_grad=True) for _ in range(4)]
    vf = vector_features(vbanks, torch.zeros(10))
    saved = vf.clone()
    with torch.no_grad():
        vbanks[0].fill_(100.)
    torch.testing.assert_close(vf, saved)
    assert vf.shape == (5, 18) and not vf.requires_grad
    scaled = vector_features([x.detach() * 2 for x in vbanks], [0.] * 10)
    original = vector_features(vbanks, [0.] * 10)
    torch.testing.assert_close(scaled[:, ::2][:, :4], original[:, ::2][:, :4])
    torch.testing.assert_close(scaled[:, 1:8:2] - original[:, 1:8:2], torch.full((5, 4), 2.).log())


def test_feature_zero_banks_finite_and_metadata_validated():
    a = torch.zeros(5, 2)
    features = coordinate_features([a] * 4, a, torch.zeros(5), [0.] * 10)
    assert torch.isfinite(features).all()
    with pytest.raises(ValueError, match='metadata'):
        coordinate_features([a] * 4, a, torch.zeros(5), [0.] * 9)


def test_default_controller_parameter_count_is_shared_not_dense():
    model = GlobalRewriter()
    count = sum(p.numel() for p in model.parameters())
    assert 500_000 < count < 2_000_000
