import json

import numpy as np
import pytest
import torch

from rewriter.algebra import Anchors
from rewriter.controller import ControllerSnapshot, MatrixObservation, VectorObservation
from rewriter.grouped_control import grouped_norm_only_action, solve_grouped_allocation


@pytest.mark.parametrize('radius', [0., .3, 2., 20.])
def test_allocation_matches_independent_constrained_optimizer(radius):
    scipy = pytest.importorskip('scipy.optimize')
    norms = np.array([.4, 2., 3., 1.])
    projected = np.array([.4, 1.1, 2.7, 0.])
    alpha, stats = solve_grouped_allocation(norms, projected, radius)
    alpha = alpha.numpy()
    assert np.linalg.norm(alpha * projected) <= radius + 1e-12
    assert alpha[3] == 0
    if radius == 0:
        assert not alpha.any()
    else:
        def loss(a):
            return np.sqrt(norms**2 - projected**2 + projected**2*(1-a)**2).dot(1/norms)/len(norms)
        result = scipy.minimize(loss, alpha * .9, method='SLSQP', bounds=[(0,1)]*4,
            constraints=[dict(type='ineq', fun=lambda a: radius**2 - ((a*projected)**2).sum())],
            options=dict(ftol=1e-13, maxiter=1000))
        assert result.success, result.message
        assert loss(alpha) == pytest.approx(result.fun, abs=1e-8)
    json.dumps(stats, allow_nan=False)


def test_active_kkt_full_access_boundary_and_no_access():
    norms = np.array([.1, 2., 3.])
    projected = np.array([.1, 1., 0.])
    alpha, stats = solve_grouped_allocation(norms, projected, .6)
    a = alpha.numpy()
    assert a[0] == 1  # Cheap full-access group reaches its norm's kink.
    assert 0 < a[1] < 1
    assert a[2] == 0
    assert stats['ideal_edit_norm'] == pytest.approx(.6, abs=1e-12)
    remaining = np.sqrt(norms[1]**2-projected[1]**2+projected[1]**2*(1-a[1])**2)
    assert (1-a[1])/(len(norms)*norms[1]*remaining) == pytest.approx(2*stats['dual_lambda']*a[1], abs=1e-12)
    assert 2*stats['dual_lambda']*norms[0]**2 <= 1/len(norms)
    alpha, zero = solve_grouped_allocation(norms, [0., 0., 0.], .6)
    assert not alpha.any()
    assert zero['ideal_mean_relative_l2_change'] == 0
    assert zero['dual_lambda'] == 0


@pytest.mark.parametrize('relative_gap', [0., 1e-15, 1e-10, 1e-5])
def test_nearly_full_projection_and_units(relative_gap):
    norms = torch.tensor([1., 4.], dtype=torch.float64)
    projected = norms * (1-relative_gap)
    alpha, stats = solve_grouped_allocation(norms, projected, 2.)
    assert torch.isfinite(alpha).all()
    assert stats['ideal_edit_norm'] == pytest.approx(2., abs=1e-12)
    for scale in (1e-50, 1e50):
        other, _ = solve_grouped_allocation(norms*scale, projected*scale, 2.*scale)
        # Near p=N the tiny inaccessible residual is sensitive to rescaling's
        # final-bit rounding; the allocation remains accurate to ~1e-11.
        torch.testing.assert_close(other, alpha, atol=1e-11, rtol=1e-11)


def test_reference_zero_projection_excess_and_invalid_radius_fail_explicitly():
    with pytest.raises(ValueError, match='positive finite'):
        solve_grouped_allocation([0.], [0.], 1.)
    with pytest.raises(ValueError, match='exceeds'):
        solve_grouped_allocation([1.], [1.1], 1.)
    for radius in (-1., float('nan'), float('inf')):
        with pytest.raises(ValueError, match='radius'):
            solve_grouped_allocation([1.], [1.], radius)
    _, stats = solve_grouped_allocation([1.], torch.tensor([1+1e-7], dtype=torch.float64), 1.)
    assert stats['projection_norm_clipped']
    assert stats['projection_norm_relative_excess'] == pytest.approx(1e-7)


def make_snapshot(parameters, radius):
    generator = torch.Generator().manual_seed(145)
    matrices, vectors = {}, {}
    for name, value in parameters.items():
        if value.ndim == 2:
            oriented = value.T if name == 'embed.weight' else value
            anchors = Anchors.random(*oriented.shape, 1, generator=generator, dtype=torch.float64)
            matrices[name] = MatrixObservation(anchors, torch.empty(0), torch.empty(0),
                                               (oriented @ anchors.B, oriented.T @ anchors.A))
        else:
            vectors[name] = VectorObservation(torch.empty(0), value)
    return ControllerSnapshot(matrices, vectors, torch.empty(0),
        sum(value.square().sum() for value in parameters.values()).sqrt(),
        torch.tensor(radius, dtype=torch.float64))


@pytest.mark.parametrize('radius', [0., .4, 20.])
def test_thin_action_matches_dense_groups_and_keeps_excluded_coordinates_zero(radius):
    generator = torch.Generator().manual_seed(155)
    parameters = {name: torch.randn(shape, generator=generator, dtype=torch.float64) for name, shape in
                  [('embed.weight', (5,3)), ('first.weight', (4,3)), ('second.weight', (4,3)),
                   ('first.bias', (4,)), ('second.bias', (4,)), ('norm.gains', (4,)), ('scale', ())]}
    groups = {'embedding': ('embed.weight',), 'weight': ('first.weight','second.weight'),
              'bias': ('first.bias','second.bias')}
    snapshot = make_snapshot(parameters, radius)
    action, stats = grouped_norm_only_action(snapshot, parameters, groups)
    assert action.norm_sq().sqrt() <= radius + 1e-12
    relative = []
    for index, names in enumerate(groups.values()):
        new_sq, old_sq = 0., 0.
        for name in names:
            value = parameters[name].T if name == 'embed.weight' else parameters[name]
            if name in action.matrices:
                anchors = snapshot.matrices[name].anchors
                a, b = anchors.A, anchors.B
                projection = value@b@b.T + a@a.T@value - a@a.T@value@b@b.T
                delta = action.matrices[name].dense()
                torch.testing.assert_close(delta, -stats['coefficients'][index]*projection)
            else:
                delta = action.vectors[name]
            new_sq += (value+delta).square().sum()
            old_sq += value.square().sum()
        relative.append(float(new_sq.sqrt()/old_sq.sqrt()-1))
    assert np.mean(relative) == pytest.approx(stats['ideal_mean_relative_l2_change'], abs=1e-12)
    assert torch.count_nonzero(action.vectors['norm.gains']) == 0
    assert torch.count_nonzero(action.vectors['scale']) == 0
    json.dumps(stats, allow_nan=False)


def test_grouped_allocation_beats_total_squared_norm_global_coefficient():
    # Full-access groups with very unequal norms expose the old objective's
    # uniform shrinkage: equal relative L2 contributions favor the small group.
    norms = np.array([.1, 10.])
    radius = .2
    alpha, stats = solve_grouped_allocation(norms, norms, radius)
    old_coefficient = radius/np.linalg.norm(norms)
    assert alpha[0] == 1
    assert stats['ideal_mean_relative_l2_change'] < -old_coefficient - .4


@pytest.mark.parametrize('projected', [[.4, 1.1], [.8, 2.]])
def test_two_group_dense_grid_agrees_without_scipy(projected):
    norms = np.array([.8, 2.])
    projected = np.array(projected)
    radius = .7
    _, stats = solve_grouped_allocation(norms, projected, radius)
    first = np.linspace(0., min(1., radius/projected[0]), 200001)
    second = np.minimum(1., np.sqrt(np.maximum(0., radius**2-(first*projected[0])**2))/projected[1])
    candidates = np.stack([first, second], axis=1)
    losses = (np.sqrt(norms**2-projected**2+projected**2*(1-candidates)**2)/norms-1).mean(axis=1)
    assert stats['ideal_mean_relative_l2_change'] <= float(losses.min()) + 1e-14
    assert stats['ideal_mean_relative_l2_change'] == pytest.approx(float(losses.min()), abs=2e-10)
