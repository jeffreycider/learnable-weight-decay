import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from rewriter.algebra import Anchors, Factors, RawAction, Measurements, guard, auxiliary_objective


def anchors(n=7, m=5, rank=2, seed=11, dtype=torch.float64):
    return Anchors.random(n, m, rank, generator=torch.Generator().manual_seed(seed), dtype=dtype)


def factors(n=7, m=5, rank=2, seed=11, dtype=torch.float64):
    a = anchors(n, m, rank, seed, dtype)
    g = torch.Generator().manual_seed(seed + 1)
    return Factors.canonical(torch.randn(n, rank, generator=g, dtype=dtype),
                             torch.randn(m, rank, generator=g, dtype=dtype), a)


def project(x, a):
    return x @ a.B, x.transpose(-1, -2) @ a.A


@pytest.mark.parametrize('shape', [(7, 5), (5, 11), (3, 17)])
def test_canonical_norm_and_contractions(shape):
    f = factors(*shape)
    w = torch.randn(*shape, dtype=torch.float64)
    g = torch.randn(4, *shape, dtype=torch.float64)
    dense = f.dense()
    torch.testing.assert_close(f.anchors.A.T @ f.P, torch.zeros(2, 2, dtype=w.dtype), atol=1e-14, rtol=0)
    torch.testing.assert_close(f.anchors.A.T @ f.anchors.A, torch.eye(2, dtype=w.dtype))
    torch.testing.assert_close(f.anchors.B.T @ f.anchors.B, torch.eye(2, dtype=w.dtype))
    torch.testing.assert_close(f.norm_sq(), dense.square().sum())
    torch.testing.assert_close(f.contract(*project(g, f.anchors)), (g * dense).sum((-2, -1)))
    action = RawAction({'w': f}, {})
    torch.testing.assert_close(action.norm_change({'w': project(w, f.anchors)}, {}),
                               (w + dense).square().sum() - w.square().sum())
    assert torch.linalg.matrix_rank(dense) <= 4


def test_embedding_uses_computational_transpose_and_vectors_count_once():
    stored = torch.randn(17, 5, dtype=torch.float64)
    w = stored.T
    f = factors(5, 17)
    v = torch.randn(4, dtype=torch.float64)
    dv = torch.randn(4, dtype=torch.float64)
    action = RawAction({'embedding': f}, {'gain': dv})
    cached = action.norm_change({'embedding': project(w, f.anchors)}, {'gain': v})
    dense = ((stored + f.dense().T).square().sum() - stored.square().sum()
             + (v + dv).square().sum() - v.square().sum())
    torch.testing.assert_close(cached, dense)
    grad = torch.randn_like(stored)
    vg = torch.randn_like(v)
    torch.testing.assert_close(action.contraction({'embedding': project(grad.T, f.anchors)}, {'gain': vg}),
                               (grad * f.dense().T).sum() + (vg * dv).sum())


@pytest.mark.parametrize('dtype,tol', [(torch.float32, 1e-5), (torch.bfloat16, 0.04)])
def test_zero_weights_and_mixed_precision_storage(dtype, tol):
    a = anchors(dtype=dtype)
    f = Factors.canonical(torch.randn(7, 2).to(dtype), torch.randn(5, 2).to(dtype), a)
    # Low-precision anchors only satisfy orthogonality to storage precision.
    assert torch.allclose(f.norm_sq().float(), f.dense().float().square().sum(), rtol=tol, atol=tol)
    zero = Factors.canonical(torch.zeros_like(a.A), torch.zeros_like(a.B), a)
    assert zero.norm_sq() == 0
    assert torch.count_nonzero(zero.dense()) == 0


def test_reject_free_product_and_bad_shape():
    a = anchors()
    with pytest.raises(ValueError, match='free PQ'):
        Factors(a.A, a.B, a, family='free_pq')
    with pytest.raises(ValueError, match='shape'):
        Factors(a.A[:1], a.B, a)
    with pytest.raises(ValueError, match='rank'):
        anchors(2, 3, 4)


def test_thin_projection_never_allocates_coordinate_square():
    class NoSquare(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            def check(x):
                if isinstance(x, torch.Tensor):
                    assert tuple(x.shape) not in {(1009, 1009), (101, 101)}
                elif isinstance(x, (tuple, list)):
                    for y in x:
                        check(y)
            check(out)
            return out
    a = anchors(1009, 101)
    with NoSquare():
        f = Factors.canonical(torch.randn_like(a.A), torch.randn_like(a.B), a)
        f.norm_sq()


def test_global_sum_before_square_and_shared_guard():
    action = RawAction({}, {'a': torch.tensor([1.]), 'b': torch.tensor([-1.])})
    measurements = Measurements({}, {'a': torch.ones(4, 1), 'b': torch.ones(4, 1)})
    assert measurements.penalty(action) == 0
    assert sum((measurements.vectors[n] * v).sum(-1).square().mean()
               for n, v in action.vectors.items()) == 2
    k = guard(action, measurements, torch.tensor(.5), .001)
    assert k.ndim == 0
    assert (k * action.vectors['a'] + k * action.vectors['b']).item() == 0
    assert k * action.norm_sq().sqrt() <= .5


def test_train_nullspace_is_visible_to_independent_audit():
    action = RawAction({}, {'v': torch.tensor([1., -1.])})
    train = Measurements({}, {'v': torch.tensor([[1., 1.]]).repeat(4, 1)})
    audit = Measurements({}, {'v': torch.tensor([[1., -1.]]).repeat(4, 1)})
    assert train.penalty(action) == 0
    assert audit.penalty(action) == 4
    k = guard(action, audit, torch.tensor(10.), .01)
    torch.testing.assert_close(k, torch.tensor(.005))


def test_fixed_anchor_invisible_subspace_and_refresh():
    a = anchors(7, 5)
    x = torch.randn(7, 5, dtype=torch.float64)
    x = x - a.A @ (a.A.T @ x)
    x = x - (x @ a.B) @ a.B.T
    torch.testing.assert_close(x @ a.B, torch.zeros(7, 2, dtype=x.dtype), atol=1e-14, rtol=0)
    torch.testing.assert_close(x.T @ a.A, torch.zeros(5, 2, dtype=x.dtype), atol=1e-14, rtol=0)
    f = factors(7, 5)
    assert abs((x * f.dense()).sum()) < 1e-13
    refreshed = anchors(7, 5, seed=92)
    assert torch.linalg.vector_norm(x @ refreshed.B) > .1


@pytest.mark.parametrize('regime', ['inactive', 'norm', 'audit'])
def test_delayed_vjp_matches_full_tiny_network_with_guard_and_finite_difference(regime):
    torch.manual_seed(42)
    a = anchors()
    p = torch.randn(7, 2, dtype=torch.float64, requires_grad=True)
    q = torch.randn(5, 2, dtype=torch.float64, requires_grad=True)
    v = torch.randn(7, dtype=torch.float64, requires_grad=True)
    w = torch.randn(7, 5, dtype=torch.float64)
    bias = torch.randn(7, dtype=torch.float64)
    inputs = torch.randn(3, 5, dtype=torch.float64)
    target = torch.randn(3, 7, dtype=torch.float64)
    sg = torch.randn(4, 7, 5, dtype=torch.float64)
    sv = torch.randn(4, 7, dtype=torch.float64)
    audit = Measurements({'w': project(sg, a)}, {'v': sv})
    radius, budget = {'inactive': (1000., 1000.), 'norm': (.1, 1000.), 'audit': (1000., .1)}[regime]
    def evaluate(pp, qq, vv):
        act = RawAction({'w': Factors.canonical(pp, qq, a)}, {'v': vv})
        k = guard(act, audit, torch.tensor(radius, dtype=w.dtype), budget)
        ww, bb = w + k * act.matrices['w'].dense(), bias + k * vv
        loss = ((inputs @ ww.T + bb).tanh() - target).square().mean()
        return loss, ww, bb, act, k
    loss, ww, bb, act, k = evaluate(p, q, v)
    direct = torch.autograd.grad(loss, (p, q, v), retain_graph=True)
    gw, gb = torch.autograd.grad(loss, (ww, bb), retain_graph=True)
    scalar = k * act.contraction({'w': project(gw.detach(), a)}, {'v': gb.detach()})
    thin = torch.autograd.grad(scalar, (p, q, v))
    for expected, actual in zip(direct, thin):
        torch.testing.assert_close(expected, actual, atol=2e-13, rtol=1e-11)
    direction = torch.randn_like(p)
    eps = 1e-5
    fd = (evaluate(p.detach() + eps * direction, q.detach(), v.detach())[0]
          - evaluate(p.detach() - eps * direction, q.detach(), v.detach())[0]) / (2 * eps)
    torch.testing.assert_close(fd, (direct[0] * direction).sum(), atol=2e-9, rtol=1e-5)
    if regime != 'inactive':
        assert k < 1
        # Omitting d(kappa)/d(phi) is measurably different in active regimes.
        _, _, _, act2, k2 = evaluate(p, q, v)
        conditional = torch.autograd.grad(k2.detach() * act2.contraction({'w': project(gw.detach(), a)}, {'v': gb.detach()}), (p, q, v))
        assert sum((x-y).abs().sum() for x,y in zip(direct, conditional)) > 1e-6


def test_zero_action_guard_backward_is_finite():
    p = torch.zeros(7, 2, dtype=torch.float64, requires_grad=True)
    q = torch.zeros(5, 2, dtype=torch.float64, requires_grad=True)
    a = anchors()
    act = RawAction({'w': Factors.canonical(p, q, a)}, {})
    audit = Measurements({'w': (torch.ones(4, 7, 2, dtype=p.dtype), torch.ones(4, 5, 2, dtype=p.dtype))}, {})
    k = guard(act, audit, torch.tensor(1., dtype=p.dtype), .001)
    (k * act.contraction({'w': (torch.ones_like(p), torch.ones_like(q))}, {})).backward()
    assert k == 1
    assert torch.isfinite(p.grad).all() and torch.isfinite(q.grad).all()
    assert q.grad.abs().sum() > 0


def test_auxiliary_objective_matches_dense_reference_including_vector_gradient():
    a = anchors()
    p = torch.randn(7, 2, dtype=torch.float64, requires_grad=True)
    q = torch.randn(5, 2, dtype=torch.float64, requires_grad=True)
    dv = torch.randn(3, dtype=torch.float64, requires_grad=True)
    action = RawAction({'w': Factors.canonical(p, q, a)}, {'v': dv})
    w, v = torch.randn(7, 5, dtype=p.dtype), torch.randn(3, dtype=p.dtype)
    grad = torch.randn(4, 7, 5, dtype=p.dtype)
    vg = torch.randn(4, 3, dtype=p.dtype)
    train = Measurements({'w': project(grad, a)}, {'v': vg})
    weight_norm = (w.square().sum() + v.square().sum()).sqrt()
    radius = torch.tensor(.2, dtype=p.dtype)
    cached = auxiliary_objective(action, {'w': project(w, a)}, {'v': v}, train,
                                 weight_norm=weight_norm, radius=radius, loss_budget=.3, beta=2., gamma=.04)
    d = action.matrices['w'].dense()
    norm_diff = (w+d).square().sum() - w.square().sum() + (v+dv).square().sum() - v.square().sum()
    z = (grad * d).sum((-2, -1)) + (vg * dv).sum(-1)
    dense = (norm_diff / (2 * weight_norm * radius + radius.square() + 1e-24)
             + 2 * z.square().mean() / .3**2
             + .04 * (d.square().sum() + dv.square().sum()) / (radius.square() + 1e-24))
    torch.testing.assert_close(cached, dense)
    expected = torch.autograd.grad(dense, (p, q, dv), retain_graph=True)
    actual = torch.autograd.grad(cached, (p, q, dv))
    for x, y in zip(expected, actual):
        torch.testing.assert_close(x, y)


def test_action_cotangent_through_post_commit_ema_then_lookahead():
    a = anchors()
    p = torch.randn(7, 2, dtype=torch.float64, requires_grad=True)
    q = torch.randn(5, 2, dtype=torch.float64, requires_grad=True)
    action = RawAction({'w': Factors.canonical(p, q, a)}, {})
    w, old_ema, inputs = torch.randn(7, 5, dtype=p.dtype), torch.randn(7, 5, dtype=p.dtype), torch.randn(3, 5, dtype=p.dtype)
    beta, gamma = .3, .95
    committed = w + action.matrices['w'].dense()
    new_ema = gamma * old_ema + (1-gamma) * committed
    actual_forward = (1+beta) * committed - beta * new_ema
    loss = (inputs @ actual_forward.T).sin().square().mean()
    direct = torch.autograd.grad(loss, (p, q), retain_graph=True)
    g = torch.autograd.grad(loss, actual_forward)[0].detach()
    scalar = action.contraction({'w': project((1+beta*gamma) * g, a)}, {})
    thin = torch.autograd.grad(scalar, (p, q))
    for x, y in zip(direct, thin):
        torch.testing.assert_close(x, y)


def test_anchor_stream_does_not_consume_baseline_rng():
    torch.manual_seed(16)
    before = torch.random.get_rng_state().clone()
    a = anchors()
    assert torch.equal(before, torch.random.get_rng_state())
    b = anchors()
    torch.testing.assert_close(a.A, b.A, atol=0, rtol=0)
    torch.testing.assert_close(a.B, b.B, atol=0, rtol=0)
