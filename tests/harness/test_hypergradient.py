import pytest
import torch
from chisel.hypergradient import transfer_gradient, normalized_transfer, edit_schedule
from chisel.muonh import sphere


def test_centered_difference_of_actual_lookahead_loss():
    torch.manual_seed(27)
    raw = torch.randn(5, 3, dtype=torch.float64)
    consolidated = torch.randn_like(raw) * .1
    direction = torch.randn_like(raw)
    inputs = torch.randn(7, 3, dtype=raw.dtype)
    targets = torch.randn(7, 5, dtype=raw.dtype)
    def loss(transfer):
        weight = sphere(raw + transfer, direction, .12) + consolidated - transfer
        return (inputs @ weight.T - targets).square().mean()
    successor = (sphere(raw, direction, .12) + consolidated).requires_grad_()
    q, = torch.autograd.grad((inputs @ successor.T - targets).square().mean(), successor)
    h = transfer_gradient(raw, direction, .12, q)
    s = torch.zeros_like(raw, requires_grad=True)
    exact, = torch.autograd.grad(loss(s), s)
    torch.testing.assert_close(h, exact, atol=1e-13, rtol=1e-12)
    direction_s = torch.randn_like(raw)
    epsilon = 1e-5
    difference = (loss(epsilon * direction_s) - loss(-epsilon * direction_s)) / (2 * epsilon)
    torch.testing.assert_close(difference, (h * direction_s).sum(), rtol=1e-7, atol=1e-9)
    assert loss(-1e-4 * h) < loss(torch.zeros_like(raw))


def test_frozen_direction_and_cotangent():
    raw, update, q = [torch.randn(4, 3, requires_grad=True) for _ in range(3)]
    lr = torch.tensor(.1, requires_grad=True)
    h = transfer_gradient(raw, update, lr, q)
    assert not h.requires_grad
    assert all(x.grad is None for x in (raw, update, q, lr))


def test_single_global_scale_and_zero_gradient():
    raw = {'a': torch.ones(2, 3), 'b': torch.ones(3, 4)}
    proposal = {k: v + .2 for k, v in raw.items()}
    gradients = {'a': torch.ones(2, 3), 'b': torch.ones(3, 4) * 7}
    edits, hnorm, unorm = normalized_transfer(raw, proposal, gradients, .1, 1e-20)
    norm = sum(v.square().sum() for v in edits.values()).sqrt()
    torch.testing.assert_close(norm, .1 * unorm)
    torch.testing.assert_close(edits['a'][0, 0], edits['b'][0, 0] / 7)
    zeros = {k: torch.zeros_like(v) for k, v in raw.items()}
    edits, _, _ = normalized_transfer(raw, proposal, zeros, .1, 1e-20)
    assert all(torch.count_nonzero(v) == 0 for v in edits.values())
    with pytest.raises(FloatingPointError):
        normalized_transfer(raw, proposal, {k: v * float('nan') for k, v in raw.items()}, .1, 1e-20)


def test_schedule():
    args = dict(size=.1, start_step=2, ramp_steps=2, terminal_identity=True)
    assert [edit_schedule(t, 6, **args) for t in range(6)] == [0, 0, .05, .1, .1, 0]


def test_zero_lr_has_zero_gradient():
    raw = torch.randn(5, 3, dtype=torch.float64)
    h = transfer_gradient(raw, torch.randn_like(raw), 0., torch.randn_like(raw))
    torch.testing.assert_close(h, torch.zeros_like(h), atol=2e-15, rtol=0)
