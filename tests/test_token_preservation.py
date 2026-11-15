import pytest
import torch

from rewriter.token_preservation import token_preservation_terms


OPTIONS = dict(multiplier=2., mean_tolerance=1e-8, tail_tolerance=1e-8,
               scale=1e-6, rho=.1)


def test_same_mean_with_large_tail_gets_stronger_penalty():
    uniform = torch.tensor([2e-8, 2e-8], dtype=torch.float64, requires_grad=True)
    tail = torch.tensor([1e-9, 3.9e-8], dtype=torch.float64, requires_grad=True)
    u = token_preservation_terms(uniform, **OPTIONS)
    t = token_preservation_terms(tail, **OPTIONS)
    torch.testing.assert_close(u['dual_linear'].mean(), t['dual_linear'].mean())
    assert t['total'].mean() > u['total'].mean()
    t['total'].sum().backward()
    assert tail.grad[1] > tail.grad[0]


def test_uneven_microbatch_accumulation_matches_joint_loss_and_gradient():
    p = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
    inputs = torch.arange(1, 8, dtype=torch.float64) * 1e-4

    def losses(x):
        kl = (p * x).square()
        rank = (p * x - 0.0003).relu().square()
        return token_preservation_terms(kl, ranking_loss=rank, ranking_weight=3., **OPTIONS)['total']

    whole = losses(inputs).mean()
    expected = torch.autograd.grad(whole, p)[0]
    total = torch.zeros_like(whole)
    for x in inputs.split(3):
        loss = losses(x).sum() / len(inputs)
        total += loss.detach()
        loss.backward()
    torch.testing.assert_close(total, whole)
    torch.testing.assert_close(p.grad, expected)


def test_scale_reparameterization_leaves_value_and_gradient_unchanged():
    kl = torch.tensor([1e-9, 3e-8], dtype=torch.float64, requires_grad=True)
    a = token_preservation_terms(kl, **OPTIONS)['total']
    changed = dict(OPTIONS, scale=OPTIONS['scale'] * 100,
                   multiplier=OPTIONS['multiplier'] * 100, rho=OPTIONS['rho'] * 100**2)
    b = token_preservation_terms(kl, **changed)['total']
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(torch.autograd.grad(a.sum(), kl)[0],
                               torch.autograd.grad(b.sum(), kl)[0])


def test_lower_tolerance_does_not_multiply_curvature():
    def curvature(tolerance):
        kl = torch.tensor([3e-8], dtype=torch.float64, requires_grad=True)
        opts = dict(OPTIONS, mean_tolerance=tolerance, tail_tolerance=tolerance)
        loss = token_preservation_terms(kl, **opts)['total'].sum()
        grad = torch.autograd.grad(loss, kl, create_graph=True)[0]
        return torch.autograd.grad(grad.sum(), kl)[0]
    torch.testing.assert_close(curvature(1e-8), curvature(1e-10))


def test_negative_numerical_kl_clamps_without_gradient():
    kl = torch.tensor([-1e-16], dtype=torch.float64, requires_grad=True)
    token_preservation_terms(kl, **OPTIONS)['total'].sum().backward()
    assert kl.grad.item() == 0


def test_invalid_configuration():
    kl = torch.ones(3)
    with pytest.raises(ValueError, match='ranking measurements'):
        token_preservation_terms(kl, ranking_weight=1, **OPTIONS)
    with pytest.raises(ValueError, match='positive scale'):
        token_preservation_terms(kl, **dict(OPTIONS, scale=0))
    with pytest.raises(ValueError, match='per KL token'):
        token_preservation_terms(kl, ranking_loss=torch.ones(2), **OPTIONS)
