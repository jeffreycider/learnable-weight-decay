"""CPU oracle for the proposed online replay gradient reduction.

This tests reduction math with a small factorized softmax model, not GPT,
distributed transport, an adopted online protocol, or GPU performance.
"""
import copy

import pytest
import torch

from rewriter.primal_dual import PrimalDual


def logits(action, inputs):
    edit = action[:2, None] @ action[None, 2:]
    return inputs @ (torch.eye(2, dtype=inputs.dtype) + edit)


def raw_kl(action, inputs):
    teacher = inputs.log_softmax(-1).detach()
    return (teacher.exp() * (teacher - logits(action, inputs).log_softmax(-1))).sum(-1).mean()


def norm_term(action):
    # Fifteen equal relative-L2 groups, evaluated once on the actor rank.
    bases = torch.arange(1, 61, dtype=action.dtype).reshape(15, 4) / 30
    return ((bases + action).norm(dim=1) / bases.norm(dim=1) - 1).mean()


@pytest.mark.parametrize('sizes', [(2,) * 8, (1, 3, 2, 4)])
def test_global_coefficient_and_one_norm_match_central_saved_adam(sizes):
    torch.manual_seed(71)
    actor = torch.nn.Linear(3, 4, bias=False).double()
    observation = torch.tensor([.2, -.4, .7], dtype=torch.float64)
    inputs = torch.randn(sum(sizes), 2, dtype=torch.float64)
    # Nonempty moments matter: a correct raw gradient must also yield the same
    # shared Adam state and step, rather than merely a matching SGD direction.
    optimizer = torch.optim.Adam(actor.parameters(), lr=2e-4)
    actor(observation).square().sum().backward()
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    reference = copy.deepcopy(actor)
    refopt = torch.optim.Adam(reference.parameters(), lr=2e-4)
    refopt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    action = actor(observation)
    local_inputs = inputs.split(sizes)
    local_values = [float(raw_kl(action.detach(), part)) for part in local_inputs]
    # Test data deliberately straddle the hinge, exposing rank-local penalties.
    tolerance = (min(local_values) + max(local_values)) / 2
    dual = PrimalDual([0], tolerance=tolerance, rho=.3, dual_lr=.1)
    dual.multipliers[0] = .2
    before_dual = copy.deepcopy(dual.state_dict())

    ref_action = reference(observation)
    mean_kl = raw_kl(ref_action, inputs)
    violation = mean_kl.clamp_min(0) / tolerance - 1
    loss = norm_term(ref_action) + .2 * violation + .15 * violation.clamp_min(0).square()
    loss.backward()

    mean = 0.
    cotangent = torch.zeros_like(action)
    incorrect_local_cotangent = torch.zeros_like(action)
    for count, part in zip(sizes, local_inputs):
        # Each rank receives detached factors and differentiates GPT/softmax
        # only. No actor replica or actor optimizer is needed on those ranks.
        local_action = action.detach().clone().requires_grad_()
        kl = raw_kl(local_action, part)
        weight = count / len(inputs)
        gradient, = torch.autograd.grad(kl, local_action)
        mean += weight * float(kl.detach())
        cotangent += weight * gradient
        incorrect_local_cotangent += weight * dual.terms(float(kl.detach()), 0)['kl_coefficient'] * gradient
    coefficient = dual.terms(mean, 0)['kl_coefficient']
    normal_cotangent, = torch.autograd.grad(norm_term(action), action, retain_graph=True)
    assert not torch.allclose(incorrect_local_cotangent, coefficient * cotangent, rtol=1e-4, atol=1e-8)
    assert not torch.allclose(coefficient * cotangent + len(sizes) * normal_cotangent,
                              coefficient * cotangent + normal_cotangent, rtol=1e-4, atol=1e-8)
    # One VJP of the root actor, with norm gradient once and globally weighted
    # raw-KL cotangent once; no additional world-size average after this SUM.
    action.backward(coefficient * cotangent + normal_cotangent)
    for p, q in zip(actor.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=2e-13, atol=2e-13)
    assert dual.state_dict() == before_dual  # gradient evaluation is not dual ascent
    for model, opt in ((actor, optimizer), (reference, refopt)):
        torch.nn.utils.clip_grad_norm_(model.parameters(), .1)
        opt.step()
    for p, q in zip(actor.parameters(), reference.parameters()):
        torch.testing.assert_close(p, q, rtol=2e-13, atol=2e-13)
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(optimizer.state[p][key], refopt.state[q][key], rtol=2e-13, atol=2e-13)
