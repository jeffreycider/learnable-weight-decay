"""Actual GPT CE validates the new lifecycle beyond weight-space toy losses.

The explicit reference runs an extra GPT evaluation in this unit test only.
The online branch consumes the single ordinary summed-token backward.
"""
import copy

import pytest
import torch
from torch.func import functional_call

from baseline.model import GPT
from baseline.fp32_execution import enable_fp32_execution
from rewriter.adaptive_online import AdaptiveOnlineLifecycle
from tests.test_adaptive_online import fixture, explicit


@pytest.mark.parametrize('radius', [1e-8, 100.])
def test_ordinary_gpt_ce_supplies_shared_and_private_direct_credit(radius):
    old, initial, snapshot, _, _ = fixture(radius)
    model = enable_fp32_execution(GPT(vocab_size=32, num_layers=1, model_dim=128).float())
    model.eval()
    live = dict(model.named_parameters())
    assert live.keys() == initial.keys()
    with torch.no_grad():
        for name, value in live.items():
            value.copy_(initial[name])
    inputs = torch.arange(8).reshape(2, 4)
    targets = (inputs + 1) % 32
    lifecycle = AdaptiveOnlineLifecycle(
        old.binding, old.shared_optimizer, old.private_optimizer,
        global_token_count=inputs.numel(), final_action_step=3,
        radius_fraction=radius, shared_clip=1e20, private_clip=1e20,
        parameter_view='unwrapped_stored_fp32')

    # Independent, explicit differentiation through the stored rewrite and GPT.
    reference = copy.deepcopy(lifecycle.binding)
    base = {name: value.detach().clone() for name, value in live.items()}
    rewritten = explicit(reference, snapshot, base, radius)
    reference_loss = functional_call(model, rewritten, (inputs, targets)) / inputs.numel()
    expected = torch.autograd.grad(reference_loss, tuple(reference.parameters()), allow_unused=True)

    lifecycle.commit(snapshot, live, action_step=0, base_step_completed=0)
    for name, value in live.items():
        torch.testing.assert_close(value, rewritten[name], rtol=0, atol=0)
    # Count only the online branch, excluding the explicit reference above.
    forwards = []
    hook = model.register_forward_hook(lambda *args: forwards.append(1))
    try:
        ordinary_loss_sum = model(inputs, targets)
        torch.testing.assert_close(ordinary_loss_sum / inputs.numel(), reference_loss, rtol=0, atol=0)
        ordinary_loss_sum.backward()
        gradients = {name: None if p.grad is None else p.grad.detach().clone() for name, p in live.items()}
        observed = lifecycle.consume(gradients, gradient_step=1)
    finally:
        hook.remove()

    assert len(forwards) == 1
    assert observed['gpt_forwards'] == observed['gpt_backwards'] == 0
    for name, p in live.items():
        if gradients[name] is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(p.grad, gradients[name], rtol=0, atol=0)
    errors = {role: [0., 0.] for role in ('shared', 'private')}
    for (name, actual), wanted in zip(lifecycle.binding.named_parameters(), expected):
        if wanted is None:
            assert actual.grad is None, name
        else:
            torch.testing.assert_close(actual.grad, wanted, rtol=3e-4, atol=3e-7, msg=name)
            role = name.split('.', 1)[0]
            errors[role][0] += float((actual.grad.double() - wanted.double()).square().sum())
            errors[role][1] += float(wanted.double().square().sum())
    # An absolute tolerance alone could hide incorrect tiny saturated-guard
    # gradients. Require relative agreement for each learned parameter family.
    for role, (squared_error, squared_reference) in errors.items():
        assert squared_reference > 0, role
        assert (squared_error / squared_reference)**.5 < 3e-4, (role, squared_error, squared_reference)
    for module in (lifecycle.binding.shared, lifecycle.binding.private):
        assert any(p.grad is not None and p.grad.norm() > 0 for p in module.parameters())

    # Consuming credit must not alter GPT weights or its base optimizer inputs.
    for name, value in live.items():
        torch.testing.assert_close(value, rewritten[name], rtol=0, atol=0)
