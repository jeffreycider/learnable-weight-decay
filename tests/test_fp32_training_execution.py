import copy

import pytest
import torch
import torch.nn.functional as F

from baseline.fp32_execution import enable_fp32_execution
from baseline.model import GPT
from rewriter.function_preservation import FunctionalPreservation


def test_training_loss_and_all_parameter_gradients_match_preservation_engine():
    torch.manual_seed(443)
    model = GPT(vocab_size=32, num_layers=2, model_dim=128).float()
    engine = FunctionalPreservation(copy.deepcopy(model), gpt_precision='fp32')
    parameters = {n: p.detach().clone().requires_grad_() for n, p in model.named_parameters()}
    inputs, targets = torch.randint(32, (2, 5)), torch.randint(32, (2, 5))
    expected = F.cross_entropy(engine._forward(parameters, inputs).reshape(-1, 32),
                               targets.flatten(), reduction='sum')
    expected_gradients = torch.autograd.grad(expected, tuple(parameters.values()))
    rng = torch.get_rng_state().clone()
    enable_fp32_execution(model)
    assert torch.equal(rng, torch.get_rng_state())
    actual = model(inputs, targets)
    actual_gradients = torch.autograd.grad(actual, tuple(model.parameters()))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for observed, reference in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(observed, reference, rtol=0, atol=0)
    assert all(torch.isfinite(g).all() for g in actual_gradients)


def test_execution_requires_explicit_promotion_and_rejects_double_install():
    model = GPT(vocab_size=32, num_layers=1, model_dim=128)
    before = model.embed.weight.detach().clone()
    with pytest.raises(ValueError, match='promote'):
        enable_fp32_execution(model)
    assert model.embed.weight.dtype == torch.bfloat16
    torch.testing.assert_close(model.embed.weight, before, rtol=0, atol=0)
    model.float()
    enable_fp32_execution(model)
    with pytest.raises(ValueError, match='already installed'):
        enable_fp32_execution(model)


def test_token_sum_units_match_disconnected_sequence_accumulation():
    torch.manual_seed(444)
    model = enable_fp32_execution(GPT(vocab_size=32, num_layers=1, model_dim=128).float())
    inputs, targets = torch.randint(32, (3, 4)), torch.randint(32, (3, 4))
    together = model(inputs, targets)
    separate = sum(model(x[None], y[None]) for x, y in zip(inputs, targets))
    torch.testing.assert_close(together, separate, rtol=2e-7, atol=2e-5)


def test_fp32_config_is_explicit_and_does_not_relabel_old_capture():
    from baseline.train import validate_config
    config = dict(max_steps=2, batch_tokens=8, seq_len=4, unscored=True,
                  compile=False, gpt_precision='fp32')
    validate_config(config)
    with pytest.raises(ValueError, match='gpt_precision'):
        validate_config(dict(config, gpt_precision='unknown'))
    with pytest.raises(ValueError, match='prefix capture'):
        validate_config(dict(config, state_bank={'steps': 2}))
