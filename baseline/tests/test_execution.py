import copy
import pytest
import torch
from baseline.execution import enable_compiled_loss, softcap_cross_entropy
from baseline.model import GPT, initialize_parameters_like_record
from baseline.train import validate_config


def test_compiled_loss_values_and_gradients():
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(31)
    source = torch.randn(2, 8, 67, generator=generator)*25  # includes softcap regime
    target = torch.randint(67, (2,8), generator=generator)
    expected_logits = source.clone().requires_grad_()
    actual_logits = source.clone().requires_grad_()
    expected = softcap_cross_entropy(expected_logits, target)
    compiled = torch.compile(softcap_cross_entropy, fullgraph=True, dynamic=False)
    actual = compiled(actual_logits, target)
    expected.backward(); actual.backward()
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-5)
    torch.testing.assert_close(actual_logits.grad, expected_logits.grad, rtol=2e-5, atol=2e-6)


def test_loss_only_execution_preserves_model_and_hooks():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    reference = GPT(64, 1, 128)
    initialize_parameters_like_record(reference)
    # Nonzero output projection lets this test exercise hidden gradients.
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            if 'proj' in name and parameter.ndim == 2:
                parameter.normal_(std=.02)
    actual = copy.deepcopy(reference)
    original_keys = list(actual.state_dict())
    calls = []
    def sensor(module, args, output):
        calls.append('forward')
        output.register_hook(lambda grad: calls.append('backward'))
    handle = actual.proj.register_forward_hook(sensor)
    enable_compiled_loss(actual)
    assert list(actual.state_dict()) == original_keys
    inputs = torch.randint(64, (2,8))
    targets = torch.randint(64, (2,8))
    expected_loss, actual_loss = reference(inputs,targets), actual(inputs,targets)
    expected_loss.backward(); actual_loss.backward()
    assert calls == ['forward','backward']
    torch.testing.assert_close(actual_loss,expected_loss,rtol=2e-6,atol=2e-5)
    for (name,p), (_,q) in zip(actual.named_parameters(),reference.named_parameters()):
        # BF16 head-adjoint rounding may differ by one representable value after
        # compiler fusion; this is execution parity, not bitwise trajectory proof.
        torch.testing.assert_close(p.grad,q.grad,rtol=.02,atol=2e-4,msg=name)
    handle.remove()


def test_compiled_loss_and_whole_model_compile_are_exclusive():
    with pytest.raises(ValueError,match='hooks must remain eager'):
        validate_config(dict(compile=True,compile_loss=True))
