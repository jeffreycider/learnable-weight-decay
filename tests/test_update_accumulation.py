import io

import pytest
import torch

from rewriter.update_accumulation import compensated_add_fp32_


def test_sub_ulp_steps_eventually_change_forward_parameter():
    p = torch.tensor([1., -1.], requires_grad=True)
    naive = p.detach().clone()
    remainder = torch.zeros_like(p)
    update = torch.tensor([1e-10, -1e-10])
    original_update = update.clone()
    reference = p.detach().double()
    for _ in range(2000):
        naive.add_(update)
        compensated_add_fp32_(p, update, remainder)
        reference.add_(update.double())
    assert torch.equal(naive, torch.tensor([1., -1.]))
    assert torch.equal(p.detach(), reference.float())
    assert not torch.equal(p.detach(), naive)
    # The remainder itself is FP32. Allow accumulated rounding at the scale of
    # a parameter ULP, while requiring the forward value to equal rounded FP64.
    bound = 2000 * torch.finfo(torch.float32).eps ** 2
    torch.testing.assert_close(p.detach().double() + remainder.double(), reference, rtol=0, atol=bound)
    assert torch.equal(update, original_update)
    assert p.grad is None and remainder.grad_fn is None


def test_mixed_signed_steps_and_noncontiguous_storage_track_double_reference():
    generator = torch.Generator().manual_seed(42)
    p = torch.randn(7, 5, generator=generator).T
    assert not p.is_contiguous()
    remainder = torch.zeros_like(p)
    reference = p.double()
    for _ in range(300):
        update = torch.randn(p.shape, generator=generator) * 1e-9
        compensated_add_fp32_(p, update, remainder)
        reference.add_(update.double())
    torch.testing.assert_close(p.double() + remainder.double(), reference, rtol=0, atol=5e-13)
    torch.testing.assert_close(p, reference.float(), rtol=0, atol=2.4e-7)


def test_saved_remainder_makes_resume_identical():
    p, remainder = torch.ones(4), torch.zeros(4)
    update = torch.full((4,), 1e-9)
    for _ in range(17):
        compensated_add_fp32_(p, update, remainder)
    buffer = io.BytesIO()
    torch.save(dict(parameter=p, remainder=remainder), buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=True)
    for _ in range(80):
        compensated_add_fp32_(p, update, remainder)
        compensated_add_fp32_(restored['parameter'], update, restored['remainder'])
    assert torch.equal(restored['parameter'], p)
    assert torch.equal(restored['remainder'], remainder)


@pytest.mark.parametrize('kind', ['dtype', 'shape'])
def test_invalid_accumulator_does_not_mutate_parameter(kind):
    p = torch.ones(4)
    update = torch.ones(4)
    remainder = torch.zeros(4, dtype=torch.float64) if kind == 'dtype' else torch.zeros(3)
    original = p.clone()
    with pytest.raises(ValueError):
        compensated_add_fp32_(p, update, remainder)
    assert torch.equal(p, original)
