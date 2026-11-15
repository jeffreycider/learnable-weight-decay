"""Default-off membership acceleration; legacy ranking objective is unchanged."""
import pytest
import torch

from rewriter.ranking_loss import topk_margin_loss
from rewriter.tiled_ranking_bitset import (
    _cpu_membership_bitset, _cpu_membership_lookup, bitset_topk_margin_loss,
)

DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA qualification required'))]


def test_membership_words_sign_bit_tail_and_duplicate_indices():
    indices = torch.tensor([[0, 31, 32, 63, 64, 96, 100, 31],
                            [1, 30, 33, 62, 65, 95, 99, 65]])
    mask = _cpu_membership_bitset(indices, 101)
    actual = _cpu_membership_lookup(mask, 101)
    expected = torch.zeros(2, 101, dtype=torch.bool).scatter(1, indices, True)
    assert mask.dtype == torch.int32 and mask.shape == (2, 4)
    assert mask[0, 0] < 0  # bit 31 survives signed storage
    assert torch.equal(actual, expected)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('k', [1, 10, 100])
def test_fp64_loss_and_weighted_student_cotangents(device, k):
    generator = torch.Generator().manual_seed(374)
    teacher = torch.randn(2, 3, 137, generator=generator).to(device).double().requires_grad_()
    student = torch.randn(2, 3, 137, generator=generator).to(device).requires_grad_()
    actual, (values, indices) = bitset_topk_margin_loss(
        teacher, student, k=k, return_teacher_topk=True)
    expected = topk_margin_loss(teacher, student.double(), k=k)
    weight = torch.tensor([[0., 1., -3.], [.01, 100., -2.]], device=device, dtype=torch.float64)
    ga = torch.autograd.grad((actual*weight).sum(), student, retain_graph=True)[0]
    ge = torch.autograd.grad((expected*weight).sum(), student)[0]
    torch.testing.assert_close(actual, expected, rtol=2e-13, atol=1e-15)
    torch.testing.assert_close(ga, ge, rtol=2e-7, atol=1e-8)
    assert teacher.grad is None
    selected = teacher.detach().reshape(-1, 137).topk(k, sorted=True)
    assert torch.equal(values, selected.values) and torch.equal(indices, selected.indices)
    assert not values.requires_grad


@pytest.mark.parametrize('device', DEVICES)
def test_top100_microgap_crossing_retains_nonzero_weighted_cotangents(device):
    teacher = (1.-torch.arange(137, device=device)*1e-6).repeat(3, 1)
    student = teacher.clone()
    student[:, 51] = teacher[:, 50]+1e-7
    student[:, 100] = teacher[:, 99]+1e-7
    student.requires_grad_()
    actual = bitset_topk_margin_loss(teacher, student, k=100)
    expected = topk_margin_loss(teacher.double(), student.double(), k=100)
    assert torch.all(expected > 0) and torch.all(actual > 0)
    torch.testing.assert_close(actual, expected, rtol=2e-13, atol=1e-26)
    weights = torch.tensor([-3., .01, 100.], device=device, dtype=torch.float64)
    ga = torch.autograd.grad((actual*weights).sum(), student, retain_graph=True)[0]
    ge = torch.autograd.grad((expected*weights).sum(), student)[0]
    assert torch.equal(ga != 0, ge != 0)
    torch.testing.assert_close(ga, ge, rtol=2e-7, atol=1e-16)


@pytest.mark.parametrize('device', DEVICES)
def test_outsider_maximum_ties_share_gradient_exactly(device):
    teacher = torch.tensor([[5., 4., 3., 2., 1.]], device=device)
    student = torch.tensor([[0., 0., 6., 5.5, 0.]], device=device, requires_grad=True)
    actual = bitset_topk_margin_loss(teacher, student, k=2)
    expected = topk_margin_loss(teacher.double(), student.double(), k=2)
    ga = torch.autograd.grad(actual.sum(), student, retain_graph=True)[0]
    ge = torch.autograd.grad(expected.sum(), student)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(ga, ge, rtol=0, atol=0)
    assert ga[0, 2] == ga[0, 3] > 0


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('vocabulary', [1, 10])
def test_full_classifier_vocabulary_has_no_membership_term(device, vocabulary):
    teacher = torch.arange(vocabulary, device=device, dtype=torch.float32)[None, :]
    student = (-teacher).clone().requires_grad_()
    actual = bitset_topk_margin_loss(teacher, student, k=vocabulary)
    values, indices = teacher.double().topk(vocabulary)
    edited = student.double().gather(-1, indices)
    expected = (.5*(values[:, :-1]-values[:, 1:])
                -(edited[:, :-1]-edited[:, 1:])).relu().square().sum(-1)/vocabulary
    ga = torch.autograd.grad(actual.sum(), student, retain_graph=True)[0]
    ge = torch.autograd.grad(expected.sum(), student)[0]
    assert torch.isfinite(actual).all() and torch.isfinite(ga).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(ga, ge, rtol=0, atol=0)


def test_first_derivative_gradcheck():
    torch.manual_seed(991)
    teacher = torch.randn(2, 113, dtype=torch.float64)
    student = torch.randn(2, 113, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda s: bitset_topk_margin_loss(teacher, s, k=100), (student,),
        eps=1e-6, atol=1e-8, rtol=1e-6, fast_mode=True)
