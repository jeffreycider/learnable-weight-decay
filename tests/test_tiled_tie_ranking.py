import pytest
import torch

from rewriter.ranking_loss import tie_aware_topk_margin_loss
from rewriter.tiled_tie_ranking import tiled_tie_aware_topk_margin_loss

DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA qualification required'))]


def compare(teacher, student, k, *, alpha=.5, tiny=False):
    student = student.requires_grad_()
    actual = tiled_tie_aware_topk_margin_loss(teacher, student, k=k, margin_fraction=alpha)
    expected = tie_aware_topk_margin_loss(teacher.double(), student.double(), k=k, margin_fraction=alpha)
    weight = torch.linspace(-3., 4., actual.numel(), device=student.device,
                            dtype=torch.float64).reshape(actual.shape)
    ga = torch.autograd.grad((actual*weight).sum(), student, retain_graph=True)[0]
    ge = torch.autograd.grad((expected*weight).sum(), student)[0]
    torch.testing.assert_close(actual, expected, rtol=3e-13, atol=1e-26 if tiny else 1e-14)
    assert torch.equal(ga != 0, ge != 0)
    torch.testing.assert_close(ga, ge, rtol=3e-7, atol=1e-16 if tiny else 1e-8)
    return actual, ga


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('k', [1, 10, 100, 137])
def test_random_score_classes_and_weighted_cotangents(device, k):
    torch.manual_seed(417)
    teacher = torch.randint(0, 17, (2, 3, 137)).to(device).double()
    student = torch.randn(2, 3, 137).to(device)
    compare(teacher, student, k, alpha=.37)


@pytest.mark.parametrize('device', DEVICES)
def test_full_classifier_and_original_ties_permit_boundary_exchange(device):
    teacher = torch.tensor([[4., 3., 3., 0.], [4., 3., 3., 0.],
                            [3., 3., 2., 0.]], device=device)
    student = torch.tensor([[4., -2., 3., 0.], [4., 3., -2., 0.],
                            [1., 4., 3., 0.]], device=device)
    loss, _ = compare(teacher, student.clone(), 2)
    assert loss[:2].eq(0).all()
    loss, _ = compare(teacher, student.clone(), 3)
    assert loss[2] > 0  # masked adjacent zero gaps would miss this violation
    compare(teacher, student.clone(), 4)


@pytest.mark.parametrize('device', DEVICES)
def test_exact_extrema_and_cutoff_ties_preserve_reference_subgradient(device):
    teacher = torch.tensor([[5., 5., 4., 4., 4., 3., 3., 2.],
                            [5., 4., 4., 4., 4., 3., 2., 1.]], device=device)
    student = torch.tensor([[1., 1., 4., 4., 4., 6., 6., 0.],
                            [3., 2., 2., 2., 2., 4., 0., 0.]], device=device)
    compare(teacher, student, 4)


@pytest.mark.parametrize('device', DEVICES)
def test_top100_tiny_crossing_nonzero_gradient(device):
    teacher = (1.-torch.arange(137, device=device)*1e-6).repeat(3, 1)
    teacher[:, 50] = teacher[:, 49]  # internal score class
    teacher[:, 100] = teacher[:, 99]  # boundary class extends beyond selected k
    student = teacher.clone()
    student[:, 50] -= 2e-6
    student[:, 101] = teacher[:, 99]+1e-7
    loss, gradient = compare(teacher, student, 100, tiny=True)
    assert loss.gt(0).all() and gradient.count_nonzero()>0


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('vocabulary', [1, 10])
def test_all_teacher_ties_have_zero_loss_and_no_nan_gradient(device, vocabulary):
    teacher = torch.ones(2, vocabulary, device=device)
    student = torch.arange(2*vocabulary, device=device, dtype=torch.float32).reshape(2, vocabulary)
    loss, gradient = compare(teacher, student, vocabulary)
    assert loss.eq(0).all() and gradient.eq(0).all()


def test_small_classifier_gradcheck():
    teacher = torch.tensor([[4., 3., 3., 2., 2., 2., 1., 0., 0., -1.]], dtype=torch.float64)
    student = torch.tensor([[1.1, 2.2, 3.3, 4.1, 2.8, 1.9, 3.4, 5.1, 4.3, .2]],
                           dtype=torch.float64, requires_grad=True)
    for k in (4, 10):
        assert torch.autograd.gradcheck(
            lambda s: tiled_tie_aware_topk_margin_loss(teacher, s, k=k),
            (student,), eps=1e-6, atol=1e-8, rtol=1e-6)
