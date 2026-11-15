import pytest
import torch

from rewriter.ranking_loss import topk_margin_loss


def test_identity_offsets_and_margin_preserving_shrink_are_zero():
    t = torch.tensor([[4., 3., 2., 1., 0.]], dtype=torch.float64)
    for s in (t, t + 70, 0.75 * t + 4):
        assert topk_margin_loss(t, s, k=3).item() == 0


def test_detects_order_and_membership_without_argmax_change():
    t = torch.tensor([[5., 4., 3., 2.], [5., 4., 3., 2.]])
    s = torch.tensor([[5., 3., 4., 2.], [5., 4., 1., 2.]])
    assert (topk_margin_loss(t, s, k=3) > 0).all()
    assert (t.argmax(-1) == s.argmax(-1)).all()


def test_worst_outside_violation_matches_explicit_all_pairs():
    gen = torch.Generator().manual_seed(7)
    t = torch.randn(2, 3, 17, generator=gen, dtype=torch.float64)
    s = torch.randn(2, 3, 17, generator=gen, dtype=torch.float64)
    for k in (1, 5, 16):
        expected = []
        for a, b in zip(t.flatten(0, 1), s.flatten(0, 1)):
            ids = a.argsort(descending=True).tolist()
            internal = sum((0.5 * (a[i] - a[j]) - (b[i] - b[j])).relu().square()
                           for i, j in zip(ids[:k-1], ids[1:k]))
            boundary = max(0.5 * (a[ids[k-1]] - a[j]) - (b[ids[k-1]] - b[j])
                           for j in ids[k:]).relu().square()
            expected.append((internal + boundary) / k)
        torch.testing.assert_close(topk_margin_loss(t, s, k=k).flatten(), torch.stack(expected))


def test_student_gradient_and_teacher_stop_gradient():
    t = torch.tensor([[4., 3., 2., 1.]], dtype=torch.float64, requires_grad=True)
    s = torch.tensor([[4., 2.1, 3., 2.8]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: topk_margin_loss(t, x, k=3), (s,))
    topk_margin_loss(t, s, k=3).sum().backward()
    assert t.grad is None and s.grad.abs().sum() > 0
    assert s.grad.sum().abs() < 1e-12  # Common logit offsets do not matter.


def test_teacher_ties_have_no_artificial_positive_margin():
    t = torch.ones(1, 4)
    assert topk_margin_loss(t, t, k=2).item() == 0


def test_invalid_arguments():
    with pytest.raises(ValueError):
        topk_margin_loss(torch.zeros(1, 4), torch.zeros(1, 4), k=4)
    with pytest.raises(ValueError):
        topk_margin_loss(torch.zeros(1, 4), torch.zeros(1, 4), k=2, margin_fraction=0)
