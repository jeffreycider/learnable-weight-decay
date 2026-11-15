import pytest
import torch
from chisel.preservation import prediction_terms, ranking_changes


def ranking(student, teacher, **kwargs):
    return prediction_terms(student, teacher, ranking_normalization='relative', **kwargs)['ranking']


def test_relative_penalty_detects_small_and_large_reversals_equally():
    teacher = torch.tensor([[2., 1., 0.], [2e-4, 1e-4, 0.]], dtype=torch.float64)
    student = teacher[:, [1, 0, 2]].clone().requires_grad_()
    values = ranking(student, teacher)
    torch.testing.assert_close(values[0], values[1])
    absolute = prediction_terms(student, teacher)['ranking']
    torch.testing.assert_close(absolute[0]/absolute[1], torch.tensor(1e8, dtype=torch.float64))
    gradient = torch.autograd.grad(values.sum(), student)[0]
    assert (gradient[:, 0] < 0).all() and (gradient[:, 1] > 0).all()


def test_rank_floor_caps_scale_without_ignoring_tiny_violations():
    teacher = torch.tensor([[2e-9, 1e-9, 0.]], dtype=torch.float64)
    student = teacher[:, [1, 0, 2]].clone().requires_grad_()
    loss = ranking(student, teacher, rank_gap_floor=1e-6).sum()
    assert loss > 0 and ranking_changes(student, teacher)['rank_changed'] == 1
    assert torch.isfinite(torch.autograd.grad(loss, student)[0]).all()
    assert ranking(teacher, teacher).item() == 0


def test_ties_remain_equal_rank_and_cutoff_crossings_are_penalized():
    teacher = torch.tensor([[3., 2., 2., 1.]], dtype=torch.float64)
    student = torch.tensor([[3., 2.1, 1.9, 1.]], dtype=torch.float64, requires_grad=True)
    assert ranking(student, teacher, top_k=3).item() == 0
    crossing = torch.tensor([[3., 2., 2., 2.1]], dtype=torch.float64, requires_grad=True)
    loss = ranking(crossing, teacher, top_k=3).sum()
    assert loss > 0 and torch.isfinite(torch.autograd.grad(loss, crossing)[0]).all()


def test_relative_margin_gradient_matches_finite_difference():
    teacher = torch.tensor([[2., 1.8, 1.]], dtype=torch.float64)
    student = torch.tensor([[1.7, 1.85, 1.]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda value: ranking(value, teacher).sum(), (student,))


def test_absolute_default_is_unchanged_and_invalid_options_fail():
    teacher = torch.tensor([[2., 1., 0.]])
    student = teacher.flip(-1)
    default = prediction_terms(student, teacher)
    explicit = prediction_terms(student, teacher, ranking_normalization='absolute')
    for key in default:torch.testing.assert_close(default[key], explicit[key], rtol=0, atol=0)
    for kwargs in [dict(ranking_normalization='unknown'), dict(rank_gap_floor=0)]:
        with pytest.raises(ValueError):prediction_terms(student, teacher, **kwargs)
