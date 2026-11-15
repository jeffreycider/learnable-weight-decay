import torch
from chisel.logit_diagnostics import logit_errors, summarize


def test_common_shift_is_reported_but_probability_invariant():
    teacher = torch.tensor([[1., 2., 3.], [-3., 0., 3.]], dtype=torch.float64)
    metrics = summarize([logit_errors(teacher + 2, teacher)])
    assert metrics['raw_rms_mean'] == 2
    assert metrics['centered_rms_max'] == 0
    assert metrics['probability_max_max'] == 0


def test_summary_uses_all_examples_not_mean_of_microbatch_quantiles():
    teacher = torch.zeros(5, 2, dtype=torch.float64)
    edited = torch.tensor([[0.,0.], [1.,-1.], [2.,-2.], [3.,-3.], [4.,-4.]])
    full = summarize([logit_errors(edited, teacher)])
    split = summarize([logit_errors(edited[:1], teacher[:1]), logit_errors(edited[1:], teacher[1:])])
    assert full == split
    assert full['centered_max_max'] == 4
    assert full['centered_rms_mean'] == 2


def test_raw_logit_objective_penalizes_common_shifts_when_selected():
    from chisel.preservation import prediction_terms
    teacher = torch.tensor([[1., 2., 3.]], dtype=torch.float64)
    student = (teacher + .01).requires_grad_()
    centered = prediction_terms(student, teacher)['mse']
    raw = prediction_terms(student, teacher, center_logits=False)['mse']
    torch.testing.assert_close(raw, torch.tensor([1e-4], dtype=torch.float64))
    assert centered.max() < 1e-25
    raw.sum().backward()
    assert student.grad.min() > 0


def test_optional_unused_terms_leave_logit_gradient_unchanged():
    from chisel.preservation import prediction_terms
    t = torch.randn(3, 7)
    s = torch.randn(3, 7, requires_grad=True)
    full = prediction_terms(s, t, center_logits=False)
    lean = prediction_terms(s, t, center_logits=False, compute_kl=False, compute_ranking=False)
    torch.testing.assert_close(full['mse'], lean['mse'])
    torch.testing.assert_close(torch.autograd.grad(full['mse'].sum(), s)[0],
                               torch.autograd.grad(lean['mse'].sum(), s)[0])
