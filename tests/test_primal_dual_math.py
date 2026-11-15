"""Independent checks against direct differentiation of the stated objective."""
from types import SimpleNamespace

import pytest
import torch

from rewriter.primal_dual import PrimalDual
from scripts.train_multistate_chisel import mean_kl_and_backward


@pytest.mark.parametrize('kl', [0.002, 0.008, 0.012, 0.03])
def test_kl_coefficient_matches_finite_difference_on_both_sides(kl):
    dual = PrimalDual([7], tolerance=0.01, rho=1.3, dual_lr=0.1)
    dual.multipliers[7] = 0.7
    h = 1e-7
    numerical = (dual.terms(kl+h, 7)['kl_penalty']
                 - dual.terms(kl-h, 7)['kl_penalty']) / (2*h)
    assert dual.terms(kl, 7)['kl_coefficient'] == pytest.approx(numerical, rel=1e-9)


class ToyPredictor:
    def __init__(self, microbatch):
        self.microbatch_sequences = microbatch
        self.forwards = 0

    def cache_teacher(self, inputs, **_):
        weights = inputs.new_tensor([[0.2, -0.4], [0.3, 0.7]])
        return [SimpleNamespace(inputs=inputs, logq=(inputs @ weights).log_softmax(-1))]

    def student_parameter_loss(self, parameters, teacher):
        self.forwards += 1
        logp = (teacher.inputs @ parameters['weight']).log_softmax(-1)
        return (teacher.logq.exp() * (teacher.logq - logp)).sum(-1).mean()


@pytest.mark.parametrize('microbatch', [1, 3, 7])
@pytest.mark.parametrize('tolerance', [0.001, 1.0])
def test_accumulated_primal_gradient_matches_direct_objective(microbatch, tolerance):
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(52)
    inputs = torch.randn(7, 2, generator=generator, dtype=torch.float64)
    direction = inputs.new_tensor([[0.4, -0.2], [0.6, 0.1]])
    dual = PrimalDual([7], tolerance=tolerance, rho=1.3, dual_lr=0.1)
    dual.multipliers[7] = 0.7
    original_dual = dual.state_dict()
    theta = inputs.new_tensor(0.07, requires_grad=True)
    engine = ToyPredictor(microbatch)
    # A nonlinear shrinkage term ensures the split VJP includes both losses.
    shrink_loss = 0.2 * theta.square() - 0.3 * theta
    measured = mean_kl_and_backward(
        engine, {'weight': direction * theta.sin()}, shrink_loss, inputs,
        kl_budget=tolerance, primal_dual=dual, state_id=7, state_weight=0.25)
    assert engine.forwards == (len(inputs)+microbatch-1)//microbatch
    assert dual.state_dict() == original_dual  # backward cannot advance dual state

    explicit = inputs.new_tensor(0.07, requires_grad=True)
    teacher = engine.cache_teacher(inputs)[0]
    kl = engine.student_parameter_loss({'weight': direction * explicit.sin()}, teacher)
    violation = kl.clamp_min(0) / tolerance - 1
    full_loss = (0.2 * explicit.square() - 0.3 * explicit
                 + 0.7 * violation + 0.5 * 1.3 * violation.clamp_min(0).square())
    (0.25 * full_loss).backward()
    torch.testing.assert_close(theta.grad, explicit.grad, rtol=1e-11, atol=1e-12)
    assert measured['objective'] == pytest.approx(float(full_loss.detach()), rel=1e-12)
    assert measured['mean_kl'] == pytest.approx(float(kl.detach()), abs=1e-14)
