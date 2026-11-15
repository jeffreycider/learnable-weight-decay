import copy

import pytest
import torch

from scripts.audit_chisel_saved_adam import parameter_pairs, recover_gradient, summarize_role


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_recovers_clipped_gradient_and_local_derivative(dtype):
    torch.manual_seed(64)
    p = torch.nn.Parameter(torch.randn(19, dtype=dtype))
    opt = torch.optim.Adam([p], lr=.003, betas=(.9, .99), eps=1e-8, foreach=False)
    for _ in range(8):
        old_p = p.detach().clone()
        old_state = copy.deepcopy(opt.state.get(p, {}))
        raw = torch.randn_like(p) * 100
        p.grad = raw.clone()
        norm = float(raw.double().norm())
        torch.nn.utils.clip_grad_norm_([p], 1.)
        clipped = p.grad.clone()
        opt.step()
        new_state = opt.state[p]
        previous = old_state.get('exp_avg', torch.zeros_like(p))
        g, error = recover_gradient(previous, new_state['exp_avg'], .9)
        assert torch.all((g-clipped.double()).abs() <= error + 1e-30)
        result = summarize_role([(old_p, p.detach(), old_state, new_state, opt.param_groups[0])], norm, .5)
        expected = float((raw.double() * (p.detach().double()-old_p.double())).sum()) * .5
        assert abs(result['actual']['gradient_dot_step']-expected) <= result['actual']['dot_roundoff_estimate'] + 1e-12


def test_flags_current_gradient_ascent_despite_momentum_step():
    p = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    opt = torch.optim.Adam([p], lr=.01, betas=(.9, .99))
    p.grad = torch.tensor([1.], dtype=p.dtype)
    opt.step()
    old_p, old_state = p.detach().clone(), copy.deepcopy(opt.state[p])
    p.grad = torch.tensor([-.1], dtype=p.dtype)
    opt.step()
    result = summarize_role([(old_p, p.detach(), old_state, opt.state[p], opt.param_groups[0])], .1)
    assert result['actual']['gradient_dot_step'] > 0
    assert result['ideal']['gradient_dot_step'] > 0


def test_reports_erased_fp32_motion_separately():
    p = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float32))
    opt = torch.optim.Adam([p], lr=1e-11)
    old = p.detach().clone()
    p.grad = torch.ones_like(p)
    opt.step()
    result = summarize_role([(old, p.detach(), {}, opt.state[p], opt.param_groups[0])], 1.)
    assert result['actual']['gradient_dot_step'] == 0
    assert result['actual']['cosine'] is None
    assert result['ideal']['gradient_dot_step'] < 0


def test_unused_parameter_must_not_move():
    old = dict(unused=torch.ones(1))
    assert parameter_pairs(old, old, dict(state={}), dict(state={}), ['unused'], dict(params=[0]), 1) == []
    with pytest.raises(AssertionError):
        parameter_pairs(old, dict(unused=torch.zeros(1)), dict(state={}), dict(state={}), ['unused'], dict(params=[0]), 1)
