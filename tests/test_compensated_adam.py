import copy

import pytest
import torch

from rewriter.update_accumulation import CompensatedAdam


def test_tiny_updates_accumulate_with_identical_native_moments_and_clocks():
    p = torch.nn.Parameter(torch.ones(3))
    q = torch.nn.Parameter(p.detach().clone())
    opt = CompensatedAdam([p], lr=1e-9, betas=(.9, .99))
    native = torch.optim.Adam([q], lr=1e-9, betas=(.9, .99), foreach=False, fused=False)
    for _ in range(128):
        p.grad = torch.ones_like(p)
        q.grad = p.grad.clone()
        opt.step(); native.step()
    assert not torch.equal(p, q)
    assert torch.equal(q, torch.ones_like(q))
    assert torch.equal(p, torch.full_like(p, 1 - 128e-9))
    for key in ('step', 'exp_avg', 'exp_avg_sq'):
        assert torch.equal(opt.state[p][key], native.state[q][key])


def test_mixed_precision_keeps_fp64_adam_identical_and_none_grad_is_noop():
    p = torch.nn.Parameter(torch.ones(3))
    d = torch.nn.Parameter(torch.ones(3, dtype=torch.float64))
    reference = torch.nn.Parameter(d.detach().clone())
    opt = CompensatedAdam([p, d], lr=1e-9)
    native = torch.optim.Adam([reference], lr=1e-9, foreach=False, fused=False)
    for i in range(4):
        p.grad = torch.ones_like(p)
        d.grad = torch.full_like(d, i + 1.)
        reference.grad = d.grad.clone()
        opt.step(); native.step()
    assert torch.equal(d, reference)
    before_p = p.detach().clone()
    before_state = copy.deepcopy(opt.state[p])
    p.grad = None; d.grad = None
    opt.step()
    assert torch.equal(p, before_p)
    assert all(torch.equal(v, opt.state[p][key]) for key, v in before_state.items())
    assert 'fp32_update_remainder' not in opt.state[d]


def test_resume_preserves_remainders_moments_and_future_weights():
    p = torch.nn.Parameter(torch.ones(4))
    opt = CompensatedAdam([p], lr=1e-9, betas=(.9, .99))
    for _ in range(17):
        p.grad = torch.ones_like(p); opt.step()
    restored_p = torch.nn.Parameter(p.detach().clone())
    restored_opt = CompensatedAdam([restored_p], lr=1e-9, betas=(.9, .99))
    restored_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    for i in range(80):
        grad = torch.full_like(p, 1. if i % 3 else -.3)
        p.grad = grad; restored_p.grad = grad.clone()
        opt.step(); restored_opt.step()
    assert torch.equal(p, restored_p)
    assert all(torch.equal(v, restored_opt.state[restored_p][key]) for key, v in opt.state[p].items())


def test_closure_is_called_once_and_state_advances_once():
    p = torch.nn.Parameter(torch.ones(1))
    opt = CompensatedAdam([p], lr=1e-9)
    calls = []
    def closure():
        calls.append(1); opt.zero_grad(); loss = p.square().sum(); loss.backward(); return loss
    loss = opt.step(closure)
    assert len(calls) == 1 and float(loss.detach()) == 1 and opt.state[p]['step'] == 1


def test_zero_lr_keeps_existing_remainder_and_weights_but_advances_moments():
    p = torch.nn.Parameter(torch.ones(1))
    opt = CompensatedAdam([p], lr=1e-9)
    for _ in range(10):
        p.grad = torch.ones_like(p); opt.step()
    old_p = p.detach().clone()
    old_remainder = opt.state[p]['fp32_update_remainder'].clone()
    opt.param_groups[0]['lr'] = 0
    opt.step()
    assert torch.equal(p, old_p)
    assert torch.equal(opt.state[p]['fp32_update_remainder'], old_remainder)
    assert opt.state[p]['step'] == 11


def test_native_adam_state_cannot_masquerade_as_exact_resume():
    p = torch.nn.Parameter(torch.ones(1))
    native = torch.optim.Adam([p])
    opt = CompensatedAdam([p])
    with pytest.raises(ValueError, match='silently convert'):
        opt.load_state_dict(native.state_dict())


def test_missing_remainder_is_rejected():
    p = torch.nn.Parameter(torch.ones(1))
    opt = CompensatedAdam([p]); p.grad = torch.ones_like(p); opt.step()
    state = copy.deepcopy(opt.state_dict())
    del state['state'][0]['fp32_update_remainder']
    with pytest.raises(ValueError, match='remainder'):
        opt.load_state_dict(state)


@pytest.mark.parametrize('group', [dict(weight_decay=.1), dict(amsgrad=True), dict(foreach=True)])
def test_unsupported_groups_are_rejected(group):
    with pytest.raises(ValueError, match='Unsupported'):
        CompensatedAdam([dict(params=[torch.nn.Parameter(torch.ones(2))], **group)])
