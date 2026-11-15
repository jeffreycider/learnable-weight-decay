import copy
import math

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from rewriter.device_primal_dual import (DevicePrimalDualInputs,
    ordered_panel_mean, read_terms, scale_cotangent)
from rewriter.primal_dual import PrimalDual


class NoHostScalar(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten._local_scalar_dense.default:
            raise AssertionError('hot path read a tensor back to the host')
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('sizes', [(16, 16, 16, 16), (3, 3, 1)])
def test_ordered_mean_matches_python_scalar_operations(dtype, sizes):
    generator = torch.Generator().manual_seed(41)
    for _ in range(30):
        losses = [x for x in torch.randn(len(sizes), generator=generator, dtype=dtype)*1e-6]
        expected = 0.
        for loss, size in zip(losses, sizes):
            fraction = size/sum(sizes)
            expected += float(loss)*fraction
        with NoHostScalar():
            actual = ordered_panel_mean(losses, sizes)
        assert float(actual) == expected


def test_terms_cast_boundaries_and_changed_dual_are_exact():
    dual = PrimalDual([1, 2], tolerance=1e-6, rho=.01, dual_lr=.1)
    staged = DevicePrimalDualInputs(dual, device='cpu')
    pointers = [x.data_ptr() for x in (staged.multiplier, staged.tolerance, staged.rho)]
    with pytest.raises(RuntimeError, match='bind'):
        staged.terms(torch.tensor(1e-6, dtype=torch.float64))
    values = [-1., -1e-15, -0., 0., math.nextafter(1e-6, 0.), 1e-6,
              math.nextafter(1e-6, math.inf), 2.18e-6, 1., 1e20]
    gradient = torch.tensor([1.234567, -2.345678, 1e-20, 1e-39, -1e-39, 0.], dtype=torch.float32)
    for multiplier in (0., 1e-8, .1, .1234567891234, 1e5):
        dual.multipliers[1] = multiplier
        dual.updates += 1
        staged.bind(dual, 1)
        for value in values:
            kl = torch.tensor(value, dtype=torch.float64)
            expected = dual.terms(value, 1)
            with NoHostScalar():
                actual = staged.terms(kl)
                scaled = scale_cotangent(gradient, actual, 1/200)
            assert read_terms(actual) == expected
            assert torch.equal(scaled, gradient*(expected['kl_coefficient']*(1/200)))
    assert pointers == [x.data_ptr() for x in (staged.multiplier, staged.tolerance, staged.rho)]
    assert staged.bound_update == dual.updates
    with pytest.raises(ValueError, match='fitting'):
        staged.bind(dual, 5)
    staged.bind(dual, 5, audit=True)
    assert read_terms(staged.terms(torch.tensor(2e-6, dtype=torch.float64))) == dual.terms(2e-6, 5)
    changed = PrimalDual([1], tolerance=2e-6)
    with pytest.raises(ValueError, match='configuration'):
        staged.bind(changed, 1)


@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf])
def test_nonfinite_is_rejected_before_optimizer(value):
    dual = PrimalDual([1], tolerance=1e-6)
    staged = DevicePrimalDualInputs(dual, device='cpu')
    staged.bind(dual, 1)
    with NoHostScalar():
        terms = staged.terms(torch.tensor(value, dtype=torch.float64))
    with pytest.raises(FloatingPointError, match='nonfinite KL'):
        read_terms(terms)
    with pytest.raises(FloatingPointError, match='nonfinite KL'):
        dual.terms(value, 1)


def test_three_adam_steps_with_full_panel_coefficient_are_exact():
    """Exercise live multipliers and mixed-precision cast at the VJP boundary."""
    from scripts.train_multistate_chisel import mean_kl_and_backward
    torch.set_num_threads(1)
    sizes = (3, 3, 1)
    generator = torch.Generator().manual_seed(888)
    features = torch.randn(7, 2, 5, generator=generator)
    target = torch.randn(7, 5, generator=generator).double().log_softmax(-1)
    class Engine:
        microbatch_sequences = 3
        def student_parameter_loss(self, parameters, teacher):
            start, stop = teacher
            logits = (parameters['weight'][None, :, None]*features[start:stop]).sum(1)
            logp = logits.double().log_softmax(-1)
            logq = target[start:stop]
            return (logq.exp()*(logq-logp)).sum(-1).mean()
    engine = Engine()
    def run(device_terms):
        parameter = torch.nn.Parameter(torch.tensor([.03, -.02]))
        optimizer = torch.optim.Adam([parameter], lr=2e-6)
        dual = PrimalDual([1, 2], tolerance=1e-6, rho=.01, dual_lr=.1)
        staged = DevicePrimalDualInputs(dual, device='cpu')
        history = []
        for state_id in (1, 2, 1):
            optimizer.zero_grad(set_to_none=True)
            output = parameter * (1. + state_id*.1)
            detached = output.detach().requires_grad_(True)
            regularizer = .13 * output.square().sum() - output.sum()
            if device_terms:
                losses = []
                start = 0
                for size in sizes:
                    kl = engine.student_parameter_loss({'weight': detached}, (start, start+size))
                    losses.append(kl.detach())
                    (kl*(size/sum(sizes))).backward()
                    start += size
                staged.bind(dual, state_id)
                mean = ordered_panel_mean(losses, sizes)
                terms = staged.terms(mean)
                cotangent = scale_cotangent(detached.grad, terms, 1/200)
                measured = read_terms(terms)
                total = float(mean)
                torch.autograd.backward([regularizer, output],
                                       [regularizer.new_tensor(1/200), cotangent])
            else:
                measured = mean_kl_and_backward(engine, {'weight': output}, regularizer,
                    torch.arange(7), kl_budget=dual.tolerance, state_weight=1/200,
                    teacher_provider=lambda start, stop: (start, stop),
                    primal_dual=dual, state_id=state_id)
                total = measured.pop('mean_kl')
                measured.pop('objective')
            gradient = parameter.grad.clone()
            torch.nn.utils.clip_grad_norm_([parameter], 1., error_if_nonfinite=True)
            optimizer.step()
            dual.step([(state_id, total)], expected_ids=[state_id])
            history.append((total, measured, gradient, parameter.detach().clone()))
        return history, copy.deepcopy(optimizer.state_dict()), dual.state_dict()
    reference, ref_opt, ref_dual = run(False)
    actual, opt, dual = run(True)
    for a, b in zip(reference, actual):
        assert a[:2] == b[:2]
        assert torch.equal(a[2], b[2]) and torch.equal(a[3], b[3])
    for name, value in ref_opt['state'][0].items():
        assert torch.equal(value, opt['state'][0][name])
    assert ref_dual == dual


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires owner-coordinated CUDA node')
def test_cuda_graph_replay_uses_updated_losses_and_multipliers():
    dual = PrimalDual([1, 2], tolerance=1e-6, rho=.01, dual_lr=.1)
    staged = DevicePrimalDualInputs(dual, device='cuda')
    staged.bind(dual, 1)
    losses = [torch.tensor(2e-6, dtype=torch.float64, device='cuda') for _ in range(4)]
    gradient = torch.tensor([1.234567, -.25, 1e-39, -1e-39], device='cuda')
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            terms = staged.terms(ordered_panel_mean(losses, [16]*4))
            scale_cotangent(gradient, terms, 1/200)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mean = ordered_panel_mean(losses, [16]*4)
        terms = staged.terms(mean)
        scaled = scale_cotangent(gradient, terms, 1/200)
    for state_id, multiplier, values in [(1, .123456789, [1e-6, 2e-6, -1e-8, 3e-6]),
                                        (2, .333333333, [-1e-8]*4),
                                        (1, .02, [0.]*4)]:
        dual.multipliers[state_id] = multiplier
        staged.bind(dual, state_id)
        for loss, value in zip(losses, values):loss.fill_(value)
        graph.replay()
        expected_mean = 0.
        for value in values:expected_mean += value*.25
        expected = dual.terms(expected_mean, state_id)
        assert float(mean) == expected_mean
        assert read_terms(terms) == expected
        assert torch.equal(scaled, gradient*(expected['kl_coefficient']*(1/200)))
