import copy
import pytest
import torch

from rewriter.qualification_metrics import (tensor_comparison, metric_within,
    parameter_delta, optimizer_snapshot, compare_adam, logit_response_comparison,
    aggregate_recorded_errors)


def test_recorded_error_aggregation_preserves_subgroup_denominators():
    reference = {'position_ids':torch.tensor([1000000., 2000000.]),
                 'feature':torch.tensor([3., 4.]), 'empty':torch.empty(0)}
    candidate = {k:v.clone() for k,v in reference.items()}
    candidate['feature'][0] += 1
    records = {k:tensor_comparison({k:v},{k:candidate[k]}) for k,v in reference.items()}
    whole = aggregate_recorded_errors(records)
    direct = tensor_comparison(reference,candidate)
    assert whole['relative_error'] == pytest.approx(direct['relative_error'])
    feature = aggregate_recorded_errors({k:records[k] for k in ['feature','empty']})
    assert feature['relative_error'] == pytest.approx(.2)
    assert feature['absolute_l2_error'] == whole['absolute_l2_error'] == 1.
    assert feature['tensor_count'] == 2
    zero = dict(reference_norm=0.,absolute_l2_error=1.,max_absolute_error=1.)
    assert aggregate_recorded_errors({'zero':zero})['relative_error'] is None
    for bad in (-1.,float('nan'),float('inf'),True):
        with pytest.raises(ValueError,match='invalid recorded'):
            aggregate_recorded_errors({'bad':{**zero,'reference_norm':bad}})
    with pytest.raises(ValueError,match='nonempty'):
        aggregate_recorded_errors({})


def test_zero_and_nonfinite_metrics_fail_closed():
    zero = {'p': torch.zeros(3)}
    assert metric_within(tensor_comparison(zero, zero), .003, .99999)
    assert not metric_within(tensor_comparison(zero, {'p': torch.ones(3)}), .003)
    for value in (float('nan'), float('inf')):
        assert not metric_within(tensor_comparison(zero, {'p': torch.full((3,), value)}), .003)
    bad = dict(finite=True, relative_error=0., cosine=float('nan'))
    assert not metric_within(bad, .003)
    signed = tensor_comparison({'p':torch.tensor(0.)}, {'p':torch.tensor(-0.)})
    assert signed['relative_error'] == 0.
    assert not signed['bit_identical']


def test_adam_delta_uses_update_not_large_parameter_norm():
    initial = {'p': torch.tensor([1000., -1000.])}
    a = {'p': initial['p']+torch.tensor([.001, -.001])}
    b = {'p': initial['p']+torch.tensor([.002, -.002])}
    assert metric_within(tensor_comparison(a, b), .003)
    delta = tensor_comparison(parameter_delta(a, initial), parameter_delta(b, initial))
    assert not metric_within(delta, .003)
    assert delta['relative_error'] == pytest.approx(1.0625)


def test_saved_adam_moments_groups_and_counters():
    params = {'head': torch.nn.Parameter(torch.tensor([.1, .3])),
              'representation': torch.nn.Parameter(torch.tensor([.2]))}
    optimizer = torch.optim.Adam([{'params':[params['head']]},
                                  {'params':[params['representation']], 'lr':2e-4}], lr=2e-5)
    for p in params.values():p.grad = torch.ones_like(p)
    optimizer.step()
    a = optimizer_snapshot(optimizer, params)
    b = copy.deepcopy(a)
    exact = compare_adam(a, b)
    assert exact['exp_avg']['global_metric']['bit_identical']
    b['states']['head']['exp_avg'].mul_(2)
    changed = compare_adam(a, b)
    assert not metric_within(changed['exp_avg']['groups'][0], .003)
    assert changed['exp_avg']['groups'][1]['bit_identical']
    b['states']['head']['step'].add_(1)
    with pytest.raises(ValueError, match='counter'):
        compare_adam(a, b)


def test_chunked_logit_response_reports_correct_denominators():
    torch.manual_seed(9)
    before=torch.randn(2,5,17)
    a=before+torch.randn_like(before)*.001
    b=a+torch.randn_like(before)*.00001
    expected=tensor_comparison({'p':a-before},{'p':b-before})
    actual=logit_response_comparison(before,a,b,chunk_tokens=3)
    assert actual['response_relative_error']==pytest.approx(expected['relative_error'],rel=1e-14)
    assert actual['absolute_response_l2_error']==pytest.approx(expected['absolute_l2_error'],rel=1e-14)
    assert actual['absolute_logit_rms_error']==pytest.approx(float((b.double()-a.double()).square().mean().sqrt()))
    assert actual['logit_elements']==170
    assert logit_response_comparison(before,before,before)['response_relative_error']==0
    bad=b.clone();bad[0,0,0]=float('nan')
    assert not logit_response_comparison(before,a,bad)['finite']
