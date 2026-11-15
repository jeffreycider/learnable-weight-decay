"""Independent ownership, zero-head and guard-value checks for adaptive credit."""
import pytest
import torch

from rewriter.adaptive_action import AdaptiveAction, AdaptiveFactors
from rewriter.adaptive_credit import AdaptiveCreditRecord, project_adaptive_gradients, adaptive_credit_contraction
from rewriter.lag_one import LagOneHypergradient, tensor_leaves


def example(zero_core=False):
    torch.manual_seed(310)
    left=torch.randn(4,2,dtype=torch.float64,requires_grad=True)
    right=torch.randn(5,2,dtype=torch.float64,requires_grad=True)
    core=(torch.zeros(2,2,dtype=torch.float64) if zero_core else torch.randn(2,2,dtype=torch.float64)).requires_grad_()
    amplitude=torch.tensor(.2,dtype=torch.float64,requires_grad=True)
    action=AdaptiveAction({'matrix':AdaptiveFactors(torch.linalg.qr(left).Q,core,torch.linalg.qr(right).Q)}, {})
    radius=torch.minimum(core.new_ones(()),core.new_tensor(.3)/action.norm_sq().clamp_min(1e-24).sqrt())
    scale=amplitude.sigmoid()*radius
    parameters=dict(left=left,right=right,core=core,amplitude=amplitude)
    return action,scale,parameters


@pytest.mark.parametrize('zero_core',[False,True])
def test_independent_scale_parameter_and_zero_head_paths_match_explicit_update(zero_core):
    action,scale,parameters=example(zero_core)
    gradient=torch.randn(4,5,dtype=torch.float64)
    explicit=(scale*action.matrices['matrix'].dense()*gradient).sum()
    expected=torch.autograd.grad(explicit,tuple(parameters.values()),retain_graph=True)
    projected=project_adaptive_gradients(action,{'matrix':gradient},global_token_count=8,loss_reduction='global_mean')
    surrogate=adaptive_credit_contraction(action,scale,projected)
    torch.testing.assert_close(surrogate,explicit,rtol=1e-12,atol=1e-12)
    for got,want in zip(torch.autograd.grad(surrogate,tuple(parameters.values())),expected):
        torch.testing.assert_close(got,want,rtol=1e-12,atol=1e-12)
    if zero_core:
        assert expected[2].norm()>0  # the zero output head can start learning
        assert expected[0].count_nonzero()==expected[1].count_nonzero()==0
        assert expected[3]==0


def test_projection_owns_all_thin_payloads_before_dense_buffer_reuse():
    action,_,_=example()
    gradient=torch.randn(4,5,dtype=torch.float64)
    projected=project_adaptive_gradients(action,{'matrix':gradient},global_token_count=8,loss_reduction='global_token_sum')
    saved=[v.clone() for v in tensor_leaves(projected)]
    gradient.fill_(123)
    for got,want in zip(tensor_leaves(projected),saved):
        assert not got.requires_grad
        torch.testing.assert_close(got,want,rtol=0,atol=0)


@pytest.mark.parametrize('mutate', ['private_parameter','action_factor'])
def test_pending_owner_must_guard_private_parameters_and_derived_action_factors(mutate):
    action,scale,parameters=example()
    factor=action.matrices['matrix']
    lag=LagOneHypergradient()
    # Parameters alone do not protect independently mutated derived factor
    # storage. The pending owner explicitly binds the retained action too.
    lag.from_contraction(action_step=5,hyperparameters=parameters,
        guarded_tensors=(factor.U,factor.C,factor.V,scale),
        contraction=lambda values:adaptive_credit_contraction(action,scale,values))
    projected=project_adaptive_gradients(action,{'matrix':torch.ones(4,5,dtype=torch.float64)},
        global_token_count=8,loss_reduction='global_mean')
    with torch.no_grad():
        (parameters['left'] if mutate=='private_parameter' else factor.U).add_(.01)
    with pytest.raises(RuntimeError,match='mutated'):
        lag.consume(projected,gradient_step=6)


def test_record_owns_factor_mapping_even_if_caller_reuses_action_dictionary():
    action,scale,parameters=example()
    gradient=torch.randn(4,5,dtype=torch.float64)
    expected=torch.autograd.grad(scale*action.matrices['matrix'].contract(gradient),
                                 tuple(parameters.values()),retain_graph=True)
    lag=LagOneHypergradient()
    record=AdaptiveCreditRecord(lag,action,scale,action_step=5,hyperparameters=parameters)
    factor=action.matrices['matrix']
    # Frozen dataclass attributes do not freeze the dict itself. This mutation
    # changes no tensor version counter; ownership must protect the mapping.
    action.matrices['matrix']=AdaptiveFactors(factor.U,2*factor.C,factor.V)
    projected=record.project({'matrix':gradient},global_token_count=8,loss_reduction='global_mean')
    result=lag.consume(projected,gradient_step=6)
    for (name,_),wanted in zip(parameters.items(),expected):
        torch.testing.assert_close(result.gradients[name],wanted,rtol=1e-12,atol=1e-12)
