import pytest
import torch
from rewriter.lag_one import LagOneHypergradient,frozen_copy


def test_sgd_learning_rate_credit_is_analytic_and_runs_no_extra_model_pass():
    calls=0
    def task(w):
        nonlocal calls
        calls+=1;return .5*(w-3).square().sum()
    w=torch.tensor([1.,2.],dtype=torch.float64,requires_grad=True)
    old_gradient,=torch.autograd.grad(task(w),w)
    eta=torch.tensor(.2,dtype=torch.float64,requires_grad=True)
    proposal=w.detach()-eta*old_gradient.detach();credit=LagOneHypergradient()
    credit.from_outputs(action_step=4,hyperparameters={'learning_rate':eta},outputs={'weight':proposal})
    actual=proposal.detach().requires_grad_();next_gradient,=torch.autograd.grad(task(actual),actual)
    before=calls;result=credit.consume({'weight':next_gradient},gradient_step=5)
    assert calls==before==2 and not credit.pending
    torch.testing.assert_close(result.gradients['learning_rate'],-(next_gradient*old_gradient).sum())
    result.assign_gradients({'learning_rate':eta});torch.testing.assert_close(eta.grad,result.gradients['learning_rate'])
    with pytest.raises(RuntimeError,match='no executed'):credit.consume({'weight':next_gradient},gradient_step=6)


def test_learned_left_right_residual_matches_explicit_next_loss_autograd():
    torch.manual_seed(3);w=torch.randn(3,4,dtype=torch.float64);g=torch.randn_like(w)
    left=torch.randn(3,3,dtype=torch.float64,requires_grad=True);right=torch.randn(4,4,dtype=torch.float64,requires_grad=True)
    lr=torch.tensor(.03,dtype=torch.float64,requires_grad=True);x=torch.randn(5,4,dtype=torch.float64);target=torch.randn(5,3,dtype=torch.float64)
    proposal=w-lr*(g+left@g+g@right)
    direct_loss=(x@proposal.T-target).square().mean();expected=torch.autograd.grad(direct_loss,(left,right,lr),retain_graph=True)
    actual=proposal.detach().requires_grad_();ordinary_loss=(x@actual.T-target).square().mean();gradient,=torch.autograd.grad(ordinary_loss,actual)
    credit=LagOneHypergradient();credit.from_outputs(action_step=0,hyperparameters={'left':left,'right':right,'lr':lr},outputs={'weight':proposal})
    result=credit.consume({'weight':gradient},gradient_step=1)
    for name,value in zip(('left','right','lr'),expected):torch.testing.assert_close(result.gradients[name],value,rtol=1e-12,atol=1e-12)


def test_wrapper_composition_includes_learned_beta_and_postcommit_ema():
    theta=torch.tensor([1.,-2.],dtype=torch.float64);old_ema=torch.tensor([.5,.2],dtype=torch.float64)
    edit=torch.tensor([.1,-.3],dtype=torch.float64,requires_grad=True);beta=torch.tensor(.6,dtype=torch.float64,requires_grad=True);gamma=.9
    def wrapper(outputs):
        committed=outputs['weight'];ema=gamma*old_ema+(1-gamma)*committed
        return {'forward_weight':(1+beta)*committed-beta*ema}
    output={'weight':theta+edit};q=wrapper(output)['forward_weight'];loss=q.square().mean();expected=torch.autograd.grad(loss,(edit,beta),retain_graph=True)
    ordinary=q.detach().requires_grad_();gradient,=torch.autograd.grad(ordinary.square().mean(),ordinary)
    credit=LagOneHypergradient();credit.from_outputs(action_step=2,hyperparameters={'edit':edit,'beta':beta},outputs=output,compose_forward=wrapper)
    result=credit.consume({'forward_weight':gradient},gradient_step=3)
    torch.testing.assert_close(result.gradients['edit'],expected[0]);torch.testing.assert_close(result.gradients['beta'],expected[1])
    torch.testing.assert_close(result.gradients['edit'],(1+beta.detach()*gamma)*gradient)


def test_thin_callback_preserves_guard_derivative_and_detaches_task_graph():
    p=torch.tensor([.3,-.1],dtype=torch.float64,requires_grad=True);unused=torch.tensor(1.,requires_grad=True)
    anchor=torch.tensor([1.,2.,-1.],dtype=torch.float64);budget=.2
    kappa=budget/p.norm();delta=kappa*p[:,None]*anchor[None,:]
    g=torch.randn(2,3,dtype=torch.float64,requires_grad=True)
    expected,=torch.autograd.grad((delta*g.detach()).sum(),p,retain_graph=True)
    projected=g@anchor
    credit=LagOneHypergradient();credit.from_contraction(action_step=0,hyperparameters={'p':p,'unused':unused},
        contraction=lambda cot:kappa*(p*cot['projected']).sum())
    result=credit.consume({'projected':projected},gradient_step=1)
    torch.testing.assert_close(result.gradients['p'],expected)
    assert result.gradients['unused'] is None and g.grad is None and not result.gradients['p'].requires_grad


def test_clocks_parameter_identity_versions_and_single_pending_graph_are_guarded():
    p=torch.tensor(2.,requires_grad=True);c=LagOneHypergradient();c.from_outputs(action_step=1,hyperparameters={'p':p},outputs={'w':p*3})
    with pytest.raises(RuntimeError,match='one pending'):c.from_outputs(action_step=2,hyperparameters={'p':p},outputs={'w':p})
    with pytest.raises(ValueError,match='next ordinary'):c.consume({'w':torch.tensor(1.)},gradient_step=3)
    with torch.no_grad():p.add_(1)
    with pytest.raises(RuntimeError,match='p mutated'):c.consume({'w':torch.tensor(1.)},gradient_step=2)


def test_frozen_observations_are_copied_and_accidental_mutation_is_detected():
    live=torch.tensor([2.,4.]);snapshot=frozen_copy({'momentum':live});live.add_(100)
    torch.testing.assert_close(snapshot['momentum'],torch.tensor([2.,4.]))
    p=torch.tensor(1.,requires_grad=True);credit=LagOneHypergradient()
    credit.from_outputs(action_step=0,hyperparameters={'p':p},outputs={'w':p*snapshot['momentum']},guarded_tensors=snapshot)
    snapshot['momentum'].add_(1)
    with pytest.raises(RuntimeError,match='frozen observation'):credit.consume({'w':torch.ones(2)},gradient_step=1)


def test_frozen_soap_residual_with_frobenius_normalization_and_displacement_ema_wrapper():
    # Toy algebra of PR395's ordering, not a claim of a complete optimizer port.
    torch.manual_seed(18);dtype=torch.float64
    w=torch.randn(3,4,dtype=dtype);S=torch.randn_like(w)
    L0=torch.diag(torch.tensor([.6,.9,1.2],dtype=dtype));R0=torch.diag(torch.tensor([.7,.8,1.1,1.3],dtype=dtype))
    ql=torch.zeros_like(L0,requires_grad=True);qr=torch.zeros_like(R0,requires_grad=True)
    lr=torch.tensor(.02,dtype=dtype,requires_grad=True)
    direction=(L0+ql)@S@(R0+qr);direction=direction/direction.norm()*w.shape[0]**.5
    proposal=w-lr*direction
    old_buffer=torch.randn_like(w)*.01;gamma=.95;beta_next=.8
    def actual_wrapper(outputs):
        committed=outputs['master_weight']
        displacement=committed-w
        buffer=gamma*old_buffer+(1-gamma)*displacement
        return {'forward_weight':committed+beta_next*buffer}
    forward=actual_wrapper({'master_weight':proposal})['forward_weight']
    x=torch.randn(2,4,dtype=dtype);target=torch.randn(2,3,dtype=dtype)
    loss=(x@forward.T-target).square().mean()
    expected=torch.autograd.grad(loss,(ql,qr,lr),retain_graph=True)
    actual=forward.detach().requires_grad_();gradient,=torch.autograd.grad((x@actual.T-target).square().mean(),actual)
    credit=LagOneHypergradient();credit.from_outputs(action_step=31,hyperparameters={'Q_left':ql,'Q_right':qr,'lr':lr},outputs={'master_weight':proposal},compose_forward=actual_wrapper)
    result=credit.consume({'forward_weight':gradient},gradient_step=32)
    for name,value in zip(('Q_left','Q_right','lr'),expected):torch.testing.assert_close(result.gradients[name],value,rtol=1e-12,atol=1e-12)
    assert result.gradients['Q_left'].norm()>0 and result.gradients['Q_right'].norm()>0
    # This real displacement-EMA ordering differs from a weight-EMA lookahead.
    g_master,=torch.autograd.grad((forward*gradient).sum(),proposal)
    torch.testing.assert_close(g_master,(1+beta_next*(1-gamma))*gradient)


def test_assign_rejects_replaced_or_updated_hyperparameters_and_nonfinite_credit():
    p=torch.tensor(2.,requires_grad=True);credit=LagOneHypergradient()
    credit.from_outputs(action_step=0,hyperparameters={'p':p},outputs={'w':p.square()})
    result=credit.consume({'w':torch.tensor(1.)},gradient_step=1)
    with pytest.raises(ValueError,match='identity'):result.assign_gradients({'p':torch.tensor(2.,requires_grad=True)})
    with torch.no_grad():p.add_(1)
    with pytest.raises(RuntimeError,match='mutated'):result.assign_gradients({'p':p})
    q=torch.tensor(0.,requires_grad=True);credit.from_outputs(action_step=1,hyperparameters={'q':q},outputs={'w':q.sqrt()})
    with pytest.raises(FloatingPointError,match='hypergradient'):credit.consume({'w':torch.tensor(1.)},gradient_step=2)
    assert not credit.pending
