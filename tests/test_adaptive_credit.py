import pytest
import torch

from rewriter.adaptive_action import AdaptiveAction, AdaptiveFactors
from rewriter.adaptive_credit import AdaptiveCreditRecord, project_adaptive_gradients, adaptive_credit_contraction
from rewriter.lag_one import LagOneHypergradient


@pytest.mark.parametrize('saturated', [False, True])
@pytest.mark.parametrize('reduction', ['global_mean', 'global_token_sum'])
def test_thin_lag_one_equals_full_task_autograd_with_trainable_frames_and_guard(saturated,reduction):
    torch.manual_seed(813)
    left = torch.randn(5,2,dtype=torch.float64,requires_grad=True)
    right = torch.randn(7,2,dtype=torch.float64,requires_grad=True)
    core = torch.randn(2,2,dtype=torch.float64,requires_grad=True)
    bias = torch.randn(5,dtype=torch.float64,requires_grad=True)
    u = torch.linalg.qr(left,mode='reduced').Q
    v = torch.linalg.qr(right,mode='reduced').Q
    action = AdaptiveAction({'embed':AdaptiveFactors(u,core,v)}, {'bias':bias})
    scale = torch.minimum(core.new_ones(()), core.new_tensor(.1 if saturated else 100.)/action.norm_sq().sqrt())
    base = torch.randn(7,5,dtype=torch.float64)
    proposed = base + scale*action.matrices['embed'].dense().T
    output = proposed.sin() @ (scale*bias).cos()
    loss = output.square().mean()
    parameters = dict(left=left,right=right,core=core,bias=bias)
    expected = torch.autograd.grad(loss,tuple(parameters.values()),retain_graph=True)
    # Cotangents of the actual committed parameter representation.
    live_w = proposed.detach().requires_grad_()
    live_b = (scale*bias).detach().requires_grad_()
    g_w,g_b = torch.autograd.grad((live_w.sin()@live_b.cos()).square().mean(),(live_w,live_b))
    multiplier = 128 if reduction == 'global_token_sum' else 1
    gradients = {'embed':g_w*multiplier,'bias':g_b*multiplier}
    projected = project_adaptive_gradients(action,gradients,global_token_count=128,
                                           loss_reduction=reduction,transpose_storage=('embed',))
    before = projected['value'].clone()
    for gradient in gradients.values():gradient.zero_()  # ordinary buffers may now be reused
    torch.testing.assert_close(projected['value'],before,rtol=0,atol=0)
    lag = LagOneHypergradient()
    lag.from_contraction(action_step=6,hyperparameters=parameters,
                         contraction=lambda cotangents:adaptive_credit_contraction(action,scale,cotangents))
    result = lag.consume(projected,gradient_step=7)
    for (name,_),reference in zip(parameters.items(),expected):
        torch.testing.assert_close(result.gradients[name],reference,rtol=1e-11,atol=1e-11)
    torch.testing.assert_close(torch.tensor(result.contraction_value,dtype=scale.dtype),scale.detach()*before)


def test_projection_rejects_ambiguous_units_and_wrong_orientation():
    factor = AdaptiveFactors(torch.eye(3,2),torch.eye(2),torch.eye(4,2))
    action = AdaptiveAction({'weight':factor},{})
    with pytest.raises(ValueError,match='units'):
        project_adaptive_gradients(action,{'weight':torch.ones(3,4)},global_token_count=8,loss_reduction='ddp_average')
    with pytest.raises(ValueError,match='dense matrix'):
        project_adaptive_gradients(action,{'weight':torch.ones(4,3)},global_token_count=8,loss_reduction='global_mean')
    with pytest.raises(ValueError,match='must be tensors'):
        project_adaptive_gradients(action,{'weight':None},global_token_count=8,loss_reduction='global_mean')
    low = AdaptiveAction({'weight':AdaptiveFactors(factor.U.bfloat16(),factor.C.bfloat16(),factor.V.bfloat16())},{})
    with pytest.raises(ValueError,match='FP32 or FP64'):
        project_adaptive_gradients(low,{'weight':torch.ones(3,4)},global_token_count=8,loss_reduction='global_mean')


@pytest.mark.parametrize('factor_dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('gradient_dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('transpose',[False,True])
def test_reused_projections_match_independent_contractions_and_own_cotangents(factor_dtype,gradient_dtype,transpose):
    generator=torch.Generator().manual_seed(872)
    factor=AdaptiveFactors(*(torch.randn(*shape,generator=generator,dtype=factor_dtype)
                             for shape in ((13,3),(3,2),(17,2))))
    vector=torch.randn(13,generator=generator,dtype=factor_dtype)
    action=AdaptiveAction({'weight':factor},{'bias':vector})
    gradient=torch.randn(13,17,generator=generator,dtype=gradient_dtype)
    original={'weight':gradient.T if transpose else gradient,
              'bias':torch.randn(13,generator=generator,dtype=gradient_dtype)}
    before={n:g.clone() for n,g in original.items()}
    projected=project_adaptive_gradients(action,original,global_token_count=127,
        loss_reduction='global_token_sum',transpose_storage=('weight',) if transpose else ())
    dense=gradient.to(factor_dtype)/127
    reference=factor.cotangents(dense)
    tolerance=dict(rtol=3e-6,atol=2e-7) if factor_dtype==torch.float32 else dict(rtol=1e-12,atol=1e-13)
    for name in ('U','C','V'):
        torch.testing.assert_close(projected['matrices']['weight'][name],getattr(reference,name),**tolerance)
    expected=factor.contract(dense)+(vector*(before['bias'].to(factor_dtype)/127)).sum()
    torch.testing.assert_close(projected['value'],expected,**tolerance)
    assert all(torch.equal(g,before[n]) for n,g in original.items())
    saved={n:t.clone() for n,t in projected['matrices']['weight'].items()}
    bias=projected['vectors']['bias'].clone()
    for g in original.values():g.zero_()
    assert all(torch.equal(t,saved[n]) for n,t in projected['matrices']['weight'].items())
    assert torch.equal(projected['vectors']['bias'],bias)


def recorded():
    core = torch.eye(2,requires_grad=True)
    factor = AdaptiveFactors(torch.eye(3,2),core*2,torch.eye(4,2))
    action = AdaptiveAction({'weight':factor},{})
    observation = torch.ones(3)
    lag = LagOneHypergradient()
    rng = torch.get_rng_state().clone()
    record = AdaptiveCreditRecord(lag,action,core.norm().reciprocal(),action_step=0,
                                  hyperparameters={'core':core},guarded_observations=(observation,))
    assert torch.equal(rng,torch.get_rng_state())
    projected = record.project({'weight':torch.ones(3,4)},global_token_count=1,loss_reduction='global_mean')
    return lag,record,projected,observation


def test_record_rejects_projection_from_another_same_shape_action():
    first,_,_,_ = recorded()
    _,_,wrong,_ = recorded()
    with pytest.raises(ValueError,match='action identity'):
        first.consume(wrong,gradient_step=1)


@pytest.mark.parametrize('which',['factor','observation','scale'])
def test_record_guards_derived_action_and_observation_storage(which):
    lag,record,projected,observation = recorded()
    value = {'factor':record.action.matrices['weight'].U,
             'observation':observation,'scale':record.scale}[which]
    with torch.no_grad():value.add_(.01)
    with pytest.raises(RuntimeError,match='mutated'):
        lag.consume(projected,gradient_step=1)


def test_record_owns_immutable_action_inventories():
    from dataclasses import FrozenInstanceError
    core = torch.eye(2,requires_grad=True)
    factor = AdaptiveFactors(torch.eye(3,2),core,torch.eye(4,2))
    matrices = {'weight':factor}
    action = AdaptiveAction(matrices,{})
    lag = LagOneHypergradient()
    record = AdaptiveCreditRecord(lag,action,torch.tensor(1.),action_step=0,hyperparameters={'core':core})
    matrices['weight'] = factor.scaled(20.)
    assert record.action.matrices['weight'] is factor
    with pytest.raises(TypeError):record.action.matrices['weight'] = matrices['weight']
    with pytest.raises(FrozenInstanceError):record.action = action
    projected = record.project({'weight':torch.ones(3,4)},global_token_count=1,loss_reduction='global_mean')
    result = lag.consume(projected,gradient_step=1)
    torch.testing.assert_close(result.gradients['core'],torch.ones_like(core))
