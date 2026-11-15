from dataclasses import replace
import pytest
import torch
from rewriter.soap46_conditioning import parameter_frame_weight
from rewriter.soap46_map import BasisSnapshot, committed_weight
from rewriter.lag_one import LagOneHypergradient


def record(dtype=torch.float64):
    torch.set_num_threads(1);g=torch.Generator().manual_seed(713)
    draw=lambda *shape:torch.randn(*shape,generator=g,dtype=dtype)
    ql=torch.linalg.qr(draw(4,4)).Q;qr=torch.linalg.qr(draw(3,3)).Q
    return BasisSnapshot(draw(4,3),draw(4,3),draw(4,3),ql,qr,draw(4,3).square()+.4,
        .03,1500,True,True,'outer-only',torch.float64 if dtype==torch.float64 else torch.bfloat16)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_zero_residual_is_bitwise_original_and_general_action_family_is_identical(dtype):
    r=record(dtype);zl=torch.zeros_like(r.q_left);zr=torch.zeros_like(r.q_right)
    assert torch.equal(parameter_frame_weight(r,zl,zr),committed_weight(r,zl,zr))
    zl=zl+.01;zr=zr-.02
    assert torch.equal(parameter_frame_weight(r,zl,zr),committed_weight(r,zl@r.q_left,zr@r.q_right))
    with pytest.raises(ValueError,match='outer-only'):parameter_frame_weight(replace(r,variant='inner-outer'),zl,zr)


def test_physical_residual_preserves_meaning_under_consistent_basis_permutations():
    r=record();zl=torch.arange(16,dtype=torch.float64).reshape(4,4)*.001;zr=torch.arange(9,dtype=torch.float64).reshape(3,3)*.002
    left=[2,0,3,1];right=[1,2,0]
    rotated=replace(r,q_left=r.q_left[:,left],q_right=r.q_right[:,right],variance=r.variance[left][:,right])
    expected=parameter_frame_weight(r,zl,zr)
    torch.testing.assert_close(parameter_frame_weight(rotated,zl,zr),expected,rtol=1e-11,atol=1e-12)
    # Keeping additive basis-column X unchanged does not perform this transport.
    fixed_xl=zl@r.q_left;fixed_xr=zr@r.q_right
    assert (committed_weight(rotated,fixed_xl,fixed_xr)-expected).norm()>1e-5


def test_full_map_parameter_frame_lag_vjp_and_finite_difference():
    r=record();zl=torch.nn.Parameter(torch.full_like(r.q_left,.001));zr=torch.nn.Parameter(torch.full_like(r.q_right,-.002))
    parameters={'left':zl,'right':zr};coefficient=1.17
    def full(zl,zr):return parameter_frame_weight(r,zl,zr)*coefficient+.1*r.weight
    inputs=torch.tensor([[.2,-.4,.3],[.1,.5,-.1]],dtype=torch.float64)
    weight=full(zl,zr);loss=(inputs@weight.T).square().mean()
    expected=torch.autograd.grad(loss,tuple(parameters.values()),retain_graph=True)
    ordinary=torch.autograd.grad(loss,weight)[0].detach()
    lag=LagOneHypergradient()
    lag.from_contraction(action_step=1499,hyperparameters=parameters,
        contraction=lambda cotangent:(parameter_frame_weight(r,zl,zr)*cotangent['weight']).sum()*coefficient,
        guarded_tensors=(r.weight,r.direction,r.gradient,r.q_left,r.q_right,r.variance))
    actual=lag.consume({'weight':ordinary},gradient_step=1500)
    for name,want in zip(parameters,expected):torch.testing.assert_close(actual.gradients[name],want,rtol=1e-11,atol=1e-12)
    def value(z):return (inputs@full(z,zr).T).square().mean().detach()
    direction=torch.arange(16,dtype=torch.float64).reshape(4,4)/16
    eps=1e-6;finite=(value(zl+eps*direction)-value(zl-eps*direction))/(2*eps)
    torch.testing.assert_close(finite,(expected[0]*direction).sum(),rtol=1e-5,atol=1e-8)
