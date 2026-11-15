import ast
import copy
import dataclasses
import math
from pathlib import Path
import pytest
import torch
from rewriter.soap46_map import BasisSnapshot,committed_weight,soap_direction,next_forward_coefficient
from rewriter.lag_one import LagOneHypergradient

torch.set_num_threads(2)


def reference_scope():
    from scripts.train_soap46 import ACTIVE_ENV
    tree=ast.parse(Path('baseline/record46/upstream.py').read_text())
    # Keep upstream helper bodies verbatim; all inactive switches are explicitly off.
    names={'gram_frobenius_norm_estimate','_ns_inner','zeropower_via_newtonschulz5',
        'tensor_cosine','trust_gate','early_trust_floor_for_step','bounded_trust_gate','norm_preserving_blend',
        'scale_radial_update','target_radius_after_update','rescale_to_radius','soap_precondition_momentum',
        'soap_update_preconditioner','soap_eigenbasis','soap_basis_qr','late_damp_frac','gram_beta_at','muon_update'}
    scope=dict(torch=torch,Tensor=torch.Tensor,SOAP_BETA2=.9,SOAP_DENOM_POWER=.5,SOAP_PRECONDITION_FREQUENCY=1,
        SOAP_KL=0,SHAMPOO_MODE=0,LATE_DAMP=0,LATE_DAMP_START=2000,LATE_DAMP_END=2900,
        GRAM_BETA2=-1, RADIAL_OUTWARD_SCALE=.5,RADIAL_INWARD_SCALE=1.,
        ATTN_TRUST_MIN_AGREE=.2,ATTN_TRUST_MIN_GRAD_ALIGN=0.,ATTN_TRUST_POWER=1.,
        ATTN_EARLY_TRUST_FLOOR=.45,ATTN_EARLY_TRUST_CAP=.85,ATTN_TRUST_FLOOR_END_STEP=1375,ATTN_TRUST_FLOOR_FADE_END_STEP=1625)
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),'<record46 helpers>','exec'),scope)
    return scope


def make_record(shape=(4,7),attention=True,output=True,step=43):
    torch.manual_seed(345)
    n,m=shape;d=torch.randn(n,m);ql=torch.linalg.qr(torch.randn(n,n))[0];qr=torch.linalg.qr(torch.randn(m,m))[0]
    return BasisSnapshot(torch.randn(n,m),d,torch.randn(n,m),ql,qr,torch.rand(n,m)+.2,.025,step,attention,output,'inner-outer')


def reference_proposal(r):
    s=reference_scope();state=dict(q_row=r.q_left,q_col=r.q_right,exp_avg_sq=r.variance.clone(),soap_step=1)
    # r.variance is already post-update. Reverse the known baseline update to
    # feed the untouched helper, allowing this test to include its V update.
    projection=r.q_left.T@r.direction@r.q_right
    state['exp_avg_sq']=(r.variance-.1*projection.square())/.9
    # Instead use helper's computed V as the record: eliminates inverse-rounding.
    pre=s['soap_precondition_momentum'](r.direction,state,step=r.step)
    r=dataclasses.replace(r,variance=state['exp_avg_sq'].clone())
    if r.attention:
        gate=s['bounded_trust_gate'](s['trust_gate'](r.direction,pre,r.gradient),r.step) if r.attention_output else torch.ones(())
        pre=s['norm_preserving_blend'](r.direction,pre,gate)
    u=s['muon_update'](pre);u=s['scale_radial_update'](u,r.weight)
    f=(.3825*r.weight.float().norm(dim=1,keepdim=True).clamp_min(1e-8)/u.float().norm(dim=1,keepdim=True).clamp_min(1e-8)).clamp(min=1.).pow(1.)
    u=(u.float()*f).to(u.dtype);radius=s['target_radius_after_update'](r.weight,u,r.lr)
    mask=(u.float()*r.weight.float()>0).to(r.weight.dtype)
    p=r.weight.clone();p.add_(u,alpha=-r.lr);s['rescale_to_radius'](p,radius);p.mul_(1-(r.lr*.025)*mask)
    return r,p


@pytest.mark.parametrize('variant',['inner-outer','outer-only'])
@pytest.mark.parametrize('shape',[(4,7),(7,4),(8,8)])
@pytest.mark.parametrize('attention,output,step',[(False,False,1),(True,False,1),(True,True,43),(True,True,1500),(True,True,1700)])
def test_zero_residual_bit_exact_upstream(variant,shape,attention,output,step):
    r=dataclasses.replace(make_record(shape,attention,output,step),variant=variant)
    r,expected=reference_proposal(r)
    actual=committed_weight(r,torch.zeros_like(r.q_left),torch.zeros_like(r.q_right))
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def double_record(variant):
    r=make_record();r=dataclasses.replace(r,variant=variant,polar_dtype=torch.float64)
    return dataclasses.replace(r,**{f.name:getattr(r,f.name).double() for f in dataclasses.fields(r) if isinstance(getattr(r,f.name),torch.Tensor)})


@pytest.mark.parametrize('variant',['inner-outer','outer-only'])
def test_nonzero_finite_difference_through_norm_polar_and_complete_weight(variant):
    r=double_record(variant);xl=(torch.randn_like(r.q_left)*.01).requires_grad_();xr=(torch.randn_like(r.q_right)*.01).requires_grad_()
    assert torch.autograd.gradcheck(lambda l,h:committed_weight(r,l,h),(xl,xr),eps=1e-6,atol=2e-5,rtol=2e-4)


@pytest.mark.parametrize('variant',['inner-outer','outer-only'])
def test_lag_one_full_next_loss_vjp_with_wrapper(variant):
    r=double_record(variant);xl=torch.zeros_like(r.q_left,requires_grad=True);xr=torch.zeros_like(r.q_right,requires_grad=True)
    inputs=torch.randn(7,5,dtype=torch.float64);target=torch.randn(4,5,dtype=torch.float64)
    previous=r.weight.clone();buffer=torch.randn_like(previous);beta=.21
    def forward():
        p=committed_weight(r,xl,xr);b=.99*buffer+.01*(p-previous)
        return p+beta*b
    q=forward();expected=torch.autograd.grad(((q@inputs-target)**2).mean(),(xl,xr))
    leaf=q.detach().requires_grad_();g=torch.autograd.grad(((leaf@inputs-target)**2).mean(),leaf)[0]
    lag=LagOneHypergradient();lag.from_contraction(action_step=302,hyperparameters={'xl':xl,'xr':xr},
        contraction=lambda cot:(committed_weight(r,xl,xr)*cot['weight']).sum()*(1+beta*.01))
    actual=lag.consume({'weight':g},gradient_step=303)
    for name,want in zip(('xl','xr'),expected):torch.testing.assert_close(actual.gradients[name],want)
    assert all(v.norm()>0 for v in expected)


def test_exact_zero_direction_has_finite_derivatives():
    r=double_record('inner-outer');r=dataclasses.replace(r,direction=torch.zeros_like(r.direction))
    xl=torch.zeros_like(r.q_left,requires_grad=True);xr=torch.zeros_like(r.q_right,requires_grad=True)
    g=torch.autograd.grad(committed_weight(r,xl,xr).sum(),(xl,xr))
    assert all(torch.isfinite(x).all() and x.norm()==0 for x in g)


@pytest.mark.parametrize('step,expected',[(298,1.),(299,1.003),(1948,1.003),(1949,1.)])
def test_wrapper_transition_cotangents(step,expected):
    import types
    wrapper=types.SimpleNamespace(it=step,prefill_steps=300,rest_steps=1950,lookahead_stepsize=.3,lookahead_ema=.99,get_lr_lambda=lambda:1.)
    assert next_forward_coefficient(wrapper)==pytest.approx(expected)
