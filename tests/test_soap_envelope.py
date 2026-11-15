import ast
from pathlib import Path
import pytest
import torch
from rewriter.soap_envelope import next_forward_envelope

DEFAULT=dict(lookahead_stepsize=.3,scheduled_lookahead=True,lr_ratio=.7,lookahead_ema=.99,
             prefill_steps=300,rest_steps=1890,tail_tau=100.,tail_start=2040,tail_end=2840,
             anchor_strength=.45,anchor_start=2100,anchor_ramp=100.)


def actual_wrapper_class():
    tree=ast.parse(Path('baseline/pr395/upstream.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EMA_Nesterov')
    scope={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),'<pinned EMA>','exec'),scope)
    return scope['EMA_Nesterov']


def literal_forward(proposal,old,buffer,ema,step,c):
    # Functional transcription of actual source order, retaining differentiability.
    new_buffer=buffer*c['lookahead_ema']+(proposal-old)*(1-c['lookahead_ema'])
    updates=c['tail_tau']>0 and c['tail_start']<=step+1<c['tail_end']
    new_ema=(proposal if ema is None else ema+(proposal-ema)/c['tail_tau']) if updates else ema
    q=proposal
    if step+2>c['prefill_steps'] and step+1<c['rest_steps']:
        beta=c['lookahead_stepsize']*(c['lr_ratio'] if c['scheduled_lookahead'] else 1.)
        q=q+beta*new_buffer
    if new_ema is not None:
        a=c['anchor_strength']*min(1.,max(0.,(step+1-c['anchor_start'])/c['anchor_ramp']))
        q=q+a*(q-new_ema)
    return q


@pytest.mark.parametrize('step',[0,298,299,300,1888,1889,2038,2039,2040,2099,2100,2199,2838,2839,2840,2898])
def test_full_forward_envelope_boundaries(step):
    c=DEFAULT;ema=torch.randn(3,4,dtype=torch.float64) if step>=2040 else None
    p=torch.randn(3,4,dtype=torch.float64,requires_grad=True);old=torch.randn_like(p);buffer=torch.randn_like(p)
    cotangent=torch.randn_like(p)
    q=literal_forward(p,old,buffer,ema,step,c)
    actual=torch.autograd.grad((q*cotangent).sum(),p)[0]
    envelope=next_forward_envelope(action_step=step,tail_exists=ema is not None,**c)
    torch.testing.assert_close(actual,cotangent*envelope.derivative,atol=1e-14,rtol=1e-14)


@pytest.mark.parametrize('tail_exists',[False,True])
def test_overlapping_lookahead_anchor_and_lazy_tail(tail_exists):
    c={**DEFAULT,'tail_start':0,'anchor_start':0,'rest_steps':3000,'prefill_steps':0,'anchor_ramp':1}
    p=torch.randn(2,3,dtype=torch.float64,requires_grad=True);old=torch.randn_like(p);buf=torch.randn_like(p)
    ema=torch.randn_like(p) if tail_exists else None
    q=literal_forward(p,old,buf,ema,5,c)
    e=next_forward_envelope(action_step=5,tail_exists=tail_exists,**c)
    torch.testing.assert_close(torch.autograd.grad(q.sum(),p)[0],torch.full_like(p,e.derivative))
    assert e.tail_ema_derivative==(1/c['tail_tau'] if tail_exists else 1.)


@pytest.mark.parametrize('step',[299,1889,2039,2100,2200,2839])
def test_actual_upstream_wrapper_finite_difference(step):
    cls=actual_wrapper_class();c=DEFAULT;old=torch.tensor([.5,-.7],dtype=torch.float64)
    proposal=torch.tensor([.6,-.6],dtype=torch.float64);oldbuf=torch.tensor([.03,-.04],dtype=torch.float64)
    ema=torch.tensor([.4,-.8],dtype=torch.float64) if step>=2040 else None
    def run(value):
        p=torch.nn.Parameter(old.clone())
        class Inner(torch.optim.Optimizer):
            def __init__(self):super().__init__([p],dict(lr=.7,initial_lr=1.))
            @torch.no_grad()
            def step(self):p.copy_(value)
        w=cls([p],Inner(),lookahead_stepsize=.3,lookahead_ema=.99,prefill_steps=300,rest_steps=1890)
        w.it=step;w.lookahead_status=True;w.state[p]['lookahead_buffer']=(oldbuf.clone(),step-1)
        w.step()
        newema=ema
        if c['tail_start']<=step+1<c['tail_end']:
            newema=p.detach().clone() if ema is None else ema+(p.detach()-ema)/c['tail_tau']
        w.nesterov_step();q=p.detach().clone()
        if newema is not None:
            a=c['anchor_strength']*min(1.,max(0.,(step+1-c['anchor_start'])/c['anchor_ramp']))
            q=q+a*(q-newema)
        return q
    e=next_forward_envelope(action_step=step,tail_exists=ema is not None,**c)
    fd=(run(proposal+1e-6)-run(proposal-1e-6))/2e-6
    torch.testing.assert_close(fd,torch.full_like(fd,e.derivative),atol=1e-9,rtol=1e-9)


def test_pending_vjp_through_full_map_and_late_anchor():
    from rewriter.soap_residual import FrozenMap,committed_proposal
    from rewriter.lag_one import LagOneHypergradient
    p=torch.randn(3,4,dtype=torch.float64);h=torch.zeros(3,3,dtype=torch.float64,requires_grad=True)
    k=torch.zeros(4,4,dtype=torch.float64,requires_grad=True)
    r=FrozenMap(p,torch.randn_like(p),torch.eye(3,dtype=torch.float64),torch.eye(4,dtype=torch.float64),
                torch.tensor(.2,dtype=torch.float64),torch.tensor(.3,dtype=torch.float64),.01,polar_dtype=torch.float64)
    old=torch.randn_like(p);buf=torch.randn_like(p);ema=torch.randn_like(p);inputs=torch.randn(4,2,dtype=torch.float64)
    output=committed_proposal(r,h,k);forward=literal_forward(output,old,buf,ema,2200,DEFAULT)
    loss=(forward@inputs).square().mean();expected=torch.autograd.grad(loss,(h,k))
    observed=forward.detach().requires_grad_();cotangent=torch.autograd.grad((observed@inputs).square().mean(),observed)[0]
    e=next_forward_envelope(action_step=2200,tail_exists=True,**DEFAULT)
    lag=LagOneHypergradient();lag.from_contraction(action_step=2200,hyperparameters={'h':h,'k':k},
        contraction=lambda g:(committed_proposal(r,h,k)*g['weight']).sum()*e.derivative)
    actual=lag.consume({'weight':cotangent},gradient_step=2201)
    for name,want in zip(['h','k'],expected):torch.testing.assert_close(actual.gradients[name],want,atol=1e-13,rtol=1e-12)
