"""Accepted record #46: additive SOAP basis residuals, followed by Muon.

V and Q are immutable snapshots of baseline SOAP state; neither is differentiated.
The square root/division by V is ELEMENTWISE. Residuals persist as additive
matrices in the stored Q coordinate convention; there is no learned state transport.
"""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class BasisSnapshot:
    weight: torch.Tensor
    direction: torch.Tensor
    gradient: torch.Tensor
    q_left: torch.Tensor | None
    q_right: torch.Tensor | None
    variance: torch.Tensor | None
    lr: float
    step: int
    attention: bool
    attention_output: bool
    variant: str
    # FP64 mode is exclusively for smooth standalone finite-difference tests.
    polar_dtype: torch.dtype = torch.bfloat16
    cwd: float = .025


def stats(x):
    return x if x.dtype == torch.float64 else x.float()


def nu(x, eps=1e-8, keepdim=False):
    x = stats(x)
    gram = x.mT @ x if x.shape[-2] > x.shape[-1] else x @ x.mT
    magnitude=gram.norm(dim=(-2,-1),keepdim=keepdim)
    # Same forward floor as sqrt().clamp_min(eps), with a defined zero derivative
    # below the floor (avoid 0*infinity in sqrt backward at an exactly zero Gram).
    active=magnitude>eps*eps
    root=torch.where(active,magnitude,torch.ones_like(magnitude)).sqrt()
    return torch.where(active,root,root.new_tensor(eps))


def cosine(a,b):
    a,b=stats(a),stats(b)
    return (a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-8)


def soap_direction(record,x_left,x_right):
    r=record;d=r.direction
    if r.variant not in ('inner-outer','outer-only'):raise ValueError('unknown basis residual variant')
    if r.q_left is None:return d
    left,right=r.q_left+x_left,r.q_right+x_right
    projected=(left.mT@stats(d)@right if r.variant=='inner-outer'
               else r.q_left.mT@stats(d)@r.q_right)
    pre=left@(projected/r.variance.clamp_min(1e-16).pow(.5))@right.mT
    pre=(pre*(nu(stats(d))/nu(pre))).to(d.dtype)
    if r.attention:
        if r.attention_output:
            agree=((cosine(pre,d)-.2)/.8).clamp(0,1)
            align=(cosine(pre,r.gradient)/cosine(d,r.gradient).clamp_min(1e-8)).clamp(0,1)
            if isinstance(r.step,torch.Tensor):
                floor=.45*((1625-r.step)/250).clamp(0,1)
                cap=torch.where(r.step<1625,r.step.new_tensor(.85),r.step.new_tensor(1.))
            else:
                floor=.45 if r.step<1375 else (0. if r.step>=1625 else .45*(1625-r.step)/250)
                cap=.85 if r.step<1625 else 1.
            gate=(agree*align).clamp(0,1).clamp(min=floor,max=cap)
        else:gate=d.new_ones(())
        blended=d+(pre-d)*gate.to(d.dtype)
        pre=(blended*(nu(d)/nu(blended)).to(blended.dtype)).to(d.dtype)
    return pre


def finite_muon(direction,compute_dtype=torch.bfloat16):
    x=direction.to(compute_dtype);transpose=direction.shape[-2]>direction.shape[-1]
    if transpose:x=x.mT
    x=x/nu(x,keepdim=True,eps=1e-7).to(x.dtype)
    for _ in range(12):
        a=x@x.mT
        # Match source association: (0.5*A)@A, not 0.5*(A@A).
        b=-1.5*a+(.5*a)@a
        x=2*x+b@x
    if transpose:x=x.mT
    return x*max(1,direction.shape[-2]/direction.shape[-1])**.5


def postprocess(record,update):
    p=record.weight;pf=stats(p);uf=stats(update)
    coeff=(uf*pf).sum()/(pf*pf).sum().clamp_min(1e-12)
    radial=coeff*pf
    braked=(uf-radial+torch.where(coeff<0,uf.new_tensor(.5),uf.new_tensor(1.))*radial).to(update.dtype)
    rrow=pf.norm(dim=1,keepdim=True).clamp_min(1e-8)
    srow=stats(braked).norm(dim=1,keepdim=True).clamp_min(1e-8)
    update=(stats(braked)*(.3825*rrow/srow).clamp_min(1.)).to(braked.dtype)
    uf=stats(update);pn=pf.norm().clamp_min(1e-8)
    radius=(pn+(-record.lr*(uf*pf).sum()/pn)).clamp_min(1e-8)
    mask=(uf*pf>0).to(p.dtype)
    # Tensor LR supports one compiled graph across changing schedules. Its
    # multiply/add association is separately qualified against the eager map.
    proposal=((p-stats(update)*record.lr) if isinstance(record.lr,torch.Tensor)
              else torch.add(p,update,alpha=-record.lr)).to(p.dtype)
    proposal=proposal*(radius/stats(proposal).norm().clamp_min(1e-8)).to(p.dtype)
    return proposal*(1-(record.lr*record.cwd)*mask)


def committed_weight(record,x_left,x_right):
    return postprocess(record,finite_muon(soap_direction(record,x_left,x_right),record.polar_dtype))


def next_forward_coefficient(wrapper):
    """Derivative of next ordinary forward w.r.t. this committed hidden weight.

    Record46 tail EMA is evaluation-only. Upcoming displacement lookahead uses
    the current LR (next set_hparams runs after the next backward).
    """
    upcoming=wrapper.it+1
    active=upcoming+1>wrapper.prefill_steps and upcoming<wrapper.rest_steps
    beta=(wrapper.lookahead_stepsize*wrapper.get_lr_lambda() if active else 0.)
    return 1+beta*(1-wrapper.lookahead_ema)
