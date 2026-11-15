"""Historical unintended extra-polar SOAP–Shampoo residual map.

Frozen SOAP/statistics feed dense dimensionless H_L/H_R. Only this optimizer
algebra is differentiated. This extra polar arose from our misdescription of
PR395 and does not implement the intended published-baseline comparison.
The corrected default is in soap_pr395_residual.py; keep this map for provenance.
"""
from dataclasses import dataclass
import math
import torch


def inverse_fourth_factor(gram, steps=8, damping=.01):
    """PR395 coupled polynomial, applied to identity; no eig/SVD derivative.

    Caller computes this frozen factor outside the hypergradient graph. Explicit
    factors change floating-point association versus upstream's direct application.
    Both treatment and Q=0 control use these same explicit factors.
    """
    P=.5*(gram+gram.mT)
    P=P+torch.diag_embed(P.diagonal().mean()*damping*torch.ones_like(P.diagonal()))
    tau=torch.linalg.matrix_norm(P,ord='fro',dim=(-2,-1),keepdim=True)+1e-20
    X=torch.eye(P.shape[-1],device=P.device,dtype=P.dtype)
    P=P/tau
    coeffs=[(3.85003,-10.8539,8.61893),(1.80992,-.587778,.0647852),
            (1.50394,-.594516,.121161),(45/32,-9/16,5/32)]
    coeffs=coeffs[:steps]+coeffs[-1:]*max(steps-len(coeffs),0)
    for a,b,c in coeffs:
        a,b,c=a/1.0001,b/1.0001**5,c/1.0001**9
        Q=torch.addmm(P,P,P,beta=b,alpha=c)
        X=torch.addmm(X,Q,X,beta=a)
        for i in range(4):
            P=torch.addmm(P,Q,P,beta=a) if i%2==0 else torch.addmm(P,P,Q,beta=a)
    return X*tau.pow(-.25)


def residual_scale(gram,damping=.01):
    # Units of inverse fourth root. 1e-20 is used ONLY for an exactly zero
    # mean Gram; positive means retain their true scale (no absolute floor).
    mean=gram.diagonal().mean()*(1+damping)
    if bool(mean<0): raise ValueError('Shampoo Gram must be positive semidefinite')
    return torch.where(mean==0,mean.new_tensor(1e-20),mean).pow(-.25)


def finite_polar(matrix,iterations=12,compute_dtype=torch.bfloat16):
    """Exact finite PR395 Muon polynomial, with its ordinary cast derivative.

    FP64 compute_dtype is available for separate numerical derivative tests.
    Production always uses BF16, exactly as the upstream Muon polar helper.
    """
    X=matrix.to(compute_dtype)
    transpose=X.shape[-2]>X.shape[-1]
    if transpose:X=X.mT
    stat=X if X.dtype==torch.float64 else X.float()
    gram=stat@stat.mT
    denominator=gram.norm(dim=(-2,-1),keepdim=True).clamp_min(1e-14).sqrt()
    X=X/denominator.to(X.dtype)
    for _ in range(iterations):
        A=X@X.mT
        B=-1.5*A+.5*(A@A)
        X=2*X+B@X
    return X.mT if transpose else X


@dataclass(frozen=True)
class FrozenMap:
    weight:torch.Tensor
    direction:torch.Tensor
    left:torch.Tensor
    right:torch.Tensor
    left_scale:torch.Tensor
    right_scale:torch.Tensor
    lr:float
    spectral_row:torch.Tensor|None=None
    spectral_col:torch.Tensor|None=None
    spectral_scale:torch.Tensor|None=None
    target_uw:float=.3825
    outward_scale:float=.5
    inward_scale:float=1.
    rowfloor:bool=True
    rowfloor_rho:float=1.
    cwd:float=.025
    polar_dtype:torch.dtype=torch.bfloat16


def committed_proposal(record,hl,hr):
    """Complete functional hidden-weight step, including every geometric stage."""
    r=record
    update=finite_polar((r.left+r.left_scale*hl)@r.direction@(r.right+r.right_scale*hr),
                        compute_dtype=r.polar_dtype)
    # Match PR395 Shampoo output magnitude BEFORE SPA/radial/rowfloor.
    update=update if update.dtype==torch.float64 else update.float()
    update=update*(math.sqrt(r.direction.shape[0])/update.norm().clamp_min(1e-20))
    if r.spectral_scale is not None:
        rotated=(r.spectral_row.mT@update@r.spectral_col)*r.spectral_scale
        shaped=r.spectral_row@rotated@r.spectral_col.mT
        update=shaped*(update.norm()/shaped.norm().clamp_min(1e-30))
    p=r.weight.float() if r.weight.dtype!=torch.float64 else r.weight
    coeff=(update*p).sum()/(p*p).sum().clamp_min(1e-12)
    radial=coeff*p
    scale=torch.where(coeff<0,coeff.new_tensor(r.outward_scale),coeff.new_tensor(r.inward_scale))
    update=update-radial+scale*radial
    pnorm=p.norm().clamp_min(1e-8)
    if r.rowfloor:
        factor=(r.target_uw*p.norm(dim=1,keepdim=True).clamp_min(1e-8)
                /update.norm(dim=1,keepdim=True).clamp_min(1e-8)).clamp_min(1).pow(r.rowfloor_rho)
        update=update*factor
    else:
        unorm=update.norm().clamp_min(1e-8)
        update=update*torch.where(unorm/pnorm<r.target_uw,r.target_uw*pnorm/unorm,torch.ones_like(pnorm))
    radius=(pnorm-r.lr*(update*p).sum()/pnorm).clamp_min(1e-8)
    # Preserve stored-weight casts, including cast before radius rescale.
    proposal=(r.weight-r.lr*update).to(r.weight.dtype)
    proposal_stats=proposal if proposal.dtype==torch.float64 else proposal.float()
    proposal=proposal*(radius/proposal_stats.norm().clamp_min(1e-8)).to(r.weight.dtype)
    if r.cwd>0:
        mask=(update*p>0).to(r.weight.dtype)
        proposal=proposal*(1-r.lr*r.cwd*mask)
    return proposal


def displacement_ema_forward(proposal,previous_weight,previous_buffer,*,gamma,beta):
    buffer=previous_buffer*gamma+(proposal-previous_weight)*(1-gamma)
    return proposal+beta*buffer
