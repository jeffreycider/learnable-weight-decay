"""Optional continuous finite-L2 norm objective; no GPT or optimizer calls.

Cached FP32 probes and finite storage rounding make this a numerical approximation
of the stored-weight metric. Report actual stored-weight metrics separately.
"""
import math
import torch
from .grouped_norm import GroupedNormObjective, _RelativeL2


def validate_norm_auxiliary(settings):
    if settings is None or settings == {'mode': 'off'}:
        return None
    keys={'mode','coefficient','learning_rate','representation_lr_multiplier','gradient_clip'}
    if not isinstance(settings,dict) or set(settings)!=keys or settings['mode']!='alternating_continuous_l2':
        raise ValueError('explicit alternating_continuous_l2 norm settings required')
    for key in keys-{'mode'}:
        if isinstance(settings[key],bool) or not isinstance(settings[key],(int,float)) or not math.isfinite(settings[key]) or settings[key]<=0:
            raise ValueError('positive finite norm auxiliary '+key+' required')
    return dict(settings)


def continuous_group_norm_objective(prepared, grouped=None):
    """Equal fifteen-group fractional L2 change, with the global cap connected.

    References and cached probes are immutable post-base/pre-edit observations.
    General Gram terms avoid assuming exactly orthonormal stored anchors.
    Frozen zero-reference groups contribute zero to the SAME fifteen-group mean.
    """
    prepared._check_versions()
    grouped=grouped or GroupedNormObjective(prepared.parameters,zero_reference_policy='freeze')
    if len(grouped.groups)!=15 or any(grouped.parameters[n] is not p for n,p in prepared.parameters.items()):
        raise ValueError('norm references must belong to this immutable fifteen-group action')
    if tuple(grouped.zero_reference_groups)!=tuple(prepared.zero_reference_groups):
        raise ValueError('norm zero-group policy differs from prepared action')
    s=prepared.scale.double();changes={}
    for group,names in grouped.groups.items():
        ref=grouped.reference_squared[group];delta=ref.new_zeros(())
        if group in grouped.zero_reference_groups:
            changes[group]=s*0
            continue
        for name in names:
            if name in prepared.action.matrices:
                f=prepared.action.matrices[name]
                p,q=f.P.double(),f.Q.double();a,b=f.anchors.A.double(),f.anchors.B.double()
                wb,wta=(v.double() for v in prepared.snapshot.matrices[name].weight_probes)
                linear=(wb*p).sum()+(wta*q).sum()
                square=((p.T@p)*(b.T@b)).sum()+((q.T@q)*(a.T@a)).sum()+2*torch.trace((p.T@a)@(q.T@b))
            else:
                v=prepared.action.vectors[name].double();w=prepared.snapshot.vectors[name].weight.double()
                linear=(w*v).sum();square=v.square().sum()
            delta=delta+2*s*linear+s.square()*square
        changes[group]=_RelativeL2.apply(delta,ref)
    mean=torch.stack(list(changes.values())).mean()
    if not bool(torch.isfinite(mean.detach())):
        raise FloatingPointError('nonfinite continuous norm auxiliary')
    return mean,changes
