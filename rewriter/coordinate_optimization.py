"""One-VJP joint multitask gradients; private updates are per checkpoint visit.

This is not a distributed private-state store or an episodic/meta-gradient trainer.
Shared gradients accumulate weighted state losses and can subsequently be SUMed.
Private gradients represent the unweighted state loss and must never be SUMed
across checkpoint identities. Optimizer clocks and clipping are separate.
"""
from __future__ import annotations
import math
import torch


def joint_state_backward(loss, *, state_weight, shared_parameters, private_parameters):
    """Consume one state graph, adding weighted shared and fresh unweighted private grads.

    Private grad slots must be empty: callers step/clear that state's private
    optimizer before reusing it. No optimizer is advanced here.
    """
    weight=float(state_weight)
    if not math.isfinite(weight) or weight<=0:
        raise ValueError('state weight must be finite and positive')
    shared=tuple(shared_parameters);private=tuple(private_parameters)
    if not shared or not private:
        raise ValueError('both shared and private parameter sets required')
    ids=[id(p) for p in (*shared,*private)]
    if len(set(ids))!=len(ids):raise ValueError('parameter ownership overlaps or repeats')
    if any(p.grad is not None for p in private):
        raise ValueError('clear private gradients before each checkpoint visit')
    gradients=torch.autograd.grad(loss*weight,(*shared,*private),allow_unused=True)
    for parameter,gradient in zip(shared,gradients[:len(shared)]):
        if gradient is not None:
            if parameter.grad is None:parameter.grad=gradient.detach()
            else:parameter.grad.add_(gradient.detach())
    for parameter,gradient in zip(private,gradients[len(shared):]):
        if gradient is not None:parameter.grad=gradient.detach().div_(weight)


def clip_private_gradients(private_parameters,max_norm):
    """Separate local clip, before the local optimizer tick; never shares a norm with phi."""
    if not math.isfinite(float(max_norm)) or max_norm<=0:raise ValueError('positive finite clip required')
    return torch.nn.utils.clip_grad_norm_(tuple(private_parameters),max_norm,error_if_nonfinite=True)


def configure_shared_continuation(optimizer,mode):
    """Explicit metadata-only optimizer transition; moments and clocks stay intact."""
    if mode not in ('original_joint','separate_representation'):raise ValueError('unknown shared continuation mode')
    roles={g.get('role'):g for g in optimizer.param_groups}
    if set(roles)!={'action_heads','representation'}:raise ValueError('named head/representation Adam groups required')
    if mode=='separate_representation':
        roles['representation']['lr']=1e-4
        roles['representation']['lr_multiplier']=1e-4/roles['action_heads']['lr']
        roles['representation']['eps']=1e-12
    return {role:{k:group[k] for k in ('lr','eps','betas')} for role,group in roles.items()}


def clip_shared_continuation(optimizer,mode,max_norm=1.):
    if mode=='original_joint':
        parameters=[p for g in optimizer.param_groups for p in g['params']]
        return dict(joint=float(torch.nn.utils.clip_grad_norm_(parameters,max_norm,error_if_nonfinite=True)))
    if mode!='separate_representation':raise ValueError('unknown shared continuation mode')
    return {group['role']:float(torch.nn.utils.clip_grad_norm_(group['params'],max_norm,error_if_nonfinite=True)) for group in optimizer.param_groups}


def effective_bfloat16_motion(before,module):
    """Parameter values actually cast by the CUDA residual backbone, not FP32 drift.

    Feature lifts and output heads stay FP32 and are deliberately excluded.
    This measures cast parameter changes, not a claim about activation changes.
    """
    from baseline.model import Linear,RMSNorm
    squared=initial_squared=master_squared=0.;changed=elements=0;details={}
    for prefix,child in module.named_modules():
        if not isinstance(child,(Linear,RMSNorm)):continue
        for leaf,p in child.named_parameters(recurse=False):
            name=prefix+'.'+leaf if prefix else leaf
            old=before[name].detach().cpu();new=p.detach().cpu()
            a=old.bfloat16().float().double();b=new.bfloat16().float().double();delta=b-a
            n=p.numel();c=int((a!=b).sum());ss=float(delta.square().sum())
            details[name]=dict(elements=n,changed_elements=c,changed_fraction=c/n,cast_change_rms=(ss/n)**.5)
            elements+=n;changed+=c;squared+=ss;initial_squared+=float(a.square().sum());master_squared+=float((new.double()-old.double()).square().sum())
    if not elements:raise ValueError('no cast residual parameters found')
    return dict(elements=elements,changed_elements=changed,changed_fraction=changed/elements,cast_change_rms=(squared/elements)**.5,
                cast_relative_l2=(squared/initial_squared)**.5 if initial_squared else None,master_change_rms=(master_squared/elements)**.5,by_parameter=details)
