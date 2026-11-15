"""Optional representation/action-head learning-rate split; one Adam state."""
import math
import torch


def build_meta_adam(actor, *, lr, representation_lr_multiplier=1.):
    if any(not math.isfinite(v) or v<=0 for v in (lr,representation_lr_multiplier)):
        raise ValueError('positive finite meta learning rates required')
    if representation_lr_multiplier==1.:
        # Preserve the original parameter order and Adam grouping by default.
        return torch.optim.Adam(actor.parameters(),lr=lr,betas=(.9,.99),eps=1e-8)
    heads=[p for head in actor.output_heads for p in head.parameters()]
    head_ids={id(p) for p in heads}
    if len(heads)!=len(head_ids):raise ValueError('action heads share parameters')
    representation=[p for p in actor.parameters() if id(p) not in head_ids]
    if not heads or not representation:raise ValueError('both parameter families required')
    return torch.optim.Adam([
        dict(params=heads,lr=lr,lr_multiplier=1.,role='action_heads'),
        dict(params=representation,lr=lr*representation_lr_multiplier,
             lr_multiplier=representation_lr_multiplier,role='representation'),
    ],lr=lr,betas=(.9,.99),eps=1e-8)


def set_meta_learning_rate(optimizer, lr):
    if not math.isfinite(lr) or lr<=0:raise ValueError('positive finite learning rate required')
    for group in optimizer.param_groups:
        group['lr']=lr*group.get('lr_multiplier',1.)


def meta_learning_rates(optimizer):
    return {group.get('role','all'):group['lr'] for group in optimizer.param_groups}
