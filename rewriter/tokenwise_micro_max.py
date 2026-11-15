"""Experimental mean-of-microbatch worst-relative-rank objective.

Bounded detached scans choose the first worst context; ranking autograd is then
recomputed only for that row. The caller still performs one model backward.
KL arithmetic/metrics use the existing qualified tiled-relative implementation.
"""
from dataclasses import replace
import torch
from .relative_rank_tail import maximum_relative_rank_loss
from .tokenwise_tiled_relative import logit_terms_and_cotangent as mean_terms


def worst_context_loss(student,teacher,*,k,margin_fraction,chunk_tokens=128,backward=True):
    flat=student.detach().reshape(-1,student.shape[-1]);target=teacher.detach().reshape_as(flat)
    if chunk_tokens<1 or not len(flat):raise ValueError('positive scan chunk and nonempty contexts required')
    with torch.no_grad():
        losses=torch.cat([maximum_relative_rank_loss(target[a:a+chunk_tokens],flat[a:a+chunk_tokens],k=k,margin_fraction=margin_fraction) for a in range(0,len(flat),chunk_tokens)])
        # torch.argmax selects the first exact maximum, including equal zeroes.
        index=losses.argmax().reshape(1)
    if backward:
        with torch.enable_grad():
            selected=flat.index_select(0,index).detach().requires_grad_()
            loss=maximum_relative_rank_loss(target.index_select(0,index),selected,k=k,margin_fraction=margin_fraction).sum()
        return loss,selected,index
    return losses[index].sum(),None,index


def logit_terms_and_cotangent(student,teacher_log_probs,*,config,multiplier,panel_tokens,
                              chunk_tokens=128,backward=True,inputs=None,balance=False,diagnostics=None):
    if config.ranking_loss_mode!='relative_gap_micro_max' or config.loss_backend!='reference_micro_max_v1' or config.ranking_tie_policy!='exact_teacher_ties':
        raise ValueError('explicit exact-tie micro-max objective/backend required')
    if balance:raise ValueError('micro-max basis calibration is not implemented')
    n=student.numel()//student.shape[-1]
    if not n or panel_tokens<n or panel_tokens%n:raise ValueError('equal complete microbatches required for mean_micro normalization')
    base=replace(config,ranking_loss_mode='relative_gap',loss_backend='tiled_relative_fp64_v1',ranking_weight=0.)
    gradient,sums,kl=mean_terms(student,teacher_log_probs,config=base,multiplier=multiplier,panel_tokens=panel_tokens,chunk_tokens=chunk_tokens,backward=backward,inputs=inputs,balance=False)
    loss,selected,index=worst_context_loss(student,teacher_log_probs,k=min(config.ranking_k,student.shape[-1]),margin_fraction=config.margin_fraction,chunk_tokens=chunk_tokens,backward=backward)
    if backward:
        rank_gradient=torch.autograd.grad(loss*(config.ranking_weight*n/panel_tokens),selected)[0]
        gradient.reshape(-1,student.shape[-1]).index_add_(0,index,rank_gradient)
    # Existing panel aggregators divide additive metrics by full panel tokens.
    sums[2]=loss.detach()*config.ranking_weight*n
    if diagnostics is not None:
        diagnostics.update(unweighted_max_relative_rank=float(loss.detach()),selected_context=int(index.item()),micro_contexts=n,normalization='mean_micro',micro_weight=n/panel_tokens)
    return gradient,sums,kl
