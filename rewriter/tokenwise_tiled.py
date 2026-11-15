"""Experimental drop-in logit-level tokenwise objective; never enabled implicitly.

This accepts the exact resident FP64 teacher log-probabilities used by the
reference, without casting targets. It replaces loss algebra only; the caller
still invokes one GPT backward and owns checkpoint scaling/Adam/dual state.
The FP64 KL/rank cotangents are individually cast to FP32 before their sum,
matching the dense reference's separate s.double() branch cast boundaries.
Reduction arithmetic can still differ. Whole-update qualification is
required; scalar loss agreement alone does not certify this experimental path.
"""
import torch
from .tiled_teacher_kl import tiled_teacher_kl
from .tiled_ranking_loss import tiled_topk_margin_loss
from .token_preservation import token_preservation_terms


def logit_terms_and_cotangent(student,teacher_log_probs,*,config,multiplier,panel_tokens,
                              chunk_tokens=128,backward=True,inputs=None,balance=False):
    from .tokenwise_reference import TOKEN_SUM_FIELDS
    if getattr(config,'ranking_k',10)!=10 or getattr(config,'ranking_tie_policy','legacy')!='legacy':
        raise ValueError('tiled_fp64_v1 is qualified only for legacy top10 ranking')
    if student.shape!=teacher_log_probs.shape or student.dtype!=torch.float32 or teacher_log_probs.dtype!=torch.float64:
        raise ValueError('matching FP32 student/FP64 teacher arrays required')
    flat=student.detach().reshape(-1,student.shape[-1]);teacher=teacher_log_probs.detach().reshape_as(flat).to(student.device)
    if panel_tokens<len(flat) or chunk_tokens<1 or student.shape[-1]<=10:raise ValueError('invalid token normalization or vocabulary')
    sums=torch.zeros(len(TOKEN_SUM_FIELDS),device=student.device,dtype=torch.float64)
    with torch.enable_grad() if backward or balance else torch.no_grad():
        s=flat.requires_grad_(backward or balance)
        kl=tiled_teacher_kl(s,teacher,reduction='none',teacher_is_log_probs=True)
        ranking=None;teacher_topk=None
        if config.ranking_weight or balance:
            ranking,teacher_topk=tiled_topk_margin_loss(teacher,s,k=10,margin_fraction=config.margin_fraction,arithmetic='fp64',return_teacher_topk=True)
        terms=token_preservation_terms(kl,multiplier=multiplier,mean_tolerance=config.mean_tolerance,
            tail_tolerance=config.tail_tolerance,scale=config.scale,rho=config.rho,
            ranking_loss=ranking,ranking_weight=config.ranking_weight)
        if balance:
            tail_g=torch.autograd.grad(terms['tail_quadratic'].sum()/panel_tokens,s,retain_graph=True)[0]
            rank_g=torch.autograd.grad(ranking.sum()/panel_tokens,s,retain_graph=backward)[0]
            sums[11]+=tail_g.double().square().sum();sums[12]+=rank_g.double().square().sum()
            del tail_g,rank_g
        gradient=torch.autograd.grad(terms['total'].sum()/panel_tokens,s)[0] if backward else None
    with torch.no_grad():
        ti=teacher.topk(10,dim=-1).indices if teacher_topk is None else teacher_topk[1]
        si=s.topk(10,dim=-1).indices
        sums[3:6]=torch.stack(((teacher.argmax(-1)!=s.argmax(-1)).sum(),(ti!=si).any(-1).sum(),
            (ti.sort(-1).values!=si.sort(-1).values).any(-1).sum())).double()
        # Retain the reference's chunk grouping for additive diagnostics.
        labels=inputs.roll(-1,1).flatten() if inputs is not None else None
        length=inputs.shape[1] if inputs is not None else None
        for start in range(0,len(flat),chunk_tokens):
            stop=min(start+chunk_tokens,len(flat))
            sums[:3]+=torch.stack([terms[k][start:stop].sum() for k in ('dual_linear','tail_quadratic','ranking')])
            if inputs is not None:
                lp=s[start:stop].double().log_softmax(-1);t=teacher[start:stop]
                valid=torch.arange(start,stop,device=student.device)%length<length-1;target=labels[start:stop,None].long()
                sums[6]+=(-lp.gather(1,target).squeeze(1))[valid].sum();sums[7]+=(-t.gather(1,target).squeeze(1))[valid].sum()
                gap=lp-t;sums[8]+=gap.square().sum();sums[9]+=(gap-gap.mean(-1,keepdim=True)).square().sum();sums[10]+=valid.sum()
    return gradient.reshape_as(student) if backward else None,sums.detach(),kl.detach().cpu()
