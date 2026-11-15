"""Default-off combined relative-gap tokenwise objective for exact teacher ties.

This is a logit-level API: one aggregate cotangent, no model execution or Adam
update. FP64 KL and ranking branches cast separately to FP32 before adding,
matching the reference's two student.double() branch boundaries. All 17 metric
fields are measured, including literal and strict tie-aware ranking counts.
Whole-model finite-update qualification is still required before adoption.
"""
import torch

from .rank_preservation import topk_preservation_flags
from .tiled_teacher_kl import tiled_teacher_kl
from .tiled_relative_ranking import tiled_relative_topk_margin_loss
from .token_preservation import token_preservation_terms


def logit_terms_and_cotangent(student, teacher_log_probs, *, config, multiplier,
                              panel_tokens, chunk_tokens=128, backward=True,
                              inputs=None, balance=False):
    from .tokenwise_reference import TOKEN_SUM_FIELDS
    if getattr(config,'ranking_loss_mode','absolute') != 'relative_gap':
        raise ValueError('tiled_relative_fp64_v1 requires explicit relative_gap mode')
    if config.ranking_tie_policy != 'exact_teacher_ties':
        raise ValueError('tiled_relative_fp64_v1 requires exact_teacher_ties')
    if (student.shape != teacher_log_probs.shape or student.ndim < 2
            or student.dtype != torch.float32 or teacher_log_probs.dtype != torch.float64
            or not student.numel()):
        raise ValueError('matching nonempty FP32 student/FP64 teacher arrays required')
    flat = student.detach().reshape(-1, student.shape[-1])
    teacher = teacher_log_probs.detach().reshape_as(flat).to(student.device)
    k = min(config.ranking_k, student.shape[-1])
    if panel_tokens < len(flat) or chunk_tokens < 1:
        raise ValueError('invalid full-panel normalization or chunk size')
    sums = torch.zeros(len(TOKEN_SUM_FIELDS), device=student.device, dtype=torch.float64)
    with torch.enable_grad() if backward or balance else torch.no_grad():
        s = flat.requires_grad_(backward or balance)
        kl = tiled_teacher_kl(s, teacher, reduction='none', teacher_is_log_probs=True)
        ranking = None
        teacher_topk = None
        if config.ranking_weight or balance:
            ranking, teacher_topk = tiled_relative_topk_margin_loss(
                teacher, s, k=k, margin_fraction=config.margin_fraction, return_teacher_topk=True)
        terms = token_preservation_terms(kl, multiplier=multiplier,
            mean_tolerance=config.mean_tolerance, tail_tolerance=config.tail_tolerance,
            scale=config.scale, rho=config.rho, ranking_loss=ranking,
            ranking_weight=config.ranking_weight)
        if balance:
            tail_g = torch.autograd.grad(terms['tail_quadratic'].sum()/panel_tokens, s, retain_graph=True)[0]
            rank_g = torch.autograd.grad(ranking.sum()/panel_tokens, s, retain_graph=backward)[0]
            sums[11] += tail_g.double().square().sum()
            sums[12] += rank_g.double().square().sum()
            del tail_g, rank_g
        gradient = torch.autograd.grad(terms['total'].sum()/panel_tokens, s)[0] if backward else None
    with torch.no_grad():
        if teacher_topk is None: teacher_topk = teacher.topk(k, dim=-1)
        student_topk = s.topk(k, dim=-1)
        flags = topk_preservation_flags(teacher, s, k=k,
            teacher_topk=teacher_topk, student_topk=student_topk)
        sums[3:6] = torch.stack([flags[name].sum() for name in
            ('literal_argmax_changed', 'literal_order_changed', 'literal_membership_changed')]).double()
        sums[13:17] = torch.stack([flags[name].sum() for name in
            ('tie_aware_argmax_changed', 'strict_order_changed',
             'strict_membership_changed', 'strict_topk_changed')]).double()
        labels = inputs.roll(-1, 1).flatten() if inputs is not None else None
        length = inputs.shape[1] if inputs is not None else None
        for start in range(0, len(flat), chunk_tokens):
            stop = min(start+chunk_tokens, len(flat))
            sums[:3] += torch.stack([terms[name][start:stop].sum() for name in
                                    ('dual_linear', 'tail_quadratic', 'ranking')])
            if inputs is not None:
                lp = s[start:stop].double().log_softmax(-1)
                t = teacher[start:stop]
                valid = torch.arange(start, stop, device=student.device) % length < length-1
                target = labels[start:stop, None].long()
                sums[6] += (-lp.gather(1, target).squeeze(1))[valid].sum()
                sums[7] += (-t.gather(1, target).squeeze(1))[valid].sum()
                gap = lp-t
                sums[8] += gap.square().sum()
                sums[9] += (gap-gap.mean(-1, keepdim=True)).square().sum()
                sums[10] += valid.sum()
    return gradient.reshape_as(student) if backward else None, sums.detach(), kl.detach().cpu()
