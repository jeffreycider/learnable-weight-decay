"""Frozen-policy preservation confidence audit: per-sequence and per-token tails.

Read-only measurement of one saved Chisel policy. No gradient reaches the policy,
no multiplier is updated, and nothing here selects a checkpoint. A checkpoint
panel mean hides the spread across sequences and tokens. In the corrected fit,
paired seen (128 sequences) and withheld (64 sequences) panels of the same
checkpoint were OBSERVED to differ by about one to six percent; that is a
measured disagreement, not an accuracy bound. This module therefore reports
the distribution across sequences and tokens together with bootstrap intervals
of the means, whose sampling assumptions are stated with them.

Population names are explicit and must be carried into every table:

* ``fitting_seen``: fitting checkpoint, cached train-role panel; these
  sequences produced fitting gradients.
* ``fitting_withheld``: same fitting checkpoint, cached audit-role panel of
  globally held-out sequence IDs, never fitted at any checkpoint.
* ``checkpoint_heldout_development``: checkpoint-held-out GPT state, cached
  audit-role panel. The trainer's LR halving and stopping consume this panel's
  feasibility score, so it is development validation, not an untouched test.
* ``fresh_next_batch``: the complete ordinary training batch immediately after
  the FINAL bank checkpoint. Fitting panels of any later checkpoint would sample
  from an earlier checkpoint's next batch, so only the final checkpoint's next
  batch is unseen by every fitting panel. The hash role is recorded per sequence.

Bootstrap intervals describe resampling uncertainty of the audited panels only.
They are not a certificate about unaudited inputs. The audited policy is used
exactly as saved; no scaled or otherwise modified action is evaluated.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .function_preservation import chunked_teacher_kl

POPULATIONS = {
    'fitting_seen': 'fitting checkpoint, cached train-role panel; sequences used for fitting gradients',
    'fitting_withheld': 'fitting checkpoint, cached audit-role panel; globally held-out sequence IDs, never fitted',
    'checkpoint_heldout_development': 'checkpoint-held-out GPT state, cached audit-role panel; consumed by LR/stopping decisions',
    'fresh_next_batch': 'complete ordinary batch after the checkpoint; unseen by every bank checkpoint up to it',
}
SEQUENCE_QUANTILES = (.5, .9, .99, 1.)
TOKEN_QUANTILES = (.5, .9, .99, .999, 1.)


@torch.no_grad()
def token_kl_and_ce(engine, parameters, teacher):
    """Per-token KL(teacher||student) and next-token NLL for one teacher microbatch.

    Arithmetic mirrors ``FunctionalPreservation.student_parameter_loss``: one FP32
    student forward, FP64 log-softmax, FP64 KL against the cached teacher log
    probabilities. The production chunked mean is evaluated on the same logits
    and returned for an exact parity check; the caller decides the tolerance.
    """
    if teacher.teacher_id != engine.teacher_id:
        raise ValueError('cached teacher probabilities belong to a different frozen checkpoint')
    if teacher.data_role not in ('past_training', 'synthetic', 'fresh_next_training_audit'):
        raise ValueError('validation/test data cannot be audited through the preservation objective')
    if len(teacher.inputs) > engine.microbatch_sequences:
        raise ValueError('teacher microbatch exceeds configured student microbatch size')
    if teacher.log_probs.dtype != torch.float64:
        raise ValueError('confidence audit requires FP64 teacher log probabilities')
    engine._validate_replacements(parameters)
    inputs = teacher.inputs.to(device=engine.device)
    logits = engine._forward(parameters, inputs)
    engine.compute.student_forwards += 1
    engine.compute.student_tokens += inputs.numel()
    production = chunked_teacher_kl(logits, teacher.log_probs, engine.chunk_tokens)
    sequences, length, _ = logits.shape
    if length < 2:
        raise ValueError('next-token CE requires sequence length >=2')
    flat = logits.flatten(0, 1)
    teacher_flat = teacher.log_probs.flatten(0, 1)
    target = inputs.roll(-1, 1).flatten()
    kl = torch.empty(len(flat), dtype=torch.float64, device=logits.device)
    student_nll = torch.empty_like(kl)
    teacher_nll = torch.empty_like(kl)
    for start in range(0, len(flat), engine.chunk_tokens):
        stop = min(start+engine.chunk_tokens, len(flat))
        logq = teacher_flat[start:stop].to(device=logits.device)
        logp = F.log_softmax(flat[start:stop].double(), dim=-1)
        kl[start:stop] = (logq.exp()*(logq-logp)).sum(-1)
        labels = target[start:stop, None]
        student_nll[start:stop] = -logp.gather(1, labels).squeeze(1)
        teacher_nll[start:stop] = -logq.gather(1, labels).squeeze(1)
    kl = kl.view(sequences, length)
    # The final position of each cached input has no known next token.
    student_nll = student_nll.view(sequences, length)[:, :-1]
    teacher_nll = teacher_nll.view(sequences, length)[:, :-1]
    production_mean = float(production)
    if not math.isfinite(production_mean) or not bool(torch.isfinite(kl).all()):
        raise FloatingPointError('nonfinite audit KL')
    return dict(kl=kl.cpu(), student_nll=student_nll.cpu(), teacher_nll=teacher_nll.cpu(),
                production_mean_kl=production_mean,
                parity_abs_difference=abs(float(kl.mean())-production_mean))


def _quantiles(values, probabilities):
    values = values.flatten().double()
    if not len(values):
        raise ValueError('empty quantile input')
    q = torch.quantile(values, torch.tensor(probabilities, dtype=torch.float64))
    return {f'p{p*100:g}': float(v) for p, v in zip(probabilities, q)}


def pooled_quantiles(values, probabilities=TOKEN_QUANTILES):
    """Exact quantiles of a large pooled vector with NumPy (linear interpolation).

    torch.quantile rejects inputs beyond about 2^24 elements on some builds; a
    production pool of 800 x 128 x 1024 tokens exceeds that. NumPy handles any
    size. Arithmetic is FP64, but if ``values`` were stored as float32 (the
    audit's on-disk pools), the quantiles are exact for those rounded values,
    not bit-identical to quantiles of the original FP64 per-token KL.
    """
    array = np.asarray(values.cpu().numpy() if isinstance(values, torch.Tensor) else values).reshape(-1)
    if array.size == 0:
        raise ValueError('empty pooled quantile input')
    if not np.isfinite(array).all():
        raise FloatingPointError('nonfinite pooled values')
    q = np.quantile(array.astype(np.float64, copy=False), np.asarray(probabilities, dtype=np.float64), method='linear')
    return {f'p{p*100:g}': float(v) for p, v in zip(probabilities, q)}


def bootstrap_mean_interval(values, *, resamples=1000, seed=0, level=.95):
    """Percentile bootstrap of the mean over the given units (sequences or states)."""
    values = torch.as_tensor(values, dtype=torch.float64).flatten()
    if len(values) < 2 or resamples < 10 or not 0 < level < 1:
        raise ValueError('bootstrap needs >=2 units, >=10 resamples and a level in (0,1)')
    generator = torch.Generator().manual_seed(int(seed))
    index = torch.randint(len(values), (resamples, len(values)), generator=generator)
    means = values[index].mean(1)
    lower, upper = torch.quantile(means, torch.tensor([(1-level)/2, 1-(1-level)/2], dtype=torch.float64))
    return dict(low=float(lower), high=float(upper), resamples=resamples, level=level, units=len(values))


def sequence_summary(kl, student_nll, teacher_nll, *, tolerance, bootstrap_seed=0, resamples=1000):
    """Distribution over sequences and tokens for one (checkpoint, population).

    ``tolerance`` is the trainer's checkpoint-mean tolerance; the sequence and
    token rows are reported against it as descriptive fractions only.
    """
    if kl.ndim != 2 or student_nll.shape != teacher_nll.shape or len(student_nll) != len(kl):
        raise ValueError('expected (sequence, token) KL and matching NLL tensors')
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('positive finite tolerance required')
    per_sequence = kl.double().mean(1)
    ce_change = (student_nll.double()-teacher_nll.double()).mean(1)
    result = dict(sequences=int(kl.shape[0]), tokens=int(kl.numel()), ce_tokens=int(student_nll.numel()),
                  ce_scope='next-token CE over the first L-1 positions of each cached input (the final position has no label); KL over all L positions; not the ordinary training-loss readout',
                  mean_kl=float(per_sequence.mean()),
                  sequence_kl_quantiles=_quantiles(per_sequence, SEQUENCE_QUANTILES),
                  token_kl_quantiles=_quantiles(kl, TOKEN_QUANTILES),
                  fraction_sequences_above_tolerance=float((per_sequence > tolerance).double().mean()),
                  fraction_tokens_above_tolerance=float((kl > tolerance).double().mean()),
                  student_ce=float(student_nll.double().mean()), teacher_ce=float(teacher_nll.double().mean()),
                  ce_change=float(ce_change.mean()),
                  sequence_ce_change_quantiles=_quantiles(ce_change, SEQUENCE_QUANTILES),
                  fraction_sequences_ce_worse=float((ce_change > 0).double().mean()),
                  per_sequence_kl=[float(v) for v in per_sequence],
                  per_sequence_ce_change=[float(v) for v in ce_change],
                  kl_tolerance=tolerance)
    if len(per_sequence) >= 2:
        result['bootstrap_mean_kl_95'] = bootstrap_mean_interval(per_sequence, resamples=resamples, seed=bootstrap_seed)
    return result


@torch.no_grad()
def compare_edits(current, reference, base, groups, editable):
    """Per-group cosine and amplitude ratio between two realized edits of one checkpoint.

    Both edits are the materialized parameters the GPT actually sees (same
    production ``build_proposal`` path), so no new action algebra is introduced.
    This distinguishes pure attenuation (cosine near one, ratio below one) from
    a changed direction (cosine well below one) without any GPT evaluation.
    """
    result = {}
    total_dot = total_cur = total_ref = 0.
    for group, names in groups.items():
        dot = cur_sq = ref_sq = 0.
        for name in names:
            if name not in editable:
                continue
            origin = base[name].double()
            d_cur = current[name].double()-origin
            d_ref = reference[name].double()-origin
            dot += float((d_cur*d_ref).sum()); cur_sq += float(d_cur.square().sum()); ref_sq += float(d_ref.square().sum())
        total_dot += dot; total_cur += cur_sq; total_ref += ref_sq
        result[group] = dict(cosine=dot/math.sqrt(cur_sq*ref_sq) if cur_sq > 0 and ref_sq > 0 else None,
                             amplitude_ratio=math.sqrt(cur_sq/ref_sq) if ref_sq > 0 else None,
                             current_edit_norm=math.sqrt(cur_sq), reference_edit_norm=math.sqrt(ref_sq))
    return dict(groups=result,
                overall_cosine=total_dot/math.sqrt(total_cur*total_ref) if total_cur > 0 and total_ref > 0 else None,
                overall_amplitude_ratio=math.sqrt(total_cur/total_ref) if total_ref > 0 else None)


def population_summary(rows, *, tolerance, bootstrap_seed=0, pooled_tokens=None):
    """Aggregate state rows of one population; never a pass/fail label.

    ``pooled_tokens`` is the concatenation of every per-token KL of the
    population's rows (NumPy array or tensor, any size); when given, true pooled
    token quantiles are reported via ``pooled_quantiles``.
    The per-key maximum over state-level token quantiles is always reported
    under an explicit name and is not a pooled quantile.
    """
    if not rows:
        raise ValueError('nonempty population required')
    expected_tokens = int(sum(r['summary']['tokens'] for r in rows))
    if pooled_tokens is not None and int(np.asarray(pooled_tokens).size if not isinstance(pooled_tokens, torch.Tensor) else pooled_tokens.numel()) != expected_tokens:
        raise ValueError('pooled token count differs from the rows of this population')
    means = torch.tensor([r['summary']['mean_kl'] for r in rows], dtype=torch.float64)
    pooled_sequences = torch.tensor([v for r in rows for v in r['summary']['per_sequence_kl']], dtype=torch.float64)
    worst = sorted(rows, key=lambda r: -r['summary']['mean_kl'])[:5]
    early = [r for r in rows if r['state_id'] <= 20]
    result = dict(states=len(rows), sequences=int(sum(r['summary']['sequences'] for r in rows)),
                  tokens=int(sum(r['summary']['tokens'] for r in rows)),
                  mean_of_state_means_kl=float(means.mean()),
                  state_mean_kl_quantiles=_quantiles(means, SEQUENCE_QUANTILES),
                  states_with_mean_kl_at_or_below_tolerance=int((means <= tolerance).sum()),
                  pooled_sequence_kl_quantiles=_quantiles(pooled_sequences, SEQUENCE_QUANTILES),
                  maximum_state_token_kl_quantiles={k: max(r['summary']['token_kl_quantiles'][k] for r in rows)
                                                    for k in rows[0]['summary']['token_kl_quantiles']},
                  pooled_token_kl_quantiles=pooled_quantiles(pooled_tokens, TOKEN_QUANTILES) if pooled_tokens is not None else None,
                  mean_ce_change=float(sum(r['summary']['ce_change'] for r in rows)/len(rows)),
                  ce_scope=rows[0]['summary'].get('ce_scope'),ce_tokens=int(sum(r['summary'].get('ce_tokens',0) for r in rows)),
                  mean_group_l2_percentage_reduction=float(sum(r['mean_group_l2_percentage_reduction'] for r in rows)/len(rows)),
                  worst_states=[dict(state_id=r['state_id'], mean_kl=r['summary']['mean_kl'],
                                     sequence_p99=r['summary']['sequence_kl_quantiles']['p99']) for r in worst],
                  early_states=[dict(state_id=r['state_id'], mean_kl=r['summary']['mean_kl']) for r in sorted(early, key=lambda r: r['state_id'])],
                  kl_tolerance=tolerance)
    if len(rows) >= 2:
        result['bootstrap_mean_of_state_means_95'] = bootstrap_mean_interval(means, seed=bootstrap_seed)
    compared = [r['reference_comparison'] for r in rows if r.get('reference_comparison')]
    if compared:
        cosines = [c['overall_cosine'] for c in compared if c['overall_cosine'] is not None]
        ratios = [c['overall_amplitude_ratio'] for c in compared if c['overall_amplitude_ratio'] is not None]
        result['reference_comparison'] = dict(states=len(compared),
            overall_cosine=dict(min=min(cosines), median=float(torch.tensor(cosines).median()), max=max(cosines)) if cosines else None,
            overall_amplitude_ratio=dict(min=min(ratios), median=float(torch.tensor(ratios).median()), max=max(ratios)) if ratios else None,
            group_median_cosine={g: float(torch.tensor([c['groups'][g]['cosine'] for c in compared if c['groups'][g]['cosine'] is not None]).median())
                                 for g in compared[0]['groups'] if any(c['groups'][g]['cosine'] is not None for c in compared)},
            group_median_amplitude_ratio={g: float(torch.tensor([c['groups'][g]['amplitude_ratio'] for c in compared if c['groups'][g]['amplitude_ratio'] is not None]).median())
                                          for g in compared[0]['groups'] if any(c['groups'][g]['amplitude_ratio'] is not None for c in compared)})
    return result
