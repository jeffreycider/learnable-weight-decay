"""Perturbation sensitivity of the small-update logit response (diagnostic only).

The owner's qualification reports the relative error of the post-Adam LOGIT
RESPONSE (logits after the update minus logits before) between a candidate and
the dense reference. Measured absolute response errors were ~5e-5 at both
nominal and 0.1x learning rate while the actor-UPDATE error scaled with the
rate. This helper measures how much logit response a parameter difference of
the ACTUAL candidate-minus-dense shape and size produces, so the candidate's
response error can be read against that sensitivity instead of being quoted
as a bare relative number.

Design (after Astra's review of the first version):
  * the SAME actor module and compiled callable are reused; parameters are
    loaded in place (no deepcopy, no fresh compiled specialisation, no address
    change);
  * the primary measurement replays the ACTUAL delta: load the dense-updated
    state, record logits; load the candidate-updated state, record logits;
    report stored per-tensor L2/max of the delta and the realised logit
    difference. That is the candidate's own number, recomputed through the same
    forward;
  * random controls are direction-randomised perturbations with the SAME
    per-tensor L2 as the actual delta (not +-max on every element, which would
    inflate the L2 by sqrt(N)); they give a sensitivity DISTRIBUTION for
    differences of that size, not a lower bound or a 'floor';
  * optional group ablation: apply the actual delta only to a named subset of
    parameters (e.g. output heads vs residual blocks) to see which part of the
    delta carries the response difference.
Nothing in the GPT/actor arithmetic changes; this is read-only diagnostics and
must not be used to relax any declared gate.

Memory guidance (Astra review): at 64 sequences the three resident FP32 logits
tensors are 3 x 13.2 GB plus the GPT's own working memory; do not run this
beside a live ~50 GB CUDA-graph private pool. Close/free the graph first, or
run the diagnostic on the first 16 sequences (3 x 3.3 GB) as the owner's
qualification scripts do.
"""
from __future__ import annotations
import math
import statistics
import torch


@torch.no_grad()
def _stream_stats(a, b, chunk_rows=2048):
    """max|a-b|, sum (a-b)^2 and sum b^2 accumulated in FP64 over row chunks of the
    FP32 inputs; no full-size FP64 copy is ever materialised (a 64x1024x50304 logits
    tensor is 13.2 GB in FP32, 26.4 GB in FP64)."""
    if a.shape != b.shape:
        raise ValueError('logits shapes differ')
    a2, b2 = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
    max_abs, sumsq, ref_sumsq = 0., 0., 0.
    for start in range(0, a2.shape[0], chunk_rows):
        da = (a2[start:start + chunk_rows].double() - b2[start:start + chunk_rows].double())
        max_abs = max(max_abs, float(da.abs().max()) if da.numel() else 0.)
        sumsq += float(da.square().sum())
        ref_sumsq += float(b2[start:start + chunk_rows].double().square().sum())
    return max_abs, sumsq, ref_sumsq, a2.numel()


@torch.no_grad()
def _load(actor, state):
    for name, parameter in actor.named_parameters():
        parameter.copy_(state[name])


@torch.no_grad()
def _delta_stats(dense_state, candidate_state):
    rows = {}
    total = 0.
    for name, dense in dense_state.items():
        difference = (candidate_state[name].double() - dense.double())
        rows[name] = dict(l2=float(difference.norm()), max_abs=float(difference.abs().max()) if difference.numel() else 0.)
        total += rows[name]['l2'] ** 2
    return rows, math.sqrt(total)


@torch.no_grad()
def response_sensitivity(actor, logits_fn, logits_before, dense_state, candidate_state, *,
                         control_seeds=(1, 2, 3, 4, 5), ablation_groups=None, chunk_rows=2048):
    """Replay the actual candidate-minus-dense delta and size-matched random controls.

    ``logits_fn(actor) -> student logits`` must build the proposal grad-enabled
    and run the GPT under no_grad (fitting-mode readout) and use the same
    compiled callable the qualification used. ``dense_state``/``candidate_state``
    are ``{name: tensor}`` post-update actor parameters. The actor's current
    parameters are restored to ``dense_state`` on return.
    """
    names = [n for n, _ in actor.named_parameters()]
    if set(names) != set(dense_state) or set(names) != set(candidate_state):
        raise ValueError('state inventories must match the actor parameters')
    per_tensor, delta_l2 = _delta_stats(dense_state, candidate_state)
    try:
        _load(actor, dense_state)
        # FP32 storage only: one resident reference logits tensor; everything else is streamed.
        dense_after = logits_fn(actor)
        if dense_after.dtype != torch.float32:
            raise ValueError('logits_fn must return FP32 logits (storage is kept in FP32; statistics accumulate in FP64 per chunk)')
        _, response_sumsq, _, _ = _stream_stats(dense_after, logits_before, chunk_rows)
        response_norm = math.sqrt(response_sumsq)

        def realised(state):
            _load(actor, state)
            current = logits_fn(actor)
            max_abs, sumsq, _, count = _stream_stats(current, dense_after, chunk_rows)
            del current
            return dict(max_abs=max_abs, rms=math.sqrt(sumsq / count) if count else 0.,
                        relative_to_response=math.sqrt(sumsq) / response_norm if response_norm else None)

        actual = realised(candidate_state)
        controls = []
        for seed in control_seeds:
            generator = torch.Generator(device='cpu').manual_seed(int(seed))
            state = {}
            realised_l2 = {}
            for name in names:
                dense = dense_state[name]
                direction = torch.randn(dense.shape, generator=generator, dtype=torch.float64)
                norm = float(direction.norm())
                scale = per_tensor[name]['l2'] / norm if norm else 0.
                # one tensor at a time: no full-model FP64 copy is kept alive
                perturbed = (dense.double() + direction.to(dense.device) * scale).to(dense.dtype)
                state[name] = perturbed
                # realised magnitude AFTER the cast to the stored dtype (FP32 rounding can shrink tiny deltas)
                realised_l2[name] = float((perturbed.double() - dense.double()).norm())
            row = realised(state)
            row['seed'] = int(seed)
            row['realised_delta_l2_total'] = math.sqrt(sum(v * v for v in realised_l2.values()))
            row['realised_l2_ratio_min'] = min((realised_l2[n] / per_tensor[n]['l2']) for n in names if per_tensor[n]['l2'] > 0) if any(per_tensor[n]['l2'] > 0 for n in names) else None
            controls.append(row)
            del state
        ablations = {}
        for label, subset in (ablation_groups or {}).items():
            subset = set(subset)
            if not subset <= set(names):
                raise ValueError('ablation group names must be actor parameters: ' + label)
            state = {n: (candidate_state[n] if n in subset else dense_state[n]) for n in names}
            ablations[label] = dict(realised(state), delta_l2=math.sqrt(sum(per_tensor[n]['l2'] ** 2 for n in subset)))
    finally:
        _load(actor, dense_state)   # restore even if logits_fn raised mid-way
    return dict(delta_l2_total=delta_l2, delta_per_tensor=per_tensor, response_norm=response_norm,
                actual=actual, controls=controls,
                control_median_max_abs=statistics.median(c['max_abs'] for c in controls) if controls else None,
                control_median_rms=statistics.median(c['rms'] for c in controls) if controls else None,
                ablations=ablations,
                memory='FP32 logits storage only: peak = logits_before + dense_after + one current logits tensor (3 x 3.3 GB at 16 seq, 3 x 13.2 GB at 64 seq) plus one FP64 row chunk; no full FP64 arrays',
                note='controls are size-matched random directions (sensitivity distribution), not a lower bound; realised_delta_l2_total reports the post-cast magnitude; gates unchanged')
