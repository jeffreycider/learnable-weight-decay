"""Finite preservation adapter for the explicitly adaptive U C V.T family.

This supports actual edited-weight or explicit thin-factor credit, not old
fixed-anchor gradient banks.
The existing tokenwise/image loss engines can consume the returned parameter
mapping and (original outputs, detached leaves) credit pair unchanged. Every
U, C, V participates in that final VJP. No model evaluation occurs here.
"""
import math

import torch

from .adaptive_action import AdaptiveAction, AdaptiveFactors
from .edit_scope import editable_names, editable_weight_norm_sq, keep_readonly_parameters, scoped_snapshot


def detached_adaptive_credit(action):
    if not isinstance(action, AdaptiveAction):
        raise TypeError('explicit adaptive action required')
    originals, leaves = [], []

    def leaf(value):
        result = value.detach().clone().requires_grad_(value.requires_grad)
        originals.append(value)
        leaves.append(result)
        return result

    detached = AdaptiveAction(
        {n:AdaptiveFactors(leaf(f.U), leaf(f.C), leaf(f.V)) for n,f in action.matrices.items()},
        {n:leaf(v) for n,v in action.vectors.items()})
    return detached, (originals, leaves)


def group_rms_scales(grouped, *, dtype=torch.float32):
    """Actor supplies its dimension normalization; apply group RMS only once."""
    result = {}
    for group, names in grouped.groups.items():
        count = sum(grouped.parameters[n].numel() for n in names)
        rms = (grouped.reference_norms[group] / math.sqrt(count)).to(dtype=dtype)
        result.update({n:rms for n in names})
    return result


def build_adaptive_proposal(actor_forward, snapshot, engine, grouped, *,
                            radius_fraction, norm_squared=None,
                            shrinkage_coefficient=1., detach_credit=True,
                            execution_capture=None, materialization_grad=True):
    """Same realized group objective and global radius, a new action family.

    Group norm evaluation scans the actual stored-dtype replacements. This
    avoids applying cached W B / W.T A identities to newly learned U/V.
    ``detach_credit=False`` is for explicit differentiation/reference tests;
    it is not a separately trained algorithm or an extra model evaluation.
    Optional execution_capture(action, scale) exposes the exact scaled factor
    graph to experimental thin-backward execution. It must not mutate inputs.
    ``materialization_grad=False`` retains the thin actor/factor/guard graph,
    but creates stored values and norm statistics without autograd. It requires
    an explicit capture. With detach_credit=True, that capture must implement
    capture_detached_materialization(editable) and restore thin training credit.
    The returned norm objective alone is a measurement, not a training loss.
    """
    if not math.isfinite(radius_fraction) or radius_fraction <= 0:
        raise ValueError('positive finite radius required')
    if not math.isfinite(shrinkage_coefficient) or shrinkage_coefficient < 0:
        raise ValueError('nonnegative finite shrinkage coefficient required')
    if type(materialization_grad) is not bool:
        raise TypeError('materialization_grad must be an explicit boolean')
    detached_capture=getattr(execution_capture,'capture_detached_materialization',None)
    if not materialization_grad and (execution_capture is None or (detach_credit and not callable(detached_capture))):
        raise ValueError('detached materialization requires explicit thin execution capture and detach_credit=False or an explicit detached-materialization adapter')
    editable = editable_names(engine.parameters, 'grouped') - grouped.frozen_parameter_names
    squared = editable_weight_norm_sq(engine.parameters, editable) if norm_squared is None else norm_squared
    snapshot = scoped_snapshot(snapshot, squared, radius_fraction, scope='grouped')
    action = actor_forward(snapshot)
    if not isinstance(action, AdaptiveAction):
        raise TypeError('adaptive proposal cannot consume fixed-anchor factors')
    expected_matrix = {n for n,p in engine.parameters.items() if p.ndim == 2}
    expected_vector = {n for n,p in engine.parameters.items() if p.ndim == 1}
    extra_matrix = set(action.matrices) - expected_matrix
    extra_vector = set(action.vectors) - expected_vector
    if (not expected_matrix.issubset(action.matrices) or
            not expected_vector.issubset(action.vectors) or
            not extra_matrix.issubset(snapshot.matrices) or
            not extra_vector.issubset(snapshot.vectors)):
        raise ValueError('adaptive action inventory differs from finite model')
    # CNN observations include frozen whitening filters and normalization
    # parameters while its finite engine exposes only writable parameters.
    # Keep their READ paths in the actor graph, but exclude their WRITE outputs
    # before constructing detached leaves, the radius or the finite proposal.
    action = AdaptiveAction({n:f for n,f in action.matrices.items() if n in expected_matrix},
                            {n:v for n,v in action.vectors.items() if n in expected_vector})
    credit = None
    if detach_credit and torch.is_grad_enabled():
        action, credit = detached_adaptive_credit(action)
    fp64 = any(f.U.dtype == torch.float64 for f in action.matrices.values())
    if fp64 and (any(f.U.dtype != torch.float64 for f in action.matrices.values()) or
                 any(v.dtype != torch.float64 for v in action.vectors.values())):
        raise ValueError('FP64 adaptive proposals require FP64 matrix and vector factors')
    scales = group_rms_scales(grouped, dtype=torch.float64 if fp64 else torch.float32)
    action = AdaptiveAction(
        {n:f.scaled(scales[n]) if n in editable else
         AdaptiveFactors(f.U.detach(), torch.zeros_like(f.C), f.V.detach()) for n,f in action.matrices.items()},
        {n:v*scales[n] if n in editable else torch.zeros_like(v) for n,v in action.vectors.items()})
    raw_norm = action.norm_sq().clamp_min(1e-24).sqrt()
    scale = torch.minimum(raw_norm.new_ones(()), snapshot.radius/(raw_norm+1e-12))
    if execution_capture is not None:
        execution_capture(action, scale)
    if not materialization_grad and callable(detached_capture):
        detached_capture(editable)
    with torch.set_grad_enabled(torch.is_grad_enabled() and materialization_grad):
        if fp64:
            # The CNN legacy adapter rounds delta before adding base. This explicit
            # mode instead computes the whole proposal in FP64 and casts ONCE.
            # Orientation is finite-engine metadata, never inferred from shapes.
            parameters = {}
            transpose = getattr(engine, 'transpose_storage', {})
            for name, base in engine.parameters.items():
                delta = action.matrices[name].dense() if base.ndim == 2 else action.vectors[name]
                if base.ndim == 2 and transpose.get(name, False):
                    delta = delta.T
                if delta.shape != base.shape:
                    raise ValueError('adaptive FP64 stored shape differs: '+name)
                parameters[name] = (base.double() + scale * delta).to(base.dtype)
        else:
            parameters = engine.edited_parameters(action, scale)
        parameters = keep_readonly_parameters(parameters, engine.parameters, editable)
        result = grouped.evaluate(parameters)
        norm_capture = getattr(execution_capture, 'capture_norm_measurement', None)
        if norm_capture is not None:
            norm_capture(parameters, grouped, result)
        unweighted = 100*result.mean_relative_change
        objective = shrinkage_coefficient*unweighted
    stats = dict(result.metrics(), action_family=action.family,
                 factor_precision='fp64' if fp64 else 'fp32',
                 proposal_storage_rule='fp64_sum_then_one_cast' if fp64 else 'legacy_engine',
                 group_scaling='group_RMS_only', norm_term=float(objective.detach()),
                 unweighted_norm_term=float(unweighted.detach()),
                 shrinkage_coefficient=shrinkage_coefficient, trust_term=0.,
                 edit_size_penalty_enabled=False, raw_action_norm=float(raw_norm.detach()),
                 global_scale=float(scale.detach()), hard_radius=float(snapshot.radius),
                 radius_saturated=bool(scale.detach()<1),
                 proposed_edit_norm=float((raw_norm*scale).detach()),
                 realized_edit_norm=float(result.edit_norm_sq.detach().sqrt()),
                 zero_reference_groups=list(grouped.zero_reference_groups),
                 frozen_zero_group_parameters=sorted(grouped.frozen_parameter_names),
                 readonly_observation_only_parameters=sorted(extra_matrix | extra_vector),
                 fixed_anchor_gradient_banks_used=False)
    return parameters, objective, stats, editable, credit
