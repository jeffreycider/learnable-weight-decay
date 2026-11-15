"""Detached exact-real-arithmetic gauge controls, never a learned-policy replacement.

These dense rescalings are outside the anchored rank-2r action family. No training
hook calls this module. Parameters/moments/EMA are never mutated or transported.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Literal

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class GaugeSpec:
    """Explicit computational-orientation names; shared tensors are unsupported.

    ``first_weight`` is MLP-up or attention-value, ``second_weight`` is MLP-down
    or attention-output. First bias participates in optimization and rescaling;
    the second bias is invariant. For attention, flattened channels must already
    be grouped by head; diagonal scaling never mixes any of those channels.
    """
    kind: Literal['relu_squared_mlp', 'attention_value']
    first_weight: str
    second_weight: str
    first_bias: str | None = None
    num_heads: int = 1


@dataclass
class GaugeControl:
    """Changed tensors only, detached copies; merge explicitly for evaluation."""
    replacements: dict[str, Tensor]
    scales: dict[str, Tensor]
    diagnostics: dict


def _attention_group_specs(parameters,model):
    from .grouped_norm import functional_groups
    groups = functional_groups(parameters)
    affected = ('attention.v.weight', 'attention.v.bias', 'attention.out.weight')
    for group in affected:
        norm_sq = sum(parameters[name].double().square().sum() for name in groups[group])
        if not torch.isfinite(norm_sq) or norm_sq <= 0:
            raise ValueError('uniform group gauge requires three nonzero finite group norms')
    specs = [s for s in baseline_gauge_specs(model) if s.kind == 'attention_value']
    expected = set(groups['attention.v.weight'] + groups['attention.v.bias'] + groups['attention.out.weight'])
    names = [n for s in specs for n in (s.first_weight, s.first_bias, s.second_weight)]
    if None in names or len(names) != len(set(names)) or set(names) != expected:
        raise ValueError('attention gauge does not cover the three groups exactly once')
    aliases = {}
    for name, value in parameters.items():
        aliases.setdefault((value.device, value.untyped_storage().data_ptr()), []).append(name)
    if any(len(aliases[(parameters[name].device, parameters[name].untyped_storage().data_ptr())]) != 1
           for name in names):
        raise ValueError('shared gauge parameters are unsupported')
    return groups,specs,affected


@torch.no_grad()
def uniform_attention_group_control(parameters: Mapping[str, Tensor], model: nn.Module, *, value_scale=2.**-.5) -> GaugeControl:
    """Uniform value/output gauge; default minimizes three equal-group norms.

    Scale value weights/biases by a and output weights by 1/a. This preserves
    the real-arithmetic function. The default a=1/sqrt(2) minimizes 2a+1/a;
    a=.5 is a binary numerical control with zero aggregate group reward.
    Full-rank rescalings generally lie outside the anchored action family.
    """
    groups,specs,affected=_attention_group_specs(parameters,model)
    scale = float(value_scale)
    if not math.isfinite(scale) or scale<=0:raise ValueError('positive finite value scale required')
    replacements = {}
    for spec in specs:
        replacements[spec.first_weight] = parameters[spec.first_weight].detach() * scale
        replacements[spec.first_bias] = parameters[spec.first_bias].detach() * scale
        replacements[spec.second_weight] = parameters[spec.second_weight].detach() / scale
    return GaugeControl(replacements, {}, dict(
        diagnostic_only=True, deployable=False, outside_anchored_action_family=True,
        value_scale=scale, output_weight_scale=1/scale,
        expected_mean_group_l2_percentage_reduction=100*(3-2*scale-1/scale)/15,
        guarantee='Exact real-arithmetic function identity; finite precision must be measured',
        changed_groups=list(affected), optimizer_state_changed=False))


@torch.no_grad()
def single_channel_binary_attention_control(parameters: Mapping[str, Tensor], model: nn.Module) -> GaugeControl:
    """Choose one binary channel gauge using only exact frozen weight energies.

    This finite list of algebraic diagnostics is not a learned optimizer or a
    GPT/query search. Rank-one corrections need not lie in fixed anchor spans.
    """
    groups,specs,affected=_attention_group_specs(parameters,model)
    energies=[sum(parameters[n].double().square().sum() for n in groups[g]) for g in affected]
    best=None
    for spec in specs:
        v=parameters[spec.first_weight].double().square().sum(1)/energies[0]
        b=parameters[spec.first_bias].double().square()/energies[1]
        o=parameters[spec.second_weight].double().square().sum(0)/energies[2]
        for scale in (.5,2.):
            gain=3-(1+(scale*scale-1)*v).sqrt()-(1+(scale*scale-1)*b).sqrt()-(1+(1/(scale*scale)-1)*o).sqrt()
            channel=int(gain.argmax());score=float(gain[channel])
            if best is None or score>best[0]:best=(score,spec,channel,scale)
    score,spec,channel,scale=best
    replacements={n:parameters[n].detach().clone() for n in (spec.first_weight,spec.first_bias,spec.second_weight)}
    replacements[spec.first_weight][channel].mul_(scale)
    replacements[spec.first_bias][channel].mul_(scale)
    replacements[spec.second_weight][:,channel].div_(scale)
    return GaugeControl(replacements,{},dict(diagnostic_only=True,deployable=False,
        not_restricted_to_anchored_action_family=True,selection='maximum exact equal-group L2 reduction from frozen weight energies only',
        candidate_count=2*sum(parameters[s.first_weight].shape[0] for s in specs),
        value_weight=spec.first_weight,channel=channel,value_scale=scale,output_weight_scale=1/scale,
        expected_mean_group_l2_percentage_reduction=100*score/15,
        guarantee='Exact real-arithmetic function identity; finite precision and ranks must be measured',
        changed_groups=list(affected),optimizer_state_changed=False))


def bounded_optimal_scales(first_energy: Tensor, second_energy: Tensor, *,
                           inverse_power: int, lower: float = .5,
                           upper: float = 2.) -> Tensor:
    """Minimize u*d^2 + v*d^(-2p), independently in each coordinate.

    p=2 for ReLU-squared, p=1 for attention values. Float64 log-domain evaluation
    avoids dividing tiny energies. Both-zero chooses identity; one-zero takes
    the correct boundary. Bounds must include identity to ensure nonincrease.
    """
    if inverse_power not in (1, 2):
        raise ValueError('inverse_power must be 1 (attention) or 2 (ReLU squared)')
    if not (math.isfinite(lower) and math.isfinite(upper) and 0 < lower <= 1 <= upper):
        raise ValueError('positive finite bounds must contain identity')
    if first_energy.shape != second_energy.shape or first_energy.device != second_energy.device:
        raise ValueError('energy tensors must have matching shape and device')
    u, v = first_energy.detach().double(), second_energy.detach().double()
    if not (torch.isfinite(u).all() and torch.isfinite(v).all() and (u >= 0).all() and (v >= 0).all()):
        raise ValueError('energies must be finite and nonnegative')
    scales = torch.ones_like(u)
    both = (u > 0) & (v > 0)
    log_d = (math.log(inverse_power) + v[both].log() - u[both].log()) / (2 * inverse_power + 2)
    scales[both] = log_d.clamp(math.log(lower), math.log(upper)).exp()
    scales[(u == 0) & (v > 0)] = upper
    scales[(u > 0) & (v == 0)] = lower
    return scales


def baseline_gauge_specs(model: nn.Module) -> list[GaugeSpec]:
    """Find all actual PR323 MLP/attention modules, including nested prefixes.

    Other architectures should supply explicit specs after proving the same
    operator identities. Similar attribute names alone do not establish them.
    """
    from baseline.model import MLP, CausalSelfAttention
    result = []
    for name, module in model.named_modules():
        prefix = name + '.' if name else ''
        if isinstance(module, MLP):
            result.append(GaugeSpec('relu_squared_mlp', prefix+'fc.weight',
                                   prefix+'proj.weight', prefix+'fc.bias'))
        elif isinstance(module, CausalSelfAttention):
            result.append(GaugeSpec('attention_value', prefix+'v.weight',
                                   prefix+'proj.weight', prefix+'v.bias', module.num_heads))
    return result


@torch.no_grad()
def norm_optimal_gauge_control(parameters: Mapping[str, Tensor], specs: list[GaugeSpec], *,
                              lower: float = .5, upper: float = 2.,
                              scale_grid: tuple[float, ...] | None = None) -> GaugeControl:
    """Generate an independent bounded norm-optimal control for specified blocks.

    Norm includes both weights and the rescaled input/value bias. Invariant output
    biases and all untouched parameters cancel exactly from the norm difference.
    ``parameters`` must expose every use/alias; tied or overlapping tensors are
    rejected because a local gauge need not preserve other uses. Returned values
    retain storage dtype; cast-rounding means machine function/norm identities
    are approximate, especially BF16. Optional scale_grid selects each coordinate
    minimum over positive grid values within the bounds, including identity.
    Powers of two can reduce rounding drift but do not guarantee machine equality
    for every input (overflow/underflow and operation rounding still matter).
    No forward/backward is executed here.
    """
    grid = None
    if scale_grid is not None:
        grid = tuple(float(value) for value in scale_grid)
        if (not grid or 1. not in grid or any(not math.isfinite(value)
                or value <= 0 or not lower <= value <= upper for value in grid)):
            raise ValueError('scale_grid must contain identity and finite positive bounded values')
        grid = (1.,) + tuple(value for value in grid if value != 1.)  # identity wins ties
    replacements, scales, records = {}, {}, []
    owned = set()
    # Conservative storage alias rejection includes views, even disjoint ones.
    aliases = {}
    for name, value in parameters.items():
        key = (value.device, value.untyped_storage().data_ptr())
        aliases.setdefault(key, []).append(name)
    before_total = after_total = predicted_total = 0.
    for spec in specs:
        if spec.kind not in ('relu_squared_mlp', 'attention_value'):
            raise ValueError('unsupported gauge kind')
        names = [spec.first_weight, spec.second_weight] + ([spec.first_bias] if spec.first_bias else [])
        if len(set(names)) != len(names) or owned.intersection(names):
            raise ValueError('overlapping gauge parameters are unsupported')
        tensors = [parameters[name] for name in names]
        for name, value in zip(names, tensors):
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError('parameters must be finite floating point')
            if len(aliases[(value.device, value.untyped_storage().data_ptr())]) != 1:
                raise ValueError(f'shared parameter storage is unsupported: {name}')
        first, second = tensors[:2]
        bias = tensors[2] if spec.first_bias else None
        if first.ndim != 2 or second.ndim != 2 or first.shape[0] != second.shape[1]:
            raise ValueError('expected first (hidden,input), second (output,hidden)')
        if any(t.device != first.device for t in tensors):
            raise ValueError('each gauge must reside on one device')
        if bias is not None and bias.shape != (first.shape[0],):
            raise ValueError('first bias must have hidden-coordinate shape')
        if spec.kind == 'attention_value' and (not isinstance(spec.num_heads, int)
                or spec.num_heads < 1 or first.shape[0] % spec.num_heads):
            raise ValueError('value coordinates must partition into whole heads')
        u = first.double().square().sum(dim=1)
        if bias is not None:
            u = u + bias.double().square()
        v = second.double().square().sum(dim=0)
        p = 2 if spec.kind == 'relu_squared_mlp' else 1
        d = bounded_optimal_scales(u, v, inverse_power=p, lower=lower, upper=upper)
        if grid is not None:
            candidates = u.new_tensor(grid)
            energy = u[:, None]*candidates.square() + v[:, None]/candidates.pow(2*p)
            d = candidates[energy.argmin(dim=1)]
        replacements[spec.first_weight] = (first.double()*d[:, None]).to(first.dtype)
        replacements[spec.second_weight] = (second.double()/d[None, :].pow(p)).to(second.dtype)
        if bias is not None:
            replacements[spec.first_bias] = (bias.double()*d).to(bias.dtype)
        before = float((u+v).sum())
        predicted = float((u*d.square()+v/d.pow(2*p)).sum())
        after = sum(float(replacements[name].double().square().sum()) for name in names)
        before_total += before; after_total += after; predicted_total += predicted
        scales[spec.first_weight] = d
        records.append(dict(kind=spec.kind, first_weight=spec.first_weight,
                            second_weight=spec.second_weight, norm_sq_before=before,
                            norm_sq_after=after, ideal_norm_sq_after=predicted,
                            min_scale=float(d.min()), max_scale=float(d.max())))
        owned.update(names)
    return GaugeControl(replacements, scales, dict(
        diagnostic_only=True, outside_anchored_action_family=True,
        extra_gpt_evaluations=0, edits_optimizer_state=False,
        scale_policy='continuous_optimum' if grid is None else 'discrete_grid_minimum',
        scale_grid=None if grid is None else list(grid),
        exact_function_claim='real arithmetic for the stated operators only',
        norm_sq_before=before_total, norm_sq_after=after_total,
        norm_sq_change=after_total-before_total,
        ideal_norm_sq_change=predicted_total-before_total,
        blocks=records))
