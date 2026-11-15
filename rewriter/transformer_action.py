"""Current grouped TransformerChisel action and thin direct lag-one credit.

This is optimizer algebra only: no GPT evaluation, fitting schedule, distributed
policy, or live trainer integration. Base parameters must be detached immutable
storage, separate from any live weights that the caller will edit. Observations
are copied so later sensor reuse/anchor refresh cannot alter an executed action.
"""
from dataclasses import dataclass, fields, is_dataclass, replace
import math
from collections.abc import Mapping

import torch

from .algebra import Factors, RawAction
from .edit_scope import editable_names, editable_weight_norm_sq, scoped_snapshot
from .grouped_norm import GroupedNormObjective


def _copy_observation(value):
    if value is None:return None
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if is_dataclass(value):
        return type(value)(**{f.name: _copy_observation(getattr(value, f.name)) for f in fields(value)})
    if isinstance(value, dict):
        return {k: _copy_observation(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_copy_observation(v) for v in value)
    raise TypeError('observation contains an unsupported field')


def _tensors(value):
    if value is None:return []
    if isinstance(value, torch.Tensor):
        return [value]
    if is_dataclass(value):
        return [t for f in fields(value) for t in _tensors(getattr(value, f.name))]
    if isinstance(value, Mapping):
        return [t for child in value.values() for t in _tensors(child)]
    if isinstance(value, (tuple, list)):
        return [t for child in value for t in _tensors(child)]
    raise TypeError('expected tensor observation tree')


@dataclass(frozen=True)
class ActionReferenceCache:
    """One immutable post-base reference shared by optional candidate/final calls."""
    parameters: dict
    grouped: GroupedNormObjective
    editable: frozenset
    norm_squared: torch.Tensor
    _versions: tuple

    def validate(self, parameters):
        if parameters.keys()!=self.parameters.keys() or any(parameters[n] is not p for n,p in self.parameters.items()):
            raise ValueError('action reference cache belongs to different base tensors')
        for name,value,version in self._versions:
            if value._version!=version:
                raise RuntimeError('immutable action reference cache changed: '+name)


def build_action_reference_cache(parameters):
    """Compute group references and the existing global-radius norm once.

    This still scans dense base weights initially: one grouped-reference pass
    and one existing combined-radius accumulation. Subsequent prepared actions
    use those constants without rescanning them or changing norm arithmetic.
    """
    parameters=dict(parameters)
    if not parameters or any(p.requires_grad for p in parameters.values()):
        raise ValueError('detached immutable base parameters required')
    grouped=GroupedNormObjective(parameters,zero_reference_policy='freeze')
    editable=frozenset(editable_names(parameters,'grouped')-grouped.frozen_parameter_names)
    norm_squared=(editable_weight_norm_sq(parameters,editable) if editable else
                  next(iter(parameters.values())).new_zeros((),dtype=torch.float64))
    constants=[('base '+n,p) for n,p in parameters.items()]
    constants += [('group squared '+n,p) for n,p in grouped.reference_squared.items()]
    constants += [('group norm '+n,p) for n,p in grouped.reference_norms.items()]
    constants += [('radius squared',norm_squared)]
    return ActionReferenceCache(parameters,grouped,editable,norm_squared,
        tuple((name,p,p._version) for name,p in constants))


@dataclass
class TransformerAction:
    action: RawAction
    scale: torch.Tensor
    snapshot: object
    parameters: dict
    embedding_names: frozenset
    editable: frozenset
    zero_reference_groups: tuple
    actor_parameters: dict
    _versions: tuple
    action_budget: dict | None = None

    def _check_versions(self):
        for label, value, version in self._versions:
            if value._version != version:
                raise RuntimeError(label + ' mutated after action preparation')

    def materialize(self):
        """Finite stored-dtype proposal, for committing/testing; no GPT graph."""
        self._check_versions()
        result = {}
        for name, base in self.parameters.items():
            if name not in self.editable:
                result[name] = base  # Preserve readonly bits, including signed zero.
                continue
            if name in self.action.matrices:
                delta = self.action.matrices[name].dense()
                if name in self.embedding_names:
                    delta = delta.T
            else:
                delta = self.action.vectors[name]
            result[name] = (base.float() + self.scale * delta).to(base.dtype)
        return result

    @torch.no_grad()
    def project_next_gradients(self, gradients, *, global_token_count,
                               loss_reduction='token_sum'):
        """Project the next globally reduced task gradient using OLD anchors.

        `token_sum` means SUM over all ranks/microbatches, before optimizer
        mutation. `global_mean` is already normalized. DDP averaging and AMP
        unscaling are the caller's responsibility; no hidden correction occurs.
        """
        self._check_versions()
        if type(global_token_count) is not int or global_token_count <= 0:
            raise ValueError('positive integer global_token_count required')
        if loss_reduction not in ('token_sum', 'global_mean'):
            raise ValueError('loss_reduction must be token_sum or global_mean')
        denominator = global_token_count if loss_reduction == 'token_sum' else 1
        matrices, vectors = {}, {}
        for name in self.parameters:
            if name not in self.editable:
                continue
            gradient = gradients.get(name)
            base = self.parameters[name]
            if not isinstance(gradient, torch.Tensor) or gradient.shape != base.shape or gradient.device != base.device:
                raise ValueError('missing or incompatible task gradient: ' + name)
            if name in self.action.matrices:
                factor = self.action.matrices[name]
                g = gradient.detach().to(factor.P.dtype)
                if name in self.embedding_names:
                    g = g.T
                matrices[name] = ((g @ factor.anchors.B / denominator).clone(),
                                  (g.T @ factor.anchors.A / denominator).clone())
            else:
                vectors[name] = (gradient.detach().to(self.action.vectors[name].dtype) / denominator).clone()
        return dict(matrices=matrices, vectors=vectors)

    def contraction(self, projected):
        self._check_versions()
        if set(projected) != {'matrices', 'vectors'} or set(projected['matrices']) != set(self.action.matrices) or set(projected['vectors']) != set(self.action.vectors):
            raise ValueError('projected cotangents differ from the executed action')
        terms = [factor.contract(*projected['matrices'][name]) for name, factor in self.action.matrices.items()]
        terms += [(projected['vectors'][name] * value).sum() for name, value in self.action.vectors.items()]
        return self.scale * sum(terms, self.scale.new_zeros(()))

    def record_executed(self, lag_one, *, action_step, actor):
        """Call only AFTER this exact proposal was committed, never for warmup."""
        self._check_versions()
        current = dict(actor.named_parameters())
        if current.keys() != self.actor_parameters.keys() or any(current[n] is not p for n, p in self.actor_parameters.items()):
            raise ValueError('actor parameters were replaced after action preparation')
        lag_one.from_contraction(action_step=action_step, hyperparameters=current,
            contraction=self.contraction,
            guarded_tensors=tuple(value for _, value, _ in self._versions))


def copy_action_budget(budget, *, device):
    """Validate and own the small action-time record, including its provenance."""
    if budget is None:
        return None
    from .action_budget import base_motion_radius
    if not isinstance(budget, dict) or set(budget) != {'action_step','base_update_norm','radius','fraction','scope'}:
        raise ValueError('invalid action budget record')
    if type(budget['action_step']) is not int or budget['action_step'] < 0 or budget['scope'] != 'all_parameters':
        raise ValueError('invalid action budget clock or scope')
    norm, radius = budget['base_update_norm'], budget['radius']
    expected = base_motion_radius(norm, fraction=budget['fraction'])
    if not isinstance(radius, torch.Tensor) or radius.requires_grad or radius.ndim != 0 or radius.dtype != expected.dtype or not torch.equal(radius, expected):
        raise ValueError('action budget radius differs from frozen base motion')
    return dict(budget, base_update_norm=norm.detach().clone().to(device), radius=radius.detach().clone().to(device))


def prepare_transformer_action(actor, snapshot, parameters, *, radius_fraction=.01,
                               embedding_names=('embed.weight',), actor_forward=None, reference_cache=None,
                               action_budget=None):
    """Connected equivalent of the current offline grouped-relative proposal.

    `actor` is the canonical module; `actor_forward` may be its persistent compiled
    callable. Unlike offline fitting, factor outputs are never detached here.
    Group scales reuse the current fitting implementation verbatim.
    """
    from scripts.train_multistate_chisel import group_action_scales

    if not math.isfinite(radius_fraction) or radius_fraction <= 0:
        raise ValueError('positive finite radius_fraction required')
    parameters = dict(parameters)
    if not parameters or any(not isinstance(p, torch.Tensor) or p.requires_grad for p in parameters.values()):
        raise ValueError('base parameters must be detached frozen tensors')
    embedding_names = frozenset(embedding_names)
    if not embedding_names.issubset(parameters) or any(parameters[n].ndim != 2 for n in embedding_names):
        raise ValueError('embedding names must identify stored matrix parameters')
    owned_snapshot = _copy_observation(snapshot)
    if reference_cache is None:
        grouped = GroupedNormObjective(parameters, zero_reference_policy='freeze')
        editable = editable_names(parameters, 'grouped') - grouped.frozen_parameter_names
    else:
        reference_cache.validate(parameters)
        grouped,editable=reference_cache.grouped,reference_cache.editable
    if editable:
        norm_sq = editable_weight_norm_sq(parameters, editable) if reference_cache is None else reference_cache.norm_squared
        owned_snapshot = scoped_snapshot(owned_snapshot, norm_sq, radius_fraction, scope='grouped')
    else:
        zero = owned_snapshot.weight_norm.new_zeros(())
        owned_snapshot = replace(owned_snapshot, weight_norm=zero, radius=zero)
    raw = (actor if actor_forward is None else actor_forward)(owned_snapshot)
    if set(raw.matrices) != set(owned_snapshot.matrices) or set(raw.vectors) != set(owned_snapshot.vectors):
        raise ValueError('actor output differs from snapshot parameter inventory')
    if not editable.issubset(set(raw.matrices) | set(raw.vectors)):
        raise ValueError('editable parameter missing from actor output')
    for name, factor in raw.matrices.items():
        anchors = owned_snapshot.matrices[name].anchors
        if factor.anchors.A is not anchors.A or factor.anchors.B is not anchors.B:
            raise ValueError('actor must retain the supplied action-time anchors: ' + name)
        if name not in parameters:
            raise ValueError('actor matrix missing from base parameters: ' + name)
        shape = parameters[name].T.shape if name in embedding_names else parameters[name].shape
        if tuple(shape) != (factor.P.shape[0], factor.Q.shape[0]):
            raise ValueError('matrix computational orientation mismatch: ' + name)
    for name, value in raw.vectors.items():
        if name not in parameters or value.shape != parameters[name].shape:
            raise ValueError('actor vector differs from base parameters: ' + name)
    ranks = {factor.P.shape[1] for factor in raw.matrices.values()}
    if len(ranks) != 1:
        raise ValueError('current grouped action requires one common probe rank')
    scales = group_action_scales(grouped, ranks.pop())
    action = RawAction(
        {n: Factors(f.P * scales[n], f.Q * scales[n], f.anchors) for n, f in raw.matrices.items() if n in editable},
        {n: value * scales[n] for n, value in raw.vectors.items() if n in editable})
    raw_norm = (action.norm_sq().clamp_min(1e-24).sqrt() if editable else owned_snapshot.radius.new_zeros(()))
    scale = torch.minimum(raw_norm.new_ones(()), owned_snapshot.radius / (raw_norm + 1e-12)) if editable else raw_norm.new_ones(())
    budget = copy_action_budget(action_budget, device=scale.device)
    if budget is not None:
        from .action_budget import restrict_global_scale
        scale = restrict_global_scale(action.norm_sq() if editable else raw_norm.square(), scale, budget['radius'])
    actor_parameters = dict(actor.named_parameters())
    guards = [('actor ' + n, p, p._version) for n, p in actor_parameters.items()]
    guards += [('base ' + n, p, p._version) for n, p in parameters.items()]
    guards += [('action-time observation', p, p._version) for p in _tensors(owned_snapshot)]
    if budget is not None:
        guards += [('action budget '+key, budget[key], budget[key]._version) for key in ('radius','base_update_norm')]
    return TransformerAction(action, scale, owned_snapshot, parameters, embedding_names,
                             frozenset(editable), grouped.zero_reference_groups, actor_parameters, tuple(guards), budget)
