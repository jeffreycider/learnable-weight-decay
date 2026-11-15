"""Opt-in offline training systems helpers; no changes to production defaults.

Only immutable teacher features are viewed here. Learned Chisel encodings must
never be cached across optimizer updates.
"""
from dataclasses import dataclass
import copy

import torch

from .teacher_cache import TeacherPanel
from .function_preservation import FunctionalPreservation


class CachedBaseNormPreservation(FunctionalPreservation):
    """Cache only the exact FP64 base norm of verified immutable checkpoints.

    The teacher still copies and validates every parameter/buffer, gets a new
    UUID, resets compute counters, and rebinds its head on every load. Unverified
    load_teacher_state calls always recompute. This engine is single-consumer,
    like FunctionalPreservation's functional parameter swaps.
    """
    def __init__(self, *args, **kwargs):
        self._base_norm_by_sha = {}
        self._loading_verified_sha = None
        super().__init__(*args, **kwargs)

    def load_verified_teacher(self, cache_reader, record):
        if self._loading_verified_sha is not None:
            raise RuntimeError('concurrent or recursive teacher load is unsupported')
        self._loading_verified_sha = record.assert_verified()
        try:
            # Owns the same SHA/architecture/precision/head checks as production.
            cache_reader.load_engine(self, record)
        finally:
            self._loading_verified_sha = None

    @torch.no_grad()
    def _refresh_base_norm(self):
        key = self._loading_verified_sha
        if key is not None and key in self._base_norm_by_sha:
            self.base_weight_norm_sq = self._base_norm_by_sha[key].clone()
        else:
            super()._refresh_base_norm()
            if key is not None:
                # Isolate from any caller mutating the public scalar in-place.
                self._base_norm_by_sha[key] = self.base_weight_norm_sq.clone()


class VerifiedStateConstants:
    """Exact group/radius scalars keyed by the owned verified checkpoint SHA.

    A single immutable-layout template shares fused pointer/index tables. Only
    small scalar banks are retained per checkpoint, never weights or encodings.
    References are computed using the original reduction order on first visit.
    The consumer must finish backwards before loading the next state, as usual.
    """
    def __init__(self, engine, norm_class):
        self.engine, self.norm_class = engine, norm_class
        self._layout = self._current_layout()
        self._groups = {}
        self._radii = {}
        self._template = None
        self._loaded = None
        self.hits = dict(groups=0, radius=0)

    def _current_layout(self):
        return tuple((n, id(v), v.shape, v.stride(), v.dtype, v.device)
                     for n, v in self.engine.parameters.items())

    def _signature(self):
        return (self.engine.teacher_id,
                tuple((n, id(v), v._version) for n, v in self.engine.parameters.items()),
                tuple((n, id(v), v._version) for n, v in self.engine.buffers.items()))

    def load(self, cache_reader, record):
        self._loaded = None
        if self._current_layout() != self._layout:
            raise RuntimeError('teacher parameter storage/layout changed')
        self.engine.load_verified_teacher(cache_reader, record)
        self._loaded = (record.assert_verified(), self._signature())

    def _key(self):
        if self._loaded is None or self._loaded[1] != self._signature():
            raise RuntimeError('state constants require an unchanged owned verified load')
        return self._loaded[0]

    @torch.no_grad()
    def grouped(self):
        key = self._key()
        if key not in self._groups:
            value = self.norm_class(self.engine.parameters, zero_reference_policy='freeze')
            self._groups[key] = dict(
                squared={n: v.clone() for n, v in value.reference_squared.items()},
                norms={n: v.clone() for n, v in value.reference_norms.items()},
                zero=value.zero_reference_groups, frozen=value.frozen_parameter_names)
            if self._template is None:
                self._template = copy.copy(value)
                self._template.groups = dict(value.groups)
            return value
        self.hits['groups'] += 1
        cached = self._groups[key]
        value = copy.copy(self._template)
        value.groups = dict(self._template.groups)
        value._reference_versions = {n: v._version for n, v in self.engine.parameters.items()}
        value.reference_squared = {n: v.clone() for n, v in cached['squared'].items()}
        value.reference_norms = {n: v.clone() for n, v in cached['norms'].items()}
        value.zero_reference_groups = cached['zero']
        value.frozen_parameter_names = cached['frozen']
        if hasattr(value, 'reference_vector'):
            value.reference_vector = torch.stack(tuple(value.reference_squared.values()))
        return value

    @torch.no_grad()
    def radius_squared(self, grouped):
        from .edit_scope import editable_names, editable_weight_norm_sq
        sha = self._key()
        if any(v._version != grouped._reference_versions[n]
               for n, v in self.engine.parameters.items()):
            raise RuntimeError('group reference belongs to a previous state load')
        editable = editable_names(self.engine.parameters, 'grouped') - grouped.frozen_parameter_names
        key = (sha, tuple(n for n in self.engine.parameters if n in editable))
        if key not in self._radii:
            self._radii[key] = editable_weight_norm_sq(self.engine.parameters, editable)
        else:
            self.hits['radius'] += 1
        return self._radii[key].clone()


def build_proposal_with_radius(actor_forward, snapshot, engine, grouped, *,
                               radius_fraction, norm_squared, timings=None):
    """Isolated qualification adapter for the production build_proposal.

    The body preserves its operation order; only the invariant input norm is
    supplied by VerifiedStateConstants. A later trainer integration can expose
    this as an optional argument rather than keeping two copies of the body.
    """
    import time
    import scripts.train_multistate_chisel as t
    from .algebra import RawAction, Factors
    editable = t.editable_names(engine.parameters, 'grouped') - grouped.frozen_parameter_names
    snapshot = t.scoped_snapshot(snapshot, norm_squared, radius_fraction, scope='grouped')
    device = next(iter(engine.parameters.values())).device
    action = t.timed(timings, 'actor_forward_seconds', lambda: actor_forward(snapshot), device)
    credit = None
    if torch.is_grad_enabled():
        action, credit = t.detached_action_credit(action)
    if device.type == 'cuda' and timings is not None:
        torch.cuda.synchronize(device)
    edit_started = time.perf_counter()
    action = t.scoped_action(action, editable)
    rank = next(iter(action.matrices.values())).P.shape[1]
    scales = t.group_action_scales(grouped, rank)
    action = RawAction({n: Factors(f.P*scales[n], f.Q*scales[n], f.anchors) for n, f in action.matrices.items()},
                       {n: value*scales[n] for n, value in action.vectors.items()})
    raw_norm = action.norm_sq().clamp_min(1e-24).sqrt()
    scale = torch.minimum(raw_norm.new_ones(()), snapshot.radius/(raw_norm+1e-12))
    parameters = t.keep_readonly_parameters(engine.edited_parameters(action, scale), engine.parameters, editable)
    if device.type == 'cuda' and timings is not None:
        torch.cuda.synchronize(device)
    if timings is not None:
        timings['action_edit_forward_seconds'] = time.perf_counter()-edit_started
    result = t.timed(timings, 'norm_forward_seconds', lambda: grouped.evaluate(parameters), device)
    norm_term = 100*result.mean_relative_change
    trust = norm_term.new_zeros(())
    stats = dict(**result.metrics(), norm_term=float(norm_term.detach()), trust_term=float(trust.detach()),
                 raw_action_norm=float(raw_norm.detach()), global_scale=float(scale.detach()),
                 hard_radius=float(snapshot.radius), radius_saturated=bool(scale.detach() < 1), edit_size_penalty_enabled=False,
                 realized_edit_norm=float(result.edit_norm_sq.detach().sqrt()),
                 zero_reference_groups=list(grouped.zero_reference_groups),
                 frozen_zero_group_parameters=sorted(grouped.frozen_parameter_names))
    stats['proposed_group_l2_percentage_reduction'] = t.proposed_group_reductions(action, scale, snapshot, grouped)
    stats['proposed_edit_norm'] = float((raw_norm*scale).detach())
    stats.update(getattr(grouped, 'state_diagnostics', {}))
    return parameters, norm_term+trust, stats, editable, credit


@dataclass(frozen=True)
class BlockSelectedTeacherPanel:
    """Logical permutation of complete cached generation microbatches.

    Selecting a panel copies small token IDs, but retains the original feature
    storage (including an mmap). Each reconstruction receives exactly the same
    contiguous generation microbatch as eager index_select. No GPU operation,
    precision change, or learned encoding cache is introduced.
    """
    parent: TeacherPanel
    blocks: tuple
    inputs: torch.Tensor
    sequence_ids: tuple
    metadata: dict

    @property
    def features(self):
        """Compatibility materialization; the training path does not call this."""
        size = self.metadata['microbatch_sequences']
        indices = torch.tensor([i for b in self.blocks for i in range(b*size, (b+1)*size)])
        return self.parent.features.index_select(0, indices)

    def select_blocks(self, blocks):
        blocks = _validated_blocks(blocks, len(self.inputs), self.metadata['microbatch_sequences'])
        result = select_panel_blocks(self.parent, [self.blocks[b] for b in blocks])
        # Match eager selection's immediate-parent provenance for nested views.
        return BlockSelectedTeacherPanel(result.parent, result.blocks, result.inputs,
                                         result.sequence_ids,
                                         dict(self.metadata, selected_parent_blocks=list(blocks)))

    def teacher_batch(self, engine, start, stop):
        size = self.metadata['microbatch_sequences']
        if not 0 <= start < stop <= len(self.inputs):
            raise ValueError('invalid panel slice')
        if start % size or stop-start != size:
            raise ValueError('lazy panel slices must be aligned complete generation microbatches')
        parent_start = self.blocks[start//size]*size
        # Delegate all precision, checkpoint binding, and head-version checks.
        return self.parent.teacher_batch(engine, parent_start, parent_start+size)


def _validated_blocks(blocks, count, size):
    blocks = tuple(int(b) for b in blocks)
    if not blocks or len(set(blocks)) != len(blocks) or any(b < 0 or (b+1)*size > count for b in blocks):
        raise ValueError('unique complete in-range microbatch blocks required')
    return blocks


def select_panel_blocks(panel, blocks):
    """Equivalent to TeacherPanel.select_blocks for aligned training consumers."""
    if isinstance(panel, BlockSelectedTeacherPanel):
        return panel.select_blocks(blocks)
    size = panel.metadata['microbatch_sequences']
    blocks = _validated_blocks(blocks, len(panel.inputs), size)
    indices = torch.tensor([i for b in blocks for i in range(b*size, (b+1)*size)], dtype=torch.long)
    return BlockSelectedTeacherPanel(
        panel, blocks, panel.inputs.index_select(0, indices),
        tuple(panel.sequence_ids[i] for i in indices.tolist()),
        dict(panel.metadata, selected_parent_blocks=list(blocks)))
