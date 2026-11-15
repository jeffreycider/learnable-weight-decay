"""Per-checkpoint staging for a replayed (CUDA-graph) training step.

In the owner's three-state reuse measurement the replayed compute took 1.967 s
per 64-sequence checkpoint while the whole loop took 2.482 s: checkpoint
reading, verified copies and buffer binding are ~0.5 s per checkpoint and run
on the host while the GPU is idle. During ``graph.replay()`` the host has
nothing to launch, so the NEXT checkpoint's file work can run on a worker
thread without competing with a launch-bound consumer (the regime in which the
eager prefetch bundle measured 2-4% slower).

``CheckpointStager`` composes two tested helpers:
  * ``rewriter.state_prefetch.StatePrefetcher`` stages the verified bank record
    (pinned weights, exact-stride staged observations) and the selected teacher
    panel one checkpoint ahead;
  * ``rewriter.static_snapshot_store.StaticSnapshotStore`` refills the fixed
    observation buffers that a captured actor reads.
Binding of the engine (``cache_reader.load_engine``: SHA/precision/head checks,
in-place parameter copy_) and the production zero-group signature check happen
on the consumer thread, in order, exactly as in the eager path. Nothing learned
is cached. ``bind`` returns the signature so the caller can reject a checkpoint
whose frozen set differs from the captured graph's, and per-phase host seconds
so bind cost is reported separately from replay time.
"""
from __future__ import annotations
import time
import torch
from .state_prefetch import StatePrefetcher
from .static_snapshot_store import StaticSnapshotStore


def frozen_signature(grouped):
    return (tuple(sorted(grouped.frozen_parameter_names)), tuple(grouped.zero_reference_groups))


class CheckpointStager:
    def __init__(self, reader, cache_reader, engine, *, device, select_panel=None, norm_class=None, stage_record=True, constants=None):
        """``constants`` (rewriter.offline_systems.VerifiedStateConstants) reuses the deployed SHA-keyed
        base-norm/group/radius caches instead of recomputing them (Astra review point 3); when given,
        the engine must be its CachedBaseNormPreservation and ``norm_class`` is ignored."""
        self.engine = engine
        self.cache_reader = cache_reader
        self.device = torch.device(device)
        self.norm_class = norm_class
        self.constants = constants
        self.prefetcher = StatePrefetcher(reader, cache_reader, device=device, select_panel=select_panel, stage_record=stage_record)
        self.store = None
        self.signature = None
        self.bound_state = None
        self.last_times = {}

    def schedule(self, state_ids):
        self.prefetcher.schedule(state_ids)

    def bind(self, state_id, *, expected_signature=None, signature_policy='reject'):
        """Stage-in the next checkpoint: wait for the worker, load the engine,
        refill the static observation store, build the grouped objective.

        Returns (record, panel, grouped, signature). With the default
        ``signature_policy='reject'`` a frozen-group signature different from
        ``expected_signature`` raises and leaves the store unbound (one graph per
        signature). With ``'report'`` the bind completes and the caller handles
        the change itself, e.g. the owner's dynamic edit-mask graph, which the
        03:50 owner note showed must validate masks against the LIVE frozen set
        per checkpoint rather than a captured initial list."""
        if signature_policy not in ('reject', 'report'):
            raise ValueError("signature_policy must be 'reject' or 'report'")
        times = {}
        # Invalidate BEFORE the first potential mutation (Astra review point 1): any failure below
        # leaves the stager unbound, never advertising a half-updated engine/store as bound.
        if self.store is not None:
            self.store.bound = False
        self.bound_state = None
        started = time.perf_counter()
        record, panel, worker = self.prefetcher.take(state_id)
        times['wait_for_worker'] = time.perf_counter() - started
        started = time.perf_counter()
        if self.constants is not None:
            self.constants.load(self.cache_reader, record)   # same verified load_engine inside, plus SHA-keyed invariants
        else:
            self.cache_reader.load_engine(self.engine, record)   # verified SHA / precision / head binding, in-place copies
        times['engine_load'] = time.perf_counter() - started
        started = time.perf_counter()
        if self.constants is not None:
            grouped = self.constants.grouped()
        else:
            grouped = self.norm_class(self.engine.parameters, zero_reference_policy='freeze') if self.norm_class else None
        signature = frozen_signature(grouped) if grouped is not None else None
        changed = expected_signature is not None and signature != expected_signature
        if changed and signature_policy == 'reject':
            raise ValueError(f'checkpoint {state_id} has a different frozen-group signature than the captured graph; rejected')
        times['signature_changed'] = bool(changed)
        times['grouped_setup'] = time.perf_counter() - started
        started = time.perf_counter()
        if self.store is None:
            self.store = StaticSnapshotStore(record.snapshot, record.train, device=self.device)
        else:
            self.store.refill(record.snapshot, record.train)
        times['observation_refill'] = time.perf_counter() - started
        times['worker'] = dict(worker)
        self.last_times = times
        self.signature = signature
        self.bound_state = state_id
        return record, panel, grouped, signature

    def static_views(self, record):
        if not self.bound:
            raise RuntimeError('stager is not bound; a previous bind failed or none happened')
        return self.store.snapshot(record.snapshot, record.train)

    @property
    def bound(self):
        return self.store is not None and self.store.bound

    def close(self):
        self.prefetcher.close()
