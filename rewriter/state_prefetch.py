"""Background host staging of the next checkpoint's immutable bytes.

Production loads each checkpoint synchronously on the critical path: SHA/stamp
check, ``torch.load`` (mmap), hundreds of small pageable host->device copies for
the observation, ~170 parameter copies from mmap pages, then the cached teacher
panel read and a 200 MB CPU ``index_select`` of network-backed features. None of
those bytes depend on the actor, so a worker thread can prepare state k+1 while
the GPU runs state k. Nothing learned is cached: only the verified bank record,
the panel selection the trainer would make for the same epoch, and pinned host
copies of those bytes. The consumer still performs every production check
(``cache_reader.load_engine`` -> ``assert_verified`` -> precision/head binding)
on the main thread, and every device-side setup op runs unchanged.

The worker is the only caller of ``StateBankReader.read`` and
``TeacherCacheReader.read`` while active, so their verify-once stamp caches are
never touched concurrently. Depth one is enough: the consumer takes state k and
the worker immediately starts k+1. On CUDA, observation leaves and weights are
pinned so that transfers are DMA from pinned memory; on CPU the worker just
performs the file work. Strides of every observation leaf are preserved exactly
as ``prefix_cache._restore_layout`` does, so downstream kernels see identical
layouts; equality of values, strides and resulting gradients is tested.
"""
from __future__ import annotations
import dataclasses
import queue
import threading
import time
import torch
from .state_bank import StateRecord
from .teacher_cache import TeacherPanel
from .controller import ControllerSnapshot, MatrixObservation, VectorObservation
from .algebra import Anchors, Measurements


def _unique_slice(tensor):
    """Narrow broadcast (stride-0) axes to one element; mirrors _restore_layout."""
    source = tensor
    for axis, (dimension, step) in enumerate(zip(tensor.shape, tensor.stride())):
        if dimension > 1 and step == 0:
            source = source.narrow(axis, 0, 1)
    return source


def stage_leaf(tensor, pin):
    """Host-side staging of one observation leaf: contiguous unique slice (+pinned)."""
    unique = _unique_slice(tensor).contiguous()
    if pin:
        unique = unique.pin_memory()
    return (tuple(tensor.shape), tuple(tensor.stride()), unique)


def restore_leaf(staged, device, non_blocking):
    """Rebuild the exact logical layout on ``device`` from a staged leaf."""
    shape, stride, unique = staged
    size = 0 if any(d == 0 for d in shape) else 1 + sum((d - 1) * s for d, s in zip(shape, stride))
    storage = torch.zeros(size, dtype=unique.dtype, device=device)
    target = storage.as_strided(shape, stride)
    destination = target
    for axis, (dimension, step) in enumerate(zip(shape, stride)):
        if dimension > 1 and step == 0:
            destination = destination.narrow(axis, 0, 1)
    destination.copy_(unique, non_blocking=non_blocking)
    return target


def stage_snapshot(snapshot, train, pin):
    matrices = {name: dict(anchors=(stage_leaf(obs.anchors.A, pin), stage_leaf(obs.anchors.B, pin)),
                           rows=stage_leaf(obs.rows, pin), columns=stage_leaf(obs.columns, pin),
                           weight_probes=tuple(stage_leaf(t, pin) for t in obs.weight_probes),
                           train=tuple(stage_leaf(t, pin) for t in train.matrices[name]))
                for name, obs in snapshot.matrices.items()}
    vectors = {name: dict(coordinates=stage_leaf(obs.coordinates, pin), weight=stage_leaf(obs.weight, pin),
                          train=stage_leaf(train.vectors[name], pin))
               for name, obs in snapshot.vectors.items()}
    return dict(matrices=matrices, vectors=vectors, trajectories=stage_leaf(snapshot.trajectories, pin),
                weight_norm=stage_leaf(snapshot.weight_norm, pin), radius=stage_leaf(snapshot.radius, pin),
                trajectory_mask=(stage_leaf(snapshot.trajectory_mask, pin)
                                 if snapshot.trajectory_mask is not None else None))


def restore_snapshot(staged, device, non_blocking):
    move = lambda leaf: restore_leaf(leaf, device, non_blocking)
    matrices, vectors, train_m, train_v = {}, {}, {}, {}
    for name, obs in staged['matrices'].items():
        matrices[name] = MatrixObservation(Anchors(move(obs['anchors'][0]), move(obs['anchors'][1])),
                                           move(obs['rows']), move(obs['columns']),
                                           tuple(move(t) for t in obs['weight_probes']))
        train_m[name] = tuple(move(t) for t in obs['train'])
    for name, obs in staged['vectors'].items():
        vectors[name] = VectorObservation(move(obs['coordinates']), move(obs['weight']))
        train_v[name] = move(obs['train'])
    snapshot = ControllerSnapshot(matrices, vectors, move(staged['trajectories']),
                                  move(staged['weight_norm']), move(staged['radius']),
                                  trajectory_mask=(move(staged['trajectory_mask'])
                                      if staged.get('trajectory_mask') is not None else None))
    return snapshot, Measurements(train_m, train_v)


@dataclasses.dataclass
class StagedState:
    state_id: int
    record: StateRecord            # model tensors are host copies (pinned on CUDA); observations staged
    staged_observation: dict
    panel: TeacherPanel | None     # selected training blocks, features host-resident (pinned on CUDA)
    worker_seconds: dict
    error: BaseException | None = None


class StatePrefetcher:
    """Depth-one worker thread preparing the next checkpoint and its panel."""

    def __init__(self, reader, cache_reader, *, device, select_panel=None, role='train', pin=None, stage_record=True):
        """``stage_record=False`` stages only the teacher panel (file read, block
        selection copy, pinning) and leaves the bank record to the consumer.
        Measured on one H100 (8 states, warm page cache, no profiler): the full
        record+panel staging bundle was 1.9-3.8% slower than the synchronous
        path (state_load 0.35 -> 0.49 s/state, selection/panel read removed).
        The cause of the extra load time is NOT established (GIL contention is
        a hypothesis); this mode exists to isolate the panel part."""
        self.reader = reader
        self.cache_reader = cache_reader
        self.device = torch.device(device)
        self.select_panel = select_panel
        self.role = role
        self.stage_record = bool(stage_record)
        self.pin = (self.device.type == 'cuda') if pin is None else bool(pin)
        self._queue = queue.Queue(maxsize=1)
        self._requests = queue.Queue()
        self._thread = threading.Thread(target=self._run, name='chisel-state-prefetch', daemon=True)
        self._thread.start()
        self.wait_seconds = 0.
        self.consumed = 0

    # ---- worker side -------------------------------------------------------
    def _stage(self, state_id):
        times = {}
        record = staged = None
        if self.stage_record:
            started = time.perf_counter()
            record = self.reader.read(state_id, device='cpu', model_device='cpu')
            times['bank_read_seconds'] = time.perf_counter() - started
            started = time.perf_counter()
            # Resident host copies of the (mmap) weights; pinned for DMA on CUDA.
            model = {}
            for name, value in record.model.items():
                copy = value.detach().clone() if not self.pin else value.detach().pin_memory()
                model[name] = copy
            times['weight_copy_seconds'] = time.perf_counter() - started
            started = time.perf_counter()
            staged = stage_snapshot(record.snapshot, record.train, self.pin)
            times['observation_stage_seconds'] = time.perf_counter() - started
            source = dict(record.verified_source)
            source['tensor_signature'] = {n: (id(v), v._version) for n, v in model.items()}
            record = dataclasses.replace(record, model=model, verified_source=source)
        panel = None
        if self.cache_reader is not None:
            started = time.perf_counter()
            panel = self.cache_reader.read(state_id, role=self.role)
            if self.select_panel is not None:
                panel = self.select_panel(panel, state_id)
            features, inputs = panel.features, panel.inputs
            if self.pin:
                features = features.pin_memory() if not features.is_pinned() else features
                inputs = inputs.pin_memory() if not inputs.is_pinned() else inputs
            else:
                features = features.clone()
                inputs = inputs.clone()
            panel = TeacherPanel(inputs, features, panel.sequence_ids, panel.metadata)
            times['panel_stage_seconds'] = time.perf_counter() - started
        return StagedState(state_id, record, staged, panel, times)

    def _run(self):
        while True:
            state_id = self._requests.get()
            if state_id is None:
                return
            try:
                item = self._stage(state_id)
            except BaseException as error:  # surfaced to the consumer, never swallowed
                item = StagedState(state_id, None, None, None, {}, error)
            self._queue.put(item)

    # ---- consumer side -----------------------------------------------------
    def schedule(self, state_ids):
        self._pending = list(state_ids)
        self._issued = 0
        self._issue()

    def _issue(self):
        if self._issued < len(self._pending):
            self._requests.put(self._pending[self._issued])
            self._issued += 1

    def take(self, state_id):
        """Block until ``state_id`` is staged; move observations to the device.

        Returns (record, panel). The record's observations are on ``device``
        with exact strides; its model tensors are host copies for the engine's
        ``copy_``; the panel features stay host-resident for ``teacher_batch``.
        """
        started = time.perf_counter()
        item = self._queue.get()
        self.wait_seconds += time.perf_counter() - started
        self._issue()  # start the next one before any device work
        if item.error is not None:
            raise item.error
        if item.state_id != state_id:
            raise RuntimeError(f'prefetch order mismatch: staged {item.state_id}, requested {state_id}')
        if not self.stage_record:
            # Panel-only mode: the consumer reads the record itself (production path).
            self.consumed += 1
            return None, item.panel, item.worker_seconds
        non_blocking = self.pin and self.device.type == 'cuda'
        snapshot, train = restore_snapshot(item.staged_observation, self.device, non_blocking)
        record = dataclasses.replace(item.record, snapshot=snapshot, train=train)
        # Keep the pinned sources referenced until the next take: the caching
        # host allocator already defers reuse behind stream events, this just
        # removes any dependence on that behaviour for in-flight async copies.
        self._last_staged = item.staged_observation
        self.consumed += 1
        return record, item.panel, item.worker_seconds

    def close(self):
        self._requests.put(None)
        self._thread.join(timeout=60)
