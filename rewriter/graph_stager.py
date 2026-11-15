"""Pinned, exact-stride host staging of the next checkpoint for a graph-replay loop.

Owner request (05:08 UTC): a bounded, reusable stager that overlaps verified
checkpoint + teacher-panel file work with CUDA-graph replay, preserving the bank
observation strides and failing closed on any integrity problem, without a
second binding path. The owner's ``NativeThinGradientGraph.load_checkpoint`` is
the ONLY binding entry: it verifies the record (``assert_verified`` inside
``load_engine``), checks the panel provenance, copies observations and teacher
blocks into its captured addresses with ``copy_``. Those ``copy_`` calls accept
host sources, so this stager hands over a RECORD WHOSE TENSORS LIVE IN PINNED
HOST MEMORY with the exact logical layout of the bank record:

  * model weights: pinned host copies (``verified_source.tensor_signature`` is
    re-issued for the copies so ``assert_verified`` still binds file stamp +
    tensor identity);
  * observation leaves (892 at production shapes): rebuilt in the WORKER thread
    into one pinned arena per dtype with ``as_strided`` views carrying the
    recorded shape/stride (column-major anchors, stride-0 broadcasts, storage
    offset 0 - the same layout ``prefix_cache._restore_layout`` produces); only
    the unique slice of broadcast axes is copied, gaps are zero;
  * panel: selected training blocks, features/inputs pinned.

The consumer thread does no copying at ``take``: it only checks order, integrity
and layout and hands the objects over. Arena lifetime: the owner's
``load_checkpoint`` issues its ``copy_`` calls with the default
``non_blocking=False`` (Astra review), so the consumer has already waited for
every H2D copy before the next ``take`` and reuse of a state's arena is trivially
safe. A future non_blocking path would additionally rely on the caching host
allocator's recorded-event lifetime for pinned blocks and must be stress-tested
(allocator churn while copies are outstanding) before use; nothing here claims
that test.

Capture vs rebind (owner/Astra review): host sources are valid for the EXISTING
graph's ``load_checkpoint`` only. The FIRST capture (and every post-audit
recapture) builds the actor's captured snapshot from the record it is given, so
it must see DEVICE observations with the reference layout - ``device_record``
below produces exactly that from a staged host record (one device arena per
dtype, exact strides, zeroed gaps), for capture only; per-state rebind keeps
using the pinned host record. The adapter's capture path therefore needs the
owner to call ``device_record`` (or capture from a normal GPU record, as the
owner plans) - this module does not plug into ``accumulate`` unconditionally.

Fail-closed rules: a worker exception, an out-of-order take, or an observation
layout signature different from the first staged state (or from
``expected_signature``) raises on the consumer thread and puts the stager into
``failed`` state; every later ``take`` raises until ``reset()`` is called
explicitly. Nothing learned is cached - only immutable bank/cache bytes.

Known costs that are NOT removed (Astra review of the design note): the owner's
``scoped_snapshot`` moves the editable norm to the record's device
(``edit_scope.py:61``), so a host ``weight_norm`` adds one D2H scalar sync per
bind. ``scalar_leaves_device`` places ``weight_norm``/``radius`` on that device
instead (a single IEEE multiply either way; the owner's GPU parity check decides
whether to use it). The owner's ``load_checkpoint`` still issues ~900 blocking
per-leaf ``copy_`` calls per bind; ``copy_arenas`` below is the optional path to
one copy per dtype when the captured observations were built with the same
arena layout (``device_record``).

Measured (owner, LOCAL, 4 states 1/997/998/999 x 64, coefficient 1, one H100,
thread start + first wait + drain inside the timer): separate runs graph
7.1855 -> 6.8803 s (-4.25%); matched alternating within one capture, 4 repeats
per mode, 7.0884 -> 6.7730 s (-4.45%); every qualification value identical to
the synchronous rebind. No FULL (8-rank, 200-state) number exists; an earlier
"0.38 s/state ceiling" was cross-workload arithmetic and is withdrawn.
"""
from __future__ import annotations
import dataclasses
import queue
import threading
import time
import torch
from .controller import ControllerSnapshot, MatrixObservation, VectorObservation
from .algebra import Anchors, Measurements
from .teacher_cache import TeacherPanel
from .state_prefetch import _unique_slice
from .static_snapshot_store import _leaves, layout_signature


class StagerError(RuntimeError):
    """Raised on the consumer thread; the stager is closed (failed) afterwards."""


def _storage_size(shape, stride):
    return 0 if any(d == 0 for d in shape) else 1 + sum((d - 1) * s for d, s in zip(shape, stride))


class PinnedLeafBuilder:
    """Rebuild exact-layout leaves inside one arena per dtype.

    ``device='cpu'`` with ``pin=True`` is the worker-side pinned host build;
    ``device='cuda:N'`` builds the same layout on the device (used only for the
    first capture / post-audit recapture via ``device_record``)."""

    def __init__(self, pin, device='cpu'):
        self.pin = bool(pin)
        self.device = torch.device(device)
        if self.pin and self.device.type != 'cpu':
            raise ValueError('pin_memory applies to host arenas only')

    @torch.no_grad()
    def build(self, snapshot, train, *, non_blocking=False):
        leaves = _leaves(snapshot, train)
        sizes = [_storage_size(tuple(t.shape), tuple(t.stride())) for _, t in leaves]
        # one arena per dtype, element offsets aligned to 64 elements (keeps 256 B alignment for FP32)
        offsets, totals = [], {}
        for (_, tensor), size in zip(leaves, sizes):
            start = totals.get(tensor.dtype, 0)
            offsets.append(start)
            totals[tensor.dtype] = start + ((size + 63) // 64) * 64
        # Whole arenas are zeroed explicitly (never trust allocator contents): in-footprint gaps AND
        # alignment padding read as zero, so an arena-level copy carries only defined bytes.
        arenas = {dtype: torch.empty(max(total, 1), dtype=dtype, device=self.device, pin_memory=self.pin).zero_()
                  for dtype, total in totals.items()}
        built = {}
        for (name, tensor), size, start in zip(leaves, sizes, offsets):
            shape, stride = tuple(tensor.shape), tuple(tensor.stride())
            region = arenas[tensor.dtype][start:start + size]
            unique = _unique_slice(tensor)
            target = region.as_strided(shape, stride)
            # copy into the unique-axis slice only: a stride-0 destination rejects a full copy_ (overlapping memory)
            _unique_slice(target).copy_(unique, non_blocking=non_blocking)
            built[name] = target
        return built, ArenaSet(arenas, layout_signature(snapshot, train), self.device)


@dataclasses.dataclass(frozen=True)
class ArenaSet:
    """Per-dtype arenas plus the exact layout signature that fixed their offsets."""
    arenas: dict
    signature: tuple
    device: torch.device

    def __iter__(self):
        return iter(self.arenas)

    def __getitem__(self, dtype):
        return self.arenas[dtype]

    def __len__(self):
        return len(self.arenas)

    def numel(self, dtype):
        return self.arenas[dtype].numel()


@torch.no_grad()
def device_record(record, device, *, non_blocking=False, return_arenas=False):
    """Same record with observations rebuilt on ``device`` in the reference layout.

    For the owner's FIRST capture and post-audit recapture only (the captured
    actor must hold device observations, graph_offline.py:49-58/88); per-state
    rebind of an existing graph takes the host record directly. Model tensors and
    ``verified_source`` are untouched, so ``assert_verified`` still holds.
    With ``return_arenas=True`` also returns ``{dtype: arena}``: if the graph is
    captured on these views, later rebinds can use ``copy_arenas`` (one copy per
    dtype) because the worker builds host arenas with the identical offsets for
    the identical layout signature.
    """
    built, arenas = PinnedLeafBuilder(False, device).build(record.snapshot, record.train, non_blocking=non_blocking)
    snapshot, train = _assemble(record.snapshot, record.train, built)
    # the as_strided views keep the arena storages alive; StateRecord is frozen, nothing else to attach
    out = dataclasses.replace(record, snapshot=snapshot, train=train)
    return (out, arenas) if return_arenas else out


@torch.no_grad()
def copy_arenas(destination, source, *, non_blocking=False):
    """Refill captured device observation arenas from a staged host record's arenas.

    ``destination``/``source`` are ``ArenaSet`` objects from ``device_record`` and
    ``GraphCheckpointStager.take(..., return_arenas=True)``. ALL metadata is
    validated before ANY byte moves (Astra review: a valid first dtype followed by
    an invalid one must not leave the destination half-written): the complete
    layout signatures (names, shapes, strides, dtypes - which fix the offsets)
    must be identical, dtype sets and arena sizes must match. Then one ``copy_``
    per dtype updates every ``as_strided`` view over the destination arenas at
    once instead of ~900 per-leaf copies. Zeroed host padding is copied too.

    Preconditions on the OWNER side (not enforceable here): the captured graph
    must hold the arena VIEWS themselves - ``NativeThinGradientGraph`` currently
    clones every supplied view (``clone_static_snapshot``), and a refill of the
    arenas does NOT reach such clones; the owner would build its static snapshot
    from the arenas or retain the aliases. A runtime failure inside ``copy_``
    must still be preceded by the owner's ``binding_valid=False``. The scoped
    ``weight_norm``/``radius`` are written AFTER this call, as ``load_checkpoint``
    does today.
    """
    if not isinstance(destination, ArenaSet) or not isinstance(source, ArenaSet):
        raise TypeError('copy_arenas needs ArenaSet objects (device_record / take return_arenas=True)')
    if destination.signature != source.signature:
        raise ValueError('observation layout signature differs from the captured arenas; '
                         'reject this checkpoint or capture a separate graph')
    if set(destination) != set(source):
        raise ValueError('arena dtype sets differ')
    for dtype in destination:
        target, arena = destination[dtype], source[dtype]
        if target.numel() != arena.numel() or target.dtype != dtype or arena.dtype != dtype:
            raise ValueError(f'arena size/dtype mismatch for {dtype}: {target.numel()} vs {arena.numel()}')
    for dtype in destination:                                   # nothing above has written anything
        destination[dtype].copy_(source[dtype], non_blocking=non_blocking)


def _assemble(template_snapshot, template_train, built):
    matrices, vectors, train_m, train_v = {}, {}, {}, {}
    for name, obs in template_snapshot.matrices.items():
        matrices[name] = MatrixObservation(Anchors(built[f'{name}.anchors.A'], built[f'{name}.anchors.B']),
                                           built[f'{name}.rows'], built[f'{name}.columns'],
                                           tuple(built[f'{name}.weight_probes.{i}'] for i in range(len(obs.weight_probes))))
        train_m[name] = tuple(built[f'{name}.train.{i}'] for i in range(len(template_train.matrices[name])))
    for name in template_snapshot.vectors:
        vectors[name] = VectorObservation(built[f'{name}.coordinates'], built[f'{name}.weight'])
        train_v[name] = built[f'{name}.train']
    return (ControllerSnapshot(matrices, vectors, built['trajectories'], built['weight_norm'], built['radius']),
            Measurements(train_m, train_v))


@dataclasses.dataclass
class _Item:
    state_id: int
    record: object = None
    panel: object = None
    arenas: dict | None = None
    signature: tuple | None = None
    seconds: dict = dataclasses.field(default_factory=dict)
    error: BaseException | None = None


class GraphCheckpointStager:
    """Depth-one worker producing pinned exact-layout host records for ``load_checkpoint``."""

    def __init__(self, reader, cache_reader, *, select_panel=None, role='train', pin=None,
                 expected_signature=None, scalar_leaves_device=None, fp32_parameter_names=None):
        """``fp32_parameter_names`` (the strict FP32 destination engine's exact
        ``engine.parameters`` names, nothing heuristic) promotes exactly those
        parameters to FP32 in the WORKER via Astra's
        ``rewriter.teacher_model_staging.stage_fp32_teacher_model`` (9793cb8):
        the bank stores the embedding in bfloat16 (baseline/model.py:102) and the
        owner's bind profile (06:10) showed ``load_teacher_state``'s per-state
        ``.float()`` as one real host upcast plus 172 no-ops (~26 ms) followed by
        ~650 MB of pageable copies. The promotion is exact, written straight into
        the pinned destination, buffers keep their checkpoint dtype, and the usual
        copied-tensor signature is re-issued afterwards. Valid for
        ``gpt_precision='fp32'`` engines only (they perform the same upcast on
        load); leave ``None`` to hand over the bank dtypes unchanged."""
        self.reader = reader
        self.cache_reader = cache_reader
        self.select_panel = select_panel
        self.role = role
        self.fp32_parameter_names = None if fp32_parameter_names is None else frozenset(fp32_parameter_names)
        if self.fp32_parameter_names is not None and not self.fp32_parameter_names:
            raise ValueError('fp32_parameter_names must be the engine parameter names, not empty')
        self.pin = torch.cuda.is_available() if pin is None else bool(pin)
        self.builder = PinnedLeafBuilder(self.pin)
        self.signature = tuple(expected_signature) if expected_signature is not None else None
        self.scalar_leaves_device = torch.device(scalar_leaves_device) if scalar_leaves_device is not None else None
        self.failed = None            # exception that closed the stager, or None
        self.consumed = 0
        self.wait_seconds = 0.
        self._pending, self._issued = [], 0
        self._outstanding = 0         # issued to the worker but not yet received by take/reset
        self._queue = queue.Queue(maxsize=1)
        self._requests = queue.Queue()
        self._previous = None         # keep the previous state's arenas referenced for one more take
        self._thread = threading.Thread(target=self._run, name='chisel-graph-stager', daemon=True)
        self._thread.start()

    # ---- worker ------------------------------------------------------------
    @torch.no_grad()
    def _stage(self, state_id):
        seconds = {}
        started = time.perf_counter()
        record = self.reader.read(state_id, device='cpu', model_device='cpu')   # SHA/stamp verified by the reader
        seconds['bank_read'] = time.perf_counter() - started
        started = time.perf_counter()
        if self.fp32_parameter_names is not None:
            from .teacher_model_staging import stage_fp32_teacher_model
            # record was SHA/stamp-verified by the reader above; promotion is exact and lands
            # directly in the (pinned) destination; buffers keep their dtype
            model = stage_fp32_teacher_model({n: v.detach() for n, v in record.model.items()},
                                             parameter_names=self.fp32_parameter_names, pin_memory=self.pin)
        else:
            model = {}
            for name, value in record.model.items():
                value = value.detach()
                model[name] = value.pin_memory() if self.pin else value.clone()
        source = dict(record.verified_source)
        source['tensor_signature'] = {n: (id(v), v._version) for n, v in model.items()}
        seconds['model_copy'] = time.perf_counter() - started
        started = time.perf_counter()
        signature = layout_signature(record.snapshot, record.train)
        built, arenas = self.builder.build(record.snapshot, record.train)
        if self.scalar_leaves_device is not None:
            for name in ('weight_norm', 'radius'):
                built[name] = built[name].to(self.scalar_leaves_device).as_strided(built[name].shape, built[name].stride())
        snapshot, train = _assemble(record.snapshot, record.train, built)
        seconds['observation_build'] = time.perf_counter() - started
        record = dataclasses.replace(record, model=model, snapshot=snapshot, train=train, verified_source=source)
        panel = None
        if self.cache_reader is not None:
            started = time.perf_counter()
            panel = self.cache_reader.read(state_id, role=self.role)
            if self.select_panel is not None:
                panel = self.select_panel(panel, state_id)
            features, inputs = panel.features, panel.inputs
            if self.pin:
                features = features if features.is_pinned() else features.pin_memory()
                inputs = inputs if inputs.is_pinned() else inputs.pin_memory()
            else:
                features, inputs = features.clone(), inputs.clone()
            panel = TeacherPanel(inputs, features, panel.sequence_ids, panel.metadata)
            seconds['panel_stage'] = time.perf_counter() - started
        return _Item(state_id, record, panel, arenas, signature, seconds)

    def _run(self):
        while True:
            state_id = self._requests.get()
            if state_id is None:
                return
            try:
                item = self._stage(state_id)
            except BaseException as error:        # surfaced on the consumer thread, never swallowed
                item = _Item(state_id, error=error)
            self._queue.put(item)

    # ---- consumer ----------------------------------------------------------
    def schedule(self, state_ids):
        """Declare the next sequence of states. Rejected while a previous sequence
        still has issued-but-untaken or not-yet-issued states (owner review: an
        unconditional overwrite could misroute queued work); finish it or reset()."""
        if self.failed is not None:
            raise StagerError('stager failed; call reset() before scheduling') from self.failed
        if self._outstanding > 0 or self._issued < len(self._pending):
            raise StagerError(f'previous sequence still active ({self._outstanding} staged/in flight, '
                              f'{len(self._pending) - self._issued} not issued); take them or reset() first')
        self._pending, self._issued = list(state_ids), 0     # repeats are fine: items are matched in order
        self._issue()

    def _issue(self):
        if self._issued < len(self._pending):
            self._requests.put(self._pending[self._issued])
            self._issued += 1
            self._outstanding += 1

    def _fail(self, error):
        self.failed = error
        raise error

    def take(self, state_id, *, return_arenas=False):
        """Return (record, panel, seconds[, arenas]) for ``state_id`` or raise and fail closed.

        The record's model/observation tensors are pinned host tensors with the
        bank's exact strides; pass it straight to the owner's ``load_checkpoint``
        (``cache_reader.load_engine`` re-asserts verification on the consumer).
        ``return_arenas=True`` additionally returns the ``{dtype: pinned arena}``
        backing the observation views, for ``copy_arenas``.
        """
        if self.failed is not None:
            raise StagerError('stager failed; call reset()') from self.failed
        started = time.perf_counter()
        item = self._queue.get()
        self._outstanding -= 1
        wait = time.perf_counter() - started
        self.wait_seconds += wait
        self._issue()                                       # start the next state before any consumer work
        if item.error is not None:
            self._fail(StagerError(f'staging state {item.state_id} failed: {item.error!r}'))
        if item.state_id != state_id:
            self._fail(StagerError(f'order mismatch: staged {item.state_id}, requested {state_id}'))
        if self.signature is None:
            self.signature = item.signature
        elif item.signature != self.signature:
            self._fail(StagerError(f'observation layout of state {state_id} differs from the captured layout; '
                                   'reject this checkpoint or capture a separate graph'))
        try:
            item.record.assert_verified()                    # file stamp + pinned-copy identity, before handing over
        except ValueError as error:
            self._fail(StagerError(f'state {state_id} lost verification during staging: {error}'))
        item.seconds['wait_for_worker'] = wait
        self._previous = item.arenas                         # previous arenas stay referenced across one take
        self.consumed += 1
        if return_arenas:
            return item.record, item.panel, item.seconds, item.arenas
        return item.record, item.panel, item.seconds

    def reset(self):
        """Explicitly clear a failure and drop every issued item, including one the
        worker is still building (waits for it, so no stale state can surface later)."""
        self.failed = None
        while self._outstanding > 0:
            self._queue.get(timeout=600)       # the worker always puts exactly one item per request
            self._outstanding -= 1
        self._pending, self._issued = [], 0

    def close(self, timeout=60):
        """Discard outstanding work, stop the worker and verify it exited.

        Raises ``StagerError`` if the worker is still alive after ``timeout``
        seconds (never a silent live thread). Safe to call after a failed take,
        which may already have issued the next state."""
        self._pending, self._issued = [], 0     # nothing further is issued
        try:
            while self._outstanding > 0:        # drain items the worker has built or is building
                self._queue.get(timeout=timeout)
                self._outstanding -= 1
        except queue.Empty as error:
            raise StagerError('worker did not deliver outstanding items before close timeout') from error
        self._requests.put(None)
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            raise StagerError('stager worker thread still alive after close timeout')
