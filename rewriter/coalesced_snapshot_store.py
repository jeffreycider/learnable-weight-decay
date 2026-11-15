"""Bounded, exact-layout CPU staging into one static device bank per dtype.

Packing happens on the host and may run ahead of the current GPU computation.
The consumer performs one bank copy per dtype; it does not allocate or zero
hundreds of device leaves. Every leaf keeps its logical values and strides.
Bank slices start at 256-byte boundaries; storage offsets differ from the
ordinary per-leaf allocator, so actual CUDA output/VJP qualification is still
required before adoption. No learned feature or dense model bank is cached.
"""
from dataclasses import dataclass, replace
from concurrent.futures import ThreadPoolExecutor
import torch
from .static_snapshot_store import StaticSnapshotStore, _leaves, _unique_view, layout_signature


@dataclass(frozen=True)
class CoalescedSnapshot:
    signature: tuple
    offsets: dict
    banks: dict


def stage_coalesced_snapshot(snapshot, train, *, pin=False):
    leaves = _leaves(snapshot, train)
    if any(value.device.type != 'cpu' for _, value in leaves):
        raise ValueError('coalesced staging requires CPU observation sources')
    signature = layout_signature(snapshot, train)
    offsets, lengths = {}, {}
    for name, value in leaves:
        alignment = 256 // value.element_size()
        offset = ((lengths.get(value.dtype, 0) + alignment - 1) // alignment) * alignment
        size = 0 if any(d == 0 for d in value.shape) else 1 + sum((d - 1) * s for d, s in zip(value.shape, value.stride()))
        offsets[name] = offset
        lengths[value.dtype] = offset + size
    banks = {dtype:torch.zeros(length, dtype=dtype, pin_memory=pin) for dtype, length in lengths.items()}
    for name, value in leaves:
        target = banks[value.dtype].as_strided(value.shape, value.stride(), offsets[name])
        _unique_view(target).copy_(_unique_view(value))
    return CoalescedSnapshot(signature, offsets, banks)


class CoalescedSnapshotStore(StaticSnapshotStore):
    """One captured layout, refillable with a prepacked immutable CPU snapshot.

    Source owners survive the copy; an owner may stage the next CPU snapshot
    while this store is in use, but must finish all consumers before refilling.
    The current stream owns both copies and subsequent actor execution.
    """
    def __init__(self, staged, *, device):
        self.device = torch.device(device)
        self.signature = staged.signature
        self.offsets = dict(staged.offsets)
        self.banks = {dtype:torch.empty_like(value, device=self.device) for dtype, value in staged.banks.items()}
        self._buffers = {name:self.banks[dtype].as_strided(shape, stride, self.offsets[name])
                         for name, shape, stride, dtype in self.signature}
        self.generation = 0
        self.bound = False
        self.refill_staged(staged)

    @torch.no_grad()
    def refill_staged(self, staged, *, non_blocking=False):
        self.bound = False
        if staged.signature != self.signature or staged.offsets != self.offsets or staged.banks.keys() != self.banks.keys():
            raise ValueError('coalesced observation layout changed')
        for dtype, value in staged.banks.items():
            if value.device.type != 'cpu' or value.dtype != dtype or value.shape != self.banks[dtype].shape or not value.is_contiguous():
                raise ValueError('coalesced bank shape/dtype/device changed')
            if non_blocking and self.device.type == 'cuda' and not value.is_pinned():
                raise ValueError('asynchronous coalesced copies require pinned banks')
        for dtype, value in staged.banks.items():
            self.banks[dtype].copy_(value, non_blocking=non_blocking)
        self._source_owner = staged
        self.generation += 1
        self.bound = True

    def refill(self, snapshot, train, *, non_blocking=False):
        self.refill_staged(stage_coalesced_snapshot(snapshot, train, pin=non_blocking and self.device.type == 'cuda'), non_blocking=non_blocking)


class BankedObservationReader:
    """Opt-in reader adapter, retaining one static observation layout per device.

    Finish the current state's actor backward before another read. Model weights
    remain the original verified CPU record; only observations are coalesced.
    Local GPT clocks, global training IDs, seeds and verification owners survive
    unchanged. Teacher loading/binding is still the caller's ordinary operation.
    """
    def __init__(self, reader):
        self.reader = reader
        self.stores = {}
        self._executor = None
        self._pending = None
        self._closed = False

    def __getattr__(self, name):
        return getattr(self.reader, name)

    def read(self, state_id, *, device='cpu', model_device=None):
        if self._closed:
            raise RuntimeError('banked reader is closed')
        device = torch.device(device)
        if device.type == 'cpu':
            if self._pending is not None:
                raise RuntimeError('cannot read another CPU state while prefetch owns the reader')
            return self.reader.read(state_id, device=device, model_device=model_device)
        if model_device is None or torch.device(model_device).type != 'cpu':
            raise ValueError('banked observation reader requires CPU model weights')
        record, staged = self._take(state_id, pin=True)
        key = str(device)
        if key not in self.stores:
            self.stores[key] = CoalescedSnapshotStore(staged, device=device)
        else:
            self.stores[key].refill_staged(staged, non_blocking=True)
        snapshot, train = self.stores[key].snapshot(record.snapshot, record.train)
        return replace(record, snapshot=snapshot, train=train)

    def _prepare(self, state_id, pin):
        record = self.reader.read(state_id, device='cpu', model_device='cpu')
        return record, stage_coalesced_snapshot(record.snapshot, record.train, pin=pin)

    def prefetch(self, state_id, *, pin=True):
        """Stage exactly one declared next state, without CUDA tensor work.

        The caller supplies its actual sampler order; this helper never guesses
        future training IDs. While a job is pending it is the reader's exclusive
        owner, so verify-once dictionaries cannot race a synchronous read.
        """
        if self._closed or self._pending is not None:
            raise RuntimeError('prefetch requires an open reader with no pending state')
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='chisel-banked-state')
        self._pending = (state_id, pin, self._executor.submit(self._prepare, state_id, pin))

    def _take(self, state_id, pin):
        if self._pending is None:
            return self._prepare(state_id, pin)
        expected, pinned, future = self._pending
        if expected != state_id or pinned != pin:
            raise ValueError('prefetched state or pinning differs from requested state')
        self._pending = None
        return future.result()

    def close(self):
        self._closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
        self._pending = None
