"""Optional bounded overlap of immutable private-state I/O with other compute.

No trainer enables this helper. The caller first finishes the private optimizer
step, then submit() owns and validates a complete CPU snapshot synchronously.
Only serialization, fsync and hashing run in a worker. No CUDA work or live
parameter/optimizer references reach that worker. finish() is mandatory before
another snapshot or transaction publication. Store close drains pending I/O.
At most one payload is retained. A failed write propagates and never publishes CURRENT.
The non-daemon worker must finish before interpreter exit; use the context manager.

Intended boundary: snapshot GPT, compute the distinct CNN visit, finish GPT,
snapshot CNN, then perform the existing joint commit. This does not overlap
optimizer updates of the same private owner or change checkpoint cadence.
"""
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

from .private_probe_store import PrivateProbeStore
from .distributed_probe_store import DistributedProbeStore


class AsyncPrivateSnapshot:
    def __init__(self, store):
        if isinstance(store, DistributedProbeStore):
            self.store = store.shard
            self._require_owner = store._require_owner
        elif isinstance(store, PrivateProbeStore):
            self.store = store
            self._require_owner = lambda identity: None
        else:
            raise TypeError('an explicit private probe store is required')
        self.store._writable()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='private-snapshot')
        self._future = None
        self._closed = False
        self.identity_token = None
        self.last_timings = {}

    def submit(self, identity, slot, optimizer, *, visits):
        if self._closed:
            raise RuntimeError('asynchronous snapshot writer is closed')
        if self._future is not None or self.store._async_snapshot_ticket is not None:
            raise RuntimeError('finish asynchronous private snapshot before submitting another')
        self._require_owner(identity)
        started = perf_counter()
        payload = self.store._prepare_slot_snapshot(identity, slot, optimizer, visits=visits)
        self.last_timings = {"prepare_seconds": perf_counter() - started}
        self.identity_token = identity.token
        self.store._async_snapshot_ticket = self
        try:
            self._future = self._executor.submit(self._write, identity, payload)
        except BaseException:
            self.store._async_snapshot_ticket = None
            self.identity_token = None
            raise

    def _write(self, identity, payload):
        started = perf_counter()
        reference = self.store._write_slot_snapshot(identity, payload)
        return reference, perf_counter() - started

    def finish(self):
        if self._future is None:
            raise RuntimeError('no asynchronous private snapshot to finish')
        started = perf_counter()
        try:
            reference, write_seconds = self._future.result()
            self.last_timings["write_seconds"] = write_seconds
            # Register only on the owning thread; the worker never changes store
            # transaction metadata or publishes a global checkpoint.
            self.store._pending[reference["token"]] = reference
            return reference
        finally:
            self.last_timings["finish_wait_seconds"] = perf_counter() - started
            self._future = None
            self.store._async_snapshot_ticket = None
            self.identity_token = None

    def close(self):
        if self._closed:
            return
        try:
            if self._future is not None:
                self.finish()
        finally:
            self._executor.shutdown(wait=True)
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except BaseException as cleanup_error:
            if exc is None:
                raise
            add_note = getattr(exc, "add_note", None)
            if callable(add_note):
                add_note(f'Asynchronous snapshot cleanup also failed: {cleanup_error!r}')
            else:
                if cleanup_error.__context__ is exc:
                    cleanup_error.__context__ = None
                exc.__context__ = cleanup_error
