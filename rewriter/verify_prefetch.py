"""Verify-ahead worker: move first-visit SHA256/stamp verification off the critical path.

Observed on the repair fit's fresh node (native owner, 03:33 UTC): 224-254 s per
update versus ~67 s warm, ranks in D state with ~45% CPU. Per first visit of a
checkpoint the production readers do, on the consumer thread:
  * StateBankReader.read: SHA256 of the 665 MB state file (local disk);
  * TeacherCacheReader.read: SHA256 of the ~404 MB panel file (network 9p),
    then torch.load(mmap) and SHA256 of the inputs and of the FEATURES tensor
    again (a second full read of the panel bytes), before the selected blocks
    are read a third time for the microbatches.
That is ~1.5 GB hashed and ~1.5-2 GB read per checkpoint on the critical path
while the GPU waits; hashlib and file reads release the GIL, so this work can
run on a worker thread one or two checkpoints ahead with the SAME functions,
SAME comparisons and SAME failure behaviour. The readers' verify-once stamps are
filled by the worker; the consumer's ordinary read() then finds a matching stamp
and skips the hashes exactly as it does from the second visit on today.
Nothing about verification is weakened: a corrupted file raises in the worker
and the consumer re-raises that error when it reaches the checkpoint. No data
(weights, features, probabilities, encodings) is cached here.
"""
from __future__ import annotations
import json
import queue
import threading
import torch
from .state_bank import _digest
from .teacher_cache import file_hash, tensor_hash, FORMAT, SOFTCAP


def verify_state(reader, completed_steps):
    """Exactly StateBankReader.read's verification block, without loading."""
    from .multibank import MultiStateBankReader
    if isinstance(reader, MultiStateBankReader):
        seed, local = reader.route(completed_steps)
        return verify_state(reader.readers[seed], local)
    if not 1 <= completed_steps <= len(reader.entries):
        raise IndexError('state outside committed bank')
    entry = reader.entries[completed_steps - 1]
    path = reader.directory / entry['file']
    before = path.stat()
    stamp = (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    if not reader.verify_once:
        return False  # the reader will hash on every read anyway
    if reader._verified.get(completed_steps) == stamp:
        return False
    if _digest(path) != entry['sha256']:
        raise ValueError('state-bank record SHA256 mismatch')
    after = path.stat()
    if stamp != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError('state-bank record changed during verification')
    reader._verified[completed_steps] = stamp
    return True


def verify_panel(cache_reader, state_id, role):
    """Exactly TeacherCacheReader.read's verification (file hash, provenance,
    tensor hashes, stamp), without returning the panel."""
    from .multibank import MultiTeacherCacheReader
    if isinstance(cache_reader, MultiTeacherCacheReader):
        seed, local = cache_reader.states.route(state_id)
        return verify_panel(cache_reader.readers[seed], local, role)
    if role not in ('train', 'audit'):
        raise ValueError('invalid panel role')
    path = cache_reader.directory / f'state_{state_id:04d}_{role}.pt'
    info = json.loads(path.with_suffix('.json').read_text())
    if cache_reader.manifest is not None:
        expected = cache_reader.expected.get((state_id, role))
        if expected is None or any(info[key] != expected[key] for key in ('sha256', 'state_sha256', 'input_sha256', 'feature_sha256', 'sequence_ids')):
            raise ValueError('panel provenance differs from complete cache manifest')
    stat = path.stat()
    stamp = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if not cache_reader.verify_once or cache_reader._verified.get((state_id, role)) == (stamp, info['sha256']):
        return False
    if file_hash(path) != info['sha256']:
        raise ValueError('teacher panel file hash mismatch')
    value = torch.load(path, map_location='cpu', weights_only=True, mmap=cache_reader.mmap)
    metadata = value['metadata']
    if metadata['format'] != FORMAT or metadata['softcap'] != SOFTCAP:
        raise ValueError('unsupported target execution')
    if metadata['state_id'] != state_id or metadata['role'] != role:
        raise ValueError('panel identity differs')
    for key, name in [('inputs', 'input_sha256'), ('features', 'feature_sha256')]:
        if tensor_hash(value[key]) != metadata[name]:
            raise ValueError('panel tensor hash mismatch')
    after = path.stat()
    if stamp != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError('teacher panel changed during verification')
    cache_reader._verified[(state_id, role)] = (stamp, info['sha256'])
    return True


class VerifyAhead:
    """Pre-verify upcoming checkpoints (bank file + panel files) on a worker thread.

    Depth-limited so at most ``depth`` checkpoints are verified ahead of the
    consumer; ``wait(state_id)`` blocks until that checkpoint's verification
    finished and re-raises any error it produced. The consumer keeps calling
    the ordinary reader.read()/cache_reader.read(); with a filled stamp they do
    no hashing, so the arithmetic/data path is untouched.
    """

    def __init__(self, reader, cache_reader=None, *, roles=('train',), depth=2):
        if depth < 1:
            raise ValueError('positive depth required')
        self.reader, self.cache_reader, self.roles, self.depth = reader, cache_reader, tuple(roles), depth
        self._results = {}
        self._lock = threading.Condition()
        self._requests = queue.Queue()
        self._thread = threading.Thread(target=self._run, name='chisel-verify-ahead', daemon=True)
        self._thread.start()
        self.hashed = dict(states=0, panels=0)

    def _run(self):
        while True:
            state_id = self._requests.get()
            if state_id is None:
                return
            try:
                if verify_state(self.reader, state_id):
                    self.hashed['states'] += 1
                if self.cache_reader is not None:
                    for role in self.roles:
                        if verify_panel(self.cache_reader, state_id, role):
                            self.hashed['panels'] += 1
                outcome = None
            except BaseException as error:
                outcome = error
            with self._lock:
                self._results[state_id] = outcome
                self._lock.notify_all()

    def schedule(self, state_ids):
        self._pending = list(state_ids)
        self._issued = 0
        for _ in range(min(self.depth, len(self._pending))):
            self._issue()

    def _issue(self):
        if self._issued < len(self._pending):
            self._requests.put(self._pending[self._issued])
            self._issued += 1

    def wait(self, state_id):
        """Block until ``state_id`` is verified; then release one more slot."""
        with self._lock:
            while state_id not in self._results:
                self._lock.wait()
            outcome = self._results.pop(state_id)
        self._issue()
        if outcome is not None:
            raise outcome

    def close(self):
        self._requests.put(None)
        self._thread.join(timeout=60)
