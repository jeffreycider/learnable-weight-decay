"""Bounded CPU checkpoint prefetch, with identity checks before any GPU mutation."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import torch
from chisel.storage import sha256


def map_tensors(value, function):
    if torch.is_tensor(value):
        return function(value)
    if isinstance(value, dict):
        return {key: map_tensors(item, function) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(map_tensors(item, function) for item in value)
    return value


class CheckpointReader:
    def __init__(self, prefetch=True, pin_memory=False):
        self.prefetch, self.pin_memory = prefetch, pin_memory
        self.verified = set()

    def read(self, entry):
        path = Path(entry['path'])
        before = path.stat()
        stamp = (str(path.resolve()), before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns, entry['sha256'])
        if stamp not in self.verified:
            if sha256(path) != entry['sha256']:
                raise ValueError('offline checkpoint SHA256 mismatch')
            self.verified.add(stamp)
        record = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        if self.pin_memory:
            record = map_tensors(record, lambda tensor: tensor.pin_memory())
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError('checkpoint changed while reading')
        return record

    def iterate(self, entries):
        if not self.prefetch:
            for entry in entries:
                yield entry, self.read(entry)
            return
        with ThreadPoolExecutor(1) as pool:
            iterator = iter(entries)
            entry = next(iterator, None)
            pending = pool.submit(self.read, entry) if entry is not None else None
            while entry is not None:
                record = pending.result()
                following = next(iterator, None)
                pending = pool.submit(self.read, following) if following is not None else None
                yield entry, record
                entry = following
