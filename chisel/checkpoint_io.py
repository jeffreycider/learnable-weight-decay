"""Bounded CPU checkpoint prefetch, with identity checks before any GPU mutation."""
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from pathlib import Path
import hashlib
import io
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
    def __init__(self, prefetch=True, pin_memory=False, cache_bytes=0, buffered=False):
        self.prefetch, self.pin_memory = prefetch, pin_memory
        self.buffered = buffered
        self.verified = set()
        self.cache, self.cache_bytes, self.resident_bytes = OrderedDict(), cache_bytes, 0

    def project(self, record):
        return record

    def read(self, entry):
        path = Path(entry['path'])
        before = path.stat()
        stamp = (str(path.resolve()), before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns, entry['sha256'])
        buffer = None
        if self.buffered and stamp not in self.cache:
            # Own and verify the same bytes used by torch.load. Avoid a second
            # storage read through mmap page faults when materializing tensors.
            buffer, digest = io.BytesIO(), hashlib.sha256()
            with path.open('rb') as source:
                for block in iter(lambda: source.read(8 << 20), b''):
                    digest.update(block)
                    buffer.write(block)
            if digest.hexdigest() != entry['sha256']:
                raise ValueError('offline checkpoint SHA256 mismatch')
            buffer.seek(0)
            self.verified.add(stamp)
        elif stamp not in self.verified:
            if sha256(path) != entry['sha256']:
                raise ValueError('offline checkpoint SHA256 mismatch')
            self.verified.add(stamp)
        if stamp in self.cache:
            record, size = self.cache.pop(stamp)
            self.cache[stamp] = (record, size)
            return record
        record = torch.load(buffer if self.buffered else path, map_location='cpu',
                            mmap=not self.buffered, weights_only=False)
        if buffer is not None:
            buffer.close()
        record = self.project(record)
        sizes = []
        map_tensors(record, lambda value: sizes.append(value.numel() * value.element_size()))
        size = sum(sizes)
        retain = self.cache_bytes > 0 and size <= self.cache_bytes
        if retain and not self.buffered:
            record = map_tensors(record, lambda value: value.clone())
        if self.pin_memory:
            record = map_tensors(record, lambda tensor: tensor.pin_memory())
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError('checkpoint changed while reading')
        if retain:
            while self.cache and self.resident_bytes + size > self.cache_bytes:
                self.resident_bytes -= self.cache.popitem(last=False)[1][1]
            self.cache[stamp] = (record, size)
            self.resident_bytes += size
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
