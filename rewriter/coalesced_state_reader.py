"""State-bank reader that moves the packed observation in a few transfers.

``StateBankReader.read`` restores 892 observation leaves (production shapes)
one by one, each an individual pageable host->device copy. This opt-in
subclass performs the identical verification, load and clock checks, then
coalesces the packed CPU observation into one contiguous transfer per dtype
(the existing helper in scripts/benchmark_observation_transfer.py) before the
unchanged ``unpack_record`` restores every logical shape and stride on the
device. Values, strides and the verified-source record are identical; only the
number of host->device copies changes. No learned state is involved.
"""
from __future__ import annotations
from pathlib import Path
import sys
import torch
from .state_bank import StateBankReader, StateRecord, _digest
from .prefix_cache import unpack_record

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_observation_transfer import transfer_packed_observation  # noqa: E402


class CoalescedStateBankReader(StateBankReader):
    def read(self, completed_steps, *, device='cpu', model_device=None):
        if not 1 <= completed_steps <= len(self.entries):
            raise IndexError('state outside committed bank')
        entry = self.entries[completed_steps - 1]
        path = self.directory / entry['file']
        before = path.stat()
        stamp = (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        if not self.verify_once or self._verified.get(completed_steps) != stamp:
            if _digest(path) != entry['sha256']:
                raise ValueError('state-bank record SHA256 mismatch')
            after = path.stat()
            if stamp != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError('state-bank record changed during verification')
            self._verified[completed_steps] = stamp
        data = torch.load(path, map_location='cpu', weights_only=False, mmap=self.mmap_model)
        if data['completed_steps'] != completed_steps or data['data_cursor_batches'] != completed_steps:
            raise ValueError('state record clocks mismatch')
        packed = data['observation']
        if torch.device(device).type != 'cpu':
            packed = transfer_packed_observation(packed, device)
        obs = unpack_record(packed, device=device)
        if obs.step != completed_steps - 1:
            raise ValueError('sensor and model clocks mismatch')
        weights = {n: v.to(device if model_device is None else model_device) for n, v in data['model'].items()}
        source = dict(path=str(path.resolve()), sha256=entry['sha256'], file_stamp=stamp,
                      tensor_signature={n: (id(v), v._version) for n, v in weights.items()})
        return StateRecord(completed_steps, weights, obs.snapshot, obs.train, data['data_cursor_batches'], source)
