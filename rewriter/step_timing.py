"""Non-blocking per-phase timing for the offline Chisel trainer.

The production trainer brackets every phase with ``torch.cuda.synchronize``
(scripts/train_multistate_chisel.py ``timed``), roughly thirty full device
drains per checkpoint. This module offers the same ``timed(timings,name,fn,device)``
call signature with CUDA events instead: each phase records a start/end event
on the current stream and the elapsed device-stream seconds are added to the
timings dictionary at the next ``resolve`` (one synchronization per checkpoint
or per update, chosen by the caller).

Semantics differ deliberately: event timing reports device-stream elapsed time
between the two records on the device's current stream. That is not an additive
partition of host wall time: host work for one phase may overlap device work
still queued from the previous phase, gaps outside recorded regions are not
attributed, and nested timers overlap. Always report the independent whole-step
wall time (and a profiler GPU-busy fraction) next to these per-phase numbers.
Host launch seconds for each phase are recorded under ``<name>_host_seconds``
for diagnosis. Nothing here changes arithmetic, data, or schedule; only when the
host waits for the device. Not for use inside ``torch.compile`` regions.
"""
from __future__ import annotations
import time
import torch


def synchronous_timed(timings, name, fn, device):
    """The production reference: identical to train_multistate_chisel.timed."""
    if timings is None:
        return fn()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    value = fn()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    timings[name] = timings.get(name, 0.) + time.perf_counter() - started
    return value


def untimed(timings, name, fn, device):
    """No measurement at all; the benchmark arm that bounds the timing cost."""
    return fn()


class EventPhaseTimer:
    """Record phases with CUDA events; resolve them with one synchronization."""

    def __init__(self):
        self.pending = []

    def timed(self, timings, name, fn, device):
        if timings is None:
            return fn()
        if device.type != 'cuda':
            started = time.perf_counter()
            value = fn()
            timings[name] = timings.get(name, 0.) + time.perf_counter() - started
            return value
        stream = torch.cuda.current_stream(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        host_started = time.perf_counter()
        value = fn()
        host_seconds = time.perf_counter() - host_started
        end.record(stream)
        self.pending.append((timings, name, start, end, host_seconds))
        return value

    def resolve(self, device=None):
        """Synchronize once and account every pending phase. Call after the
        checkpoint's last launch (or at the end of the update)."""
        if not self.pending:
            return 0
        if device is not None:
            torch.cuda.synchronize(device)
        else:
            torch.cuda.synchronize()
        count = len(self.pending)
        for timings, name, start, end, host_seconds in self.pending:
            timings[name] = timings.get(name, 0.) + start.elapsed_time(end) / 1e3
            host_key = name[:-len('_seconds')] if name.endswith('_seconds') else name
            host_key += '_host_seconds'
            timings[host_key] = timings.get(host_key, 0.) + host_seconds
        self.pending.clear()
        return count


MODES = ('sync', 'event', 'none')


def install(module, mode):
    """Replace ``module.timed`` for the chosen mode; return (timer, restore).

    ``module`` is the imported trainer module whose functions reference the
    module-global ``timed``. The returned ``restore`` callable puts the original
    back; the timer is None except in event mode.
    """
    if mode not in MODES:
        raise ValueError('timing mode must be one of ' + ', '.join(MODES))
    original = module.timed
    timer = None
    if mode == 'sync':
        replacement = synchronous_timed
    elif mode == 'none':
        replacement = untimed
    else:
        timer = EventPhaseTimer()
        replacement = timer.timed
    module.timed = replacement

    def restore():
        module.timed = original
    return timer, restore
