"""Bounded device-resident private probes; durable snapshots still own CPU copies."""
from collections import OrderedDict
from chisel.offline import ProbeStore
from chisel.checkpoint_io import map_tensors


class DeviceProbeStore(ProbeStore):
    def __init__(self, root, config, device, cache_bytes):
        super().__init__(root, config, device)
        if cache_bytes < 0:
            raise ValueError('device cache budget must be nonnegative')
        self.device_budget = cache_bytes
        self.device_cache, self.device_bytes = OrderedDict(), 0
        self.device_hits = self.device_evictions = 0

    def load(self, identity, snapshot, reference=None, actor=None):
        if identity not in self.device_cache:
            return super().load(identity, snapshot, reference, actor)
        entry = self.device_cache.pop(identity)
        self.device_cache[identity] = entry
        self.device_hits += 1
        probes, optimizer = entry[:2]
        # A reconstructed module has no gradients; preserve that load contract.
        optimizer.zero_grad(set_to_none=True)
        return probes, optimizer

    @staticmethod
    def size(probes, optimizer):
        sizes = []
        tensors = list(probes.parameters())
        tensors += [p.grad for p in probes.parameters() if p.grad is not None]
        map_tensors((tensors, list(optimizer.state.values())),
                    lambda value: sizes.append(value.numel() * value.element_size()))
        return sum(sizes)

    def save(self, identity, step, probes, optimizer):
        size = self.size(probes, optimizer)
        if identity in self.device_cache:
            self.device_bytes -= self.device_cache.pop(identity)[3]
        if not self.device_budget or size > self.device_budget:
            return super().save(identity, step, probes, optimizer)
        # A CPU copy from an earlier visit must never shadow this updated state.
        if identity in self.cache:
            self.resident_bytes -= self.cache.pop(identity)[2]
        while self.device_cache and self.device_bytes + size > self.device_budget:
            oldest, (old_probes, old_optimizer, old_step, old_size) = self.device_cache.popitem(last=False)
            if old_step is not None:
                super().save(oldest, old_step, old_probes, old_optimizer)
            self.device_bytes -= old_size
            self.device_evictions += 1
        self.device_cache[identity] = (probes, optimizer, step, size)
        self.device_bytes += size

    def flush_identity(self, identity):
        if identity not in self.device_cache:
            return super().flush_identity(identity)
        probes, optimizer, step, size = self.device_cache[identity]
        if step is not None:
            from chisel.recording import cpu_copy
            self.persist(identity, cpu_copy(dict(probes=probes.state_dict(), optimizer=optimizer.state_dict())), step)
            self.device_cache[identity] = (probes, optimizer, None, size)

    def flush(self):
        super().flush()
        for identity in self.device_cache:
            self.flush_identity(identity)

    def load_state_dict(self, saved):
        super().load_state_dict(saved)
        self.device_cache.clear()
        self.device_bytes = 0

    def cache_metrics(self):
        return super().cache_metrics() | dict(private_device_cache_hits=self.device_hits,
            private_device_cache_evictions=self.device_evictions, private_device_cache_bytes=self.device_bytes)
