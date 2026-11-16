# Private probe cache

Set `private_probes.cache_bytes` large enough for the checkpoint states revisited by each process. For FP32 Adam, each checkpoint needs approximately `24 * probe_rank * sum(rows + columns)` bytes for its read/write probe weights and Adam moments, plus scalar steps and metadata. A budget below the working set can force a disk write and reload on every visit.

Offline logs now include `private_cache_hits`, `private_disk_loads`, `private_disk_writes`, `private_cache_evictions`, and `private_cache_bytes`. Counters cover the current process lifetime and reset when a process restarts; distributed logs sum across checkpoint owners. Bytes count resident tensor payloads, excluding Python objects and temporary copies. A fresh probe initialization is neither a cache hit nor a disk load. Checkpoint flushes count as writes even without eviction.

The cache contains copied CPU tensors. Live optimizers receive independent state, and committed model checkpoints retain immutable private-state file pointers. Increasing the budget changes storage behavior only; weights, Adam steps, checkpoint sampling and objective remain unchanged. Use the existing hash-pinned resume compatibility mechanism to change the budget in a continuing run.

`private_probes.buffered_load: true` reads each private-state file into a temporary byte buffer, deserializes on CPU, and lets the probe module and Adam restore tensors to their devices. This avoids many small reads from shared storage and redundant early GPU allocations. The temporary buffer requires the serialized file size in host memory; the default remains the direct file loader. Both paths restore the same probe weights and Adam state.
