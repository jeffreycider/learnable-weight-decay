# Keep private probe state in memory between shared checkpoints

`private_probes.cache_bytes` enables a bounded CPU cache of checkpoint-private probe weights and Adam state. Updates remain private to their source checkpoint. Cache hits avoid shared-filesystem reloads and per-visit writes; an eviction writes the current private state before discarding it.

Creating an actor checkpoint flushes every dirty private entry before recording its file pointer. Distributed audits also flush every owner before sharing pointers. Older actor checkpoints continue to reference immutable older private files. A crash can lose updates after the last actor checkpoint, just as it can without this cache; it cannot silently pair that checkpoint with newer private weights.

The default is zero bytes, preserving immediate per-visit persistence. Cached tensors are copied on load so an optimizer cannot mutate a saved cache entry before `save`.
