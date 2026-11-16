# Reuse normalized immutable sensors

`cache_normalized_sensors: true` computes each detached matrix bank's RMS normalization and each activity bank's normalization once when loading a checkpoint snapshot. Chisel's private read/write probes remain trainable and are applied again on every optimization step. The cache stores normalized tensors, not projected observations or learned outputs.

This trades extra GPU storage for fewer repeated dense reductions and divisions. It is opt-in until a workload benchmark justifies the storage cost. Refresh the entire cache whenever the underlying snapshot changes; the replay loader creates it from a new immutable snapshot on each load.
