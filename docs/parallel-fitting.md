# Parallel preservation fitting

`chisel.parallel_diagnostics` partitions one fixed checkpoint's preservation
examples equally across GPU replicas. It averages gradients of the shared Chisel
backbone and checkpoint-private probes before every L-BFGS closure returns.
The norm objective is replicated, then averaged, so it is counted once.
Every replica receives the same objective and gradients for its line search.

Set `distributed_diagnostic: true`, use a preservation example count divisible
by the YAML runtime world size, and add `parallel_diagnostics.setup` after
`diagnostic_lbfgs.setup`. Replace the diagnostic measurement hook with
`parallel_diagnostics.measure`. Launch through `torch.distributed.run`.
The total example count and optimizer settings need not change with GPU count.

Only rank zero measures the complete fitting batch and saves private probes.
Private-state file pointers are broadcast before ordinary actor checkpoints.
Use explicit staged initialization for subsequent runs; exact restart of this
cached diagnostic task is not implemented. This path supports Chisel, equal
example shards, and unclipped gradients. It is a fitting diagnostic, not online
NanoGPT training or a measurement of critical batch size.

For a fixed checkpoint cohort, set `distributed_diagnostic: true` and use
`cohort_diagnostics.setup` and `cohort_diagnostics.measure` directly. Checkpoints
are assigned evenly to replicas, with one shared backbone and independent probe
parameters. All replicas hold the same optimizer parameter order, average the
local objectives/gradients, and take identical L-BFGS steps. Checkpoint count must
be divisible by world size. Per-checkpoint measurements remain on rank zero.
