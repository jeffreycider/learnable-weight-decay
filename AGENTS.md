# Training architecture

- Use `chisel.harness` for production training. It implements the lifecycle from Muoff commit `8f3e4a9f32249a506c8c929e06a3a0419b523800`.
- Specify experiments in YAML: constructor dependencies, hook ordering, architecture, optimizers, schedules, seeds, data, recording and evaluation. A YAML export of an argparse-driven experiment is insufficient.
- Extend injected components; do not add another standalone training loop. Historical trainers are retained only for comparison.
- Direct-transfer MuonH experiments follow `docs/direct-muonh.md`: no learned Chisel network or offline objective; differentiate the literal MuonH map with task gradients and incoming state frozen. Preview the complete successor on the next training batch, rewind data/RNG, and advance base optimizer state exactly once.
- Reuse `baseline.model.Block` for NanoGPT, CIFAR and Chisel. Preserve QK normalization and learned per-head scale in the selected CIFAR recipe. No runtime architecture monkeypatching.
- Chisel uses `W=R+C` and independently authors `delta_R` and `delta_C`. Offline training rewards group-average fractional reduction of effective `R+C` norm, constrained by actual prediction preservation. Redistribution alone earns no shrinkage reward. Online training uses direct next-step task credit through unchanged MuonH on R. Preserve MuonH's published scalar floating-point operations. No gradient ring.
- Checkpoint-private read/write probes and their optimizer state must actually train and persist.
- Keep each PR below 200 changed non-test Python lines, counting additions and deletions. Do not compress code to evade the limit.
- Run focused tests plus `tests/harness` integration tests. GPU numerical parity and compilation reuse require actual CUDA checks.
- Keep useful research outside the active tree on `archive/research-20261115`. Keep existing run artifacts intact. Never publish run artifacts or credentials just to establish a PR base.

- Checkpoint contents are specified in `docs/checkpoints.md`. Preserve raw gradients, replay state, source identities, and declared activation shards. Never call RMS summaries raw activations.
- PR30 is the explicit user-approved deletion-only exception to the 200-line limit; implementation PRs still count added and removed lines.
- Project memory and report-writing references live in `hippocampus/`. The conserving-transfer-only design is superseded; do not restore it as the offline objective.
