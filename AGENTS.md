# Training architecture

- Use `chisel.harness` for production training. It implements the lifecycle from Muoff commit `8f3e4a9f32249a506c8c929e06a3a0419b523800`.
- Specify experiments in YAML: constructor dependencies, hook ordering, architecture, optimizers, schedules, seeds, data, recording and evaluation. A YAML export of an argparse-driven experiment is insufficient.
- Extend injected components; do not add another standalone training loop. Historical trainers are retained only for comparison.
- Reuse `baseline.model.Block` for NanoGPT, CIFAR and Chisel. Preserve QK normalization and learned per-head scale in the selected CIFAR recipe. No runtime architecture monkeypatching.
- Chisel uses the MuonH consolidation specification: `W=R+C`, a conserving transfer before ordinary MuonH, then direct next-step task credit. Preserve MuonH's published scalar floating-point operations. No gradient ring.
- Checkpoint-private read/write probes and their optimizer state must actually train and persist.
- Keep each PR below 200 changed non-test Python lines, counting additions and deletions. Do not compress code to evade the limit.
- Run focused tests plus `tests/harness` integration tests. GPU numerical parity and compilation reuse require actual CUDA checks.
- Keep unrelated research files and existing runs intact. Never publish run artifacts or credentials just to establish a PR base.
