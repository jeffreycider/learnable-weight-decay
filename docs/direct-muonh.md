# Direct hypergradient transfers

**The forward pass uses `R+C`; the ordinary MuonH optimizer updates `R`.**
The injected `DirectTransfer` optimizer owns the split and checkpoints both
matrices. It calls the existing `chisel.muonh.sphere` for execution and autodiff.
The harness loop, model architecture and ordinary optimizer settings are unchanged.

For the current task gradient and incoming optimizer state held fixed, let
\(U_t(r)\) be the actual MuonH successor. At the complete unedited successor,
the next training batch supplies \(q\). The edit is

\[
h=D_R U_t(R)^\top q-q,\qquad
s=-a_t\frac{\|U_t(R)-R\|}{\|h\|}h,\qquad
(R,C)\leftarrow\big(U_t(R+s),C-s\big).
\]

All norms above combine the eligible MuonH matrices. Other parameters receive
their ordinary updates before the lookahead. Muon momentum and Adam state advance
once. The implementation temporarily installs the unedited successor in the same
model, avoiding a second model/optimizer copy. The next ordinary step recomputes
its gradient on the same training batch at the edited successor.

**The lookahead preserves the ordinary data stream and RNG state.** It uses the
same accumulated batch, precision, loss reduction and distributed gradient
reduction as ordinary training. Data cursors, RNG states and module buffers are
restored afterward. CIFAR previews replay the deterministic augmentations without
restarting prefetch workers. Validation provides no edit gradients.

| Experiment | YAML |
|---|---|
| Ordinary MuonH, 40 GB starting configuration | `recipes/nanogpt_muonh_40gb.yaml` |
| Zero edit, including lookahead computation | `recipes/nanogpt_direct_zero.yaml` |
| Fixed edit/base-step norm ratio of 0.1 | `recipes/nanogpt_direct_transfer.yaml` |
| Full-size model, 20-step memory/throughput check | `recipes/nanogpt_direct_smoke.yaml` |
| CIFAR with the same injected method | `recipes/cifar_direct_transfer.yaml` |

```bash
torchrun --standalone --nproc_per_node=8 -m chisel.harness recipes/nanogpt_direct_smoke.yaml
```

**The NanoGPT recipes retain 524,288 tokens per ordinary iteration.** Eight
sequences per microbatch means eight accumulated microbatches per rank on eight
GPUs. Every lookahead adds that same amount of model computation. The fixed-size
recipe also edits the final step, using the next training batch for lookahead;
that extra data exposure and backward are included in the counters.

| Tuning control | Configuration |
|---|---|
| Transfer size, start, ramp, interval | `direct_transfer.schedule.kwargs` |
| Ordinary MuonH LR | `optimizers.hidden.optimizer.lr` |
| Initial hidden-matrix radii relative to baseline | `hyperball_radius_multiplier` |
| Start from an ordinary harness checkpoint | `baseline_checkpoint: .../checkpoint_001000_rank{rank}.pt` |
| Resume a direct-transfer run | `resume: .../latest_rank{rank}.pt` |

**Checkpoint branching starts with `R=W, C=0` and preserves optimizer history.**
Branching validates the model, data, seeds, optimizer algorithms and source hashes,
then records the source checkpoint hash and configuration. LR can change at the
branch; radius tuning requires a fresh run. Resuming a direct-transfer run restores
`R`, `C`, optimizer state, RNG, data position and computation counters.

**Numerically vanishing hypergradients are skipped.** The initial cutoff is
`max(1e-20, 1e-6 * ||q||)` in the configured gradient units, to avoid normalizing
FP32 cancellation noise in `VJP - q`. These numerical settings are explicit YAML
parameters. No norm-shrinkage term or learned auxiliary model is present.

**Compare validation curves against both ordinary iterations and total model
computation.** `model_training_tokens` and `model_backward_microbatches` include
lookahead work; direct-transfer counters start at the branch when branching.
`effective_norm_relative_change` measures `(edited_norm - unedited_norm) /
unedited_norm` over eligible effective matrices at each step. Negative means
the edit reduced effective norm relative to the ordinary successor. Logs also
include transfer norm, base displacement norm, consolidated norm, runtime and
CUDA peak allocation. Numerical correctness does not establish a training gain.
