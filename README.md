# Chisel on MuonH

**One YAML recipe drives one shared training loop.** Model construction, data, optimizers, schedules, observations, capture, offline fitting and online hypergradients are injected components. The lifecycle follows [the requested Muoff commit](https://github.com/jeffreycider/muoff/blob/8f3e4a9f32249a506c8c929e06a3a0419b523800/records/track_3_optimization/harness/loop.py).

| Task | Recipe |
|---|---|
| CIFAR classification with NanoGPT blocks | [cifar.yaml](recipes/cifar.yaml) |
| Published PR267 MuonH text configuration | [nanogpt.yaml](recipes/nanogpt.yaml) |
| Capture CIFAR / text training states | [cifar_capture.yaml](recipes/cifar_capture.yaml) / [nanogpt_capture.yaml](recipes/nanogpt_capture.yaml) |
| Offline Chisel fitting | [offline.yaml](recipes/offline.yaml) |
| Online Chisel | [cifar_online.yaml](recipes/cifar_online.yaml) / [nanogpt_online.yaml](recipes/nanogpt_online.yaml) |

Run from this checkout:

```bash
git submodule update --init --recursive
pip install -e '.[test,cifar]'
python -m chisel.harness recipes/cifar_capture.yaml
# Text configuration uses eight ranks:
torchrun --standalone --nproc_per_node=8 -m chisel.harness recipes/nanogpt_capture.yaml
```

**Edit YAML to change an experiment.** Relative `extends` merges mappings and replaces lists. A dependency's `_target_` names its constructor; lifecycle entries name hook factories. The CLI accepts only a recipe path. Paths, seeds and output directories must be set for the run. JSONL contains measurements, not configuration.

**CIFAR retains QK normalization and learns one scale per head.** Width, depth, head size, MLP expansion, attention mask and scale are constructor arguments. The image interface changes only patch inputs and classification readout. The pinned augmentation dependency lives in a Git submodule, not a run directory.

**The new Chisel implements consolidation, not the retired shrinkage/KL experiment.** Each hidden matrix is `W = R + C`. Chisel transfers `D` through `R ← R − D, C ← C + D`, then the unchanged MuonH map updates `R`. A global transfer multiplier keeps raw radii above the configured floor. The next ordinary backward supplies the task cotangent; Chisel differentiates through the saved transfer and MuonH algebra. No additional GPT evaluation occurs online. The terminal iteration consumes pending credit and makes no new transfer.

**Checkpoint-private probes actually learn.** A shared NanoGPT transformer processes all matrix summaries. Each saved GPT state has its own trainable read/write maps and Adam state. Offline batches accumulate shared-backbone gradients across checkpoints before one Adam update; each visited checkpoint's private maps are loaded, updated and saved. Offline edited-GPT forward, loss and backward are compiled together; the shared Chisel forward is compiled separately. Known architecture variants may compile separately, but changing checkpoint values must reuse their graphs.

**Offline task supervision uses the next batch after the saved base state.** It replays the complete base optimizer proposal, substitutes the Chisel-dependent hidden matrices, and differentiates next-batch loss plus the consolidated-weight penalty. This permits extra GPT evaluations offline. It is not a claim of NanoGPT benchmark compliance or improved learning.

**Capture stores replayable transitions without duplicating CIFAR images.** Records contain weights, raw gradients, optimizer state, activity, resolved configuration, base-source hashes and the next data cursor. CIFAR augmentation and mixing are determined by explicit seeds and the step; FineWeb uses its shard cursor. Receipts contain file size and SHA256. Dataset files must remain available at their recorded identities. Terminal records say whether a subsequent prescribed training step exists.

**Offline training and validation manifests must contain disjoint checkpoint hashes.** A manifest is a YAML mapping with `states`, each containing `path`, `sha256` and `domain`. Relative paths resolve against the manifest. Validation visits every listed checkpoint without optimizer updates. Unseen checkpoints use deterministic, unfitted private probes; this metric does not establish performance after support fitting. Validation is logged per domain and never enters the online update.

**Restart includes pending credit and private fitting state.** Numbered checkpoints are retained, with `latest_rank{rank}.pt` pointing to the newest one. Set `resume` to that file; a shorter `stop_step` leaves the original learning-rate schedule intact. Offline checkpoints commit pointers to immutable private-state versions, so retain the `private/` directory with the shared checkpoint. Source/config changes are rejected on resume. Runs save resolved YAML, source archives, package versions and RNG state.

## Review ladder

Each draft PR targets its immediate predecessor. Counts include added **and** deleted non-test Python lines. Some later rungs contain corrections found by integration/GPU tests; review the complete stack before deployment.

| PR | Change | Python lines |
|---|---|---:|
| [1](https://github.com/jeffreycider/learnable-weight-decay/pull/1) | YAML construction and Muoff lifecycle | 108 |
| [2](https://github.com/jeffreycider/learnable-weight-decay/pull/2) | Architecture arguments and CIFAR interface | 79 |
| [3](https://github.com/jeffreycider/learnable-weight-decay/pull/3) | MuonH component; CUDA arithmetic corrected in 18 | 99 |
| [4](https://github.com/jeffreycider/learnable-weight-decay/pull/4) | Supervised task and optimizer partitions | 136 |
| [5](https://github.com/jeffreycider/learnable-weight-decay/pull/5) | Deterministic CIFAR replay | 150 |
| [6](https://github.com/jeffreycider/learnable-weight-decay/pull/6) | Run records and restart | 103 |
| [7](https://github.com/jeffreycider/learnable-weight-decay/pull/7) | Base recipes and evaluation | 36 |
| [8](https://github.com/jeffreycider/learnable-weight-decay/pull/8) | Ordinary-pass observations | 72 |
| [9](https://github.com/jeffreycider/learnable-weight-decay/pull/9) | Shared trajectory capture | 67 |
| [10](https://github.com/jeffreycider/learnable-weight-decay/pull/10) | Chisel transformer and private maps | 73 |
| [11](https://github.com/jeffreycider/learnable-weight-decay/pull/11) | Transfer algebra and delayed credit | 55 |
| [12](https://github.com/jeffreycider/learnable-weight-decay/pull/12) | Online integration | 100 |
| [13](https://github.com/jeffreycider/learnable-weight-decay/pull/13) | Stable compiled sensor buffers | 17 |
| [14](https://github.com/jeffreycider/learnable-weight-decay/pull/14) | Offline state sampling and replay | 102 |
| [15](https://github.com/jeffreycider/learnable-weight-decay/pull/15) | Offline fitting and persistent private Adam | 103 |
| [16](https://github.com/jeffreycider/learnable-weight-decay/pull/16) | Offline/online YAML compositions | 0 |
| [17](https://github.com/jeffreycider/learnable-weight-decay/pull/17) | Restart, validation and compilation qualification | 89 |
| [18](https://github.com/jeffreycider/learnable-weight-decay/pull/18) | Published CUDA arithmetic and base-source checks | 55 |
| [19](https://github.com/jeffreycider/learnable-weight-decay/pull/19) | Review guide, checkpoint retention and metadata backup | 73 |

**The active implementation is about 1,400 Python lines.** The source-only review base preserves existing code for comparison; old standalone trainers under `scripts/` and `baseline/train.py` are historical, not production entrypoints. The original dirty research checkout remains intact.

## Verification

```bash
python -m pytest -q tests/harness
```

Tests cover block parity, literal MuonH arithmetic, global gradient/activation reduction, immutable capture, deterministic augmentation across worker counts, private optimizer persistence, exact one-step VJPs, pending-credit restart, and capture → offline fitting → online execution. CUDA tests additionally check compilation reuse and the real CIFAR pipeline. Qualification used PyTorch 2.11.0 / torchvision 0.26.0 on A100, plus CPU tests. These are correctness checks, not evidence of a validation-loss improvement.

A GitHub Actions workflow could not be published with the current OAuth token's scopes. The commands above were run directly; no green CI status is being claimed.
