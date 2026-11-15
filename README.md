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
pip install -e '.[test,cifar,storage]'
python -m chisel.harness recipes/cifar_capture.yaml
# Text configuration uses eight ranks:
torchrun --standalone --nproc_per_node=8 -m chisel.harness recipes/nanogpt_capture.yaml
```

**Edit YAML to change an experiment.** Relative `extends` merges mappings and replaces lists. A dependency's `_target_` names its constructor; lifecycle entries name hook factories. The CLI accepts only a recipe path. Paths, seeds and output directories must be set for the run. JSONL contains measurements, not configuration.

**Text capture uses microbatches of 32 sequences on A100.** Gradient accumulation preserves the prescribed global batch of 524,288 tokens; eight ranks each accumulate two microbatches. This avoids the measured 80 GB memory overflow with 64-sequence microbatches under instrumentation.

**CIFAR retains QK normalization and learns one scale per head.** Width, depth, head size, MLP expansion, attention mask and scale are constructor arguments. The image interface changes only patch inputs and classification readout. The pinned augmentation dependency lives in a Git submodule, not a run directory.

**Chisel learns simplification offline and consolidation online.** For each hidden matrix,

$$R'=R+\Delta R,\qquad C'=C+\Delta C,\qquad W'=R'+C'.$$

The two edits are independent. Moving weights from R to C without changing W earns no shrinkage reward. Online, unchanged MuonH updates R; the next ordinary backward supplies task credit through the edits and optimizer algebra. No extra GPT evaluation occurs online.

**Offline training rewards smaller effective weights while preserving predictions.** The shrinkage term averages fractional norm changes equally across weight families. Teacher/student evaluations on a seeded sample of previously seen minibatches supply continuous centered-logit error, KL and tie-aware ranking penalties. A dual multiplier increases pressure when mean KL exceeds its configured tolerance. The consolidated-weight penalty discourages storing unnecessary weight in C. No edit-size penalty is used. These extra offline model evaluations are not benchmark-compliant training.

**Checkpoint-private probes actually learn.** A shared NanoGPT transformer processes all matrix summaries. Each saved GPT state has its own trainable read/write maps and Adam state. YAML selects the private fitting steps per visit and the checkpoint batch size for shared-backbone updates. Each visit loads, trains and saves private state. Compilation covers the probe encoder, shared transformer and edit decoder together; the edited model forward and AOT backward compile separately. Changing checkpoint values must reuse those graphs.

**Capture stores replayable transitions and optional raw activation shards without duplicating CIFAR images.** Records contain weights, raw gradients, optimizer state, activity, resolved configuration, base-source hashes and the next data cursor. CIFAR augmentation and mixing are determined by explicit seeds and the step; FineWeb uses its shard cursor. Receipts contain file size and SHA256. Dataset files must remain available at their recorded identities. Terminal records say whether a subsequent prescribed training step exists.

**Offline training and validation manifests must contain disjoint checkpoint hashes.** A manifest is a YAML mapping with `states`, each containing `path`, `sha256` and `domain`. Relative paths resolve against the manifest. Validation visits every listed checkpoint without optimizer updates. Unseen checkpoints use deterministic, unfitted private probes; this metric does not establish performance after support fitting. Audits report mean and worst checkpoint shrinkage, mean and maximum token KL, and ranking changes with their denominator. Validation never enters the online update.

**Restart includes pending credit and private fitting state.** Numbered checkpoints are retained, with `latest_rank{rank}.pt` pointing to the newest one. Set `resume` to that file; a shorter `stop_step` leaves the original learning-rate schedule intact. Offline checkpoints commit pointers to immutable private-state versions, so retain the `private/` directory with the shared checkpoint. Source/config changes are rejected on resume. Runs save resolved YAML, source archives, package versions and RNG state.

## Saved data and retained systems work

[Checkpoint contents and replay contract](docs/checkpoints.md) describe the actual saved fields, capture clocks, activation sampling and R2 backup.

| Capability | Active implementation |
|---|---|
| Offline I/O | Bounded CPU prefetch; optional pinned memory; checkpoint SHA verification |
| Multi-GPU offline fitting | Fixed ownership of private probes; shared-gradient reduction; committed restart generations |
| Compilation | Configurable actor and edited-model compilation; tests reject repeated compilation on new checkpoint values |
| Fused activation | Optional FP32 Triton backend; default remains native compiled PyTorch |
| CUDA graphs | Selectable through compiler `mode: reduce-overhead`; no manually captured old-objective graph reused |
| Storage operations | YAML-driven trajectory uploader, dataset backup and complete-checkpoint catalog |

The A100 [MLP benchmark](docs/mlp-benchmark-a100.json) found no meaningful advantage over native compilation (about 1.5 ms per forward/backward/Adam step). This is not a full offline-training benchmark. Persistent kernels require a new layout and objective integration; their code is archived, with no speedup claimed for the new trainer.

## Review ladder

Each draft PR targets its immediate predecessor. PR30 is the user-approved deletion-only exception to the size limit. Counts include added **and** deleted non-test Python lines. Some later rungs contain corrections found by integration/GPU tests; review the complete stack before deployment.

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
| [20](https://github.com/jeffreycider/learnable-weight-decay/pull/20) | Checkpoint phase, replay cursor and hyperparameters | 28 |
| [21](https://github.com/jeffreycider/learnable-weight-decay/pull/21) | Unprojected activation and adjoint shards | 122 |
| [22](https://github.com/jeffreycider/learnable-weight-decay/pull/22) | Verified immutable R2 storage | 89 |
| [23](https://github.com/jeffreycider/learnable-weight-decay/pull/23) | Upload workers and catalog creation | 116 |
| [24](https://github.com/jeffreycider/learnable-weight-decay/pull/24) | Bounded checkpoint prefetch | 81 |
| [25](https://github.com/jeffreycider/learnable-weight-decay/pull/25) | Optional fused FP32 activation | 97 |
| [26](https://github.com/jeffreycider/learnable-weight-decay/pull/26) | Systems benchmarks and compiler modes | 58 |
| [27](https://github.com/jeffreycider/learnable-weight-decay/pull/27) | Distributed offline fitting and restart | 85 |
| [28](https://github.com/jeffreycider/learnable-weight-decay/pull/28) | Dataset identities and RNG metadata | 47 |
| [29](https://github.com/jeffreycider/learnable-weight-decay/pull/29) | Explicit source compatibility | 28 |
| [30](https://github.com/jeffreycider/learnable-weight-decay/pull/30) | Delete obsolete and archived code; approved size exception | 60717 |
| [31](https://github.com/jeffreycider/learnable-weight-decay/pull/31) | Migration guide and end-to-end qualification | see PR |
| [32](https://github.com/jeffreycider/learnable-weight-decay/pull/32) | Allow independent raw and consolidated edits instead of forcing redistribution | 51 |
| [33](https://github.com/jeffreycider/learnable-weight-decay/pull/33) | Define grouped effective-norm reward and exact prediction-preservation measurements | 71 |
| [34](https://github.com/jeffreycider/learnable-weight-decay/pull/34) | Differentiate actual teacher-student preservation and audit fitted checkpoints | 105 |
| [35](https://github.com/jeffreycider/learnable-weight-decay/pull/35) | Train offline Chisel to shrink effective weights while preserving predictions | 108 |
| [36](https://github.com/jeffreycider/learnable-weight-decay/pull/36) | Cache replayed immutable batches without caching learned representations | 32 |
| [37](https://github.com/jeffreycider/learnable-weight-decay/pull/37) | Allow private probes to equilibrate before each shared-backbone update | 37 |
| [38](https://github.com/jeffreycider/learnable-weight-decay/pull/38) | Document effective-weight simplification and qualified A100 batching | 0 |
| [39](https://github.com/jeffreycider/learnable-weight-decay/pull/39) | Import sharded PR267 trajectories with token replay and provenance checks | 136 |
| [40](https://github.com/jeffreycider/learnable-weight-decay/pull/40) | Train prediction preservation on deterministic samples of all prior minibatches | 42 |
| [41](https://github.com/jeffreycider/learnable-weight-decay/pull/41) | Match teacher and student compiled forward arithmetic at zero edit | 6 |
| [42](https://github.com/jeffreycider/learnable-weight-decay/pull/42) | Restore missing archived shards and retain parallel import provenance | 53 |
| [43](https://github.com/jeffreycider/learnable-weight-decay/pull/43) | Configure domain minibatches and linear ranking constraints for mixed fitting | 16 |
| [44](https://github.com/jeffreycider/learnable-weight-decay/pull/44) | Restore representative FineWeb states before filling archival gaps | 25 |
| [45](https://github.com/jeffreycider/learnable-weight-decay/pull/45) | Own restored trajectory bytes independently of legacy uploader deletion | 14 |
| [46](https://github.com/jeffreycider/learnable-weight-decay/pull/46) | Audit actual effective Frobenius norm separately from equal-family reward | 23 |

| [47](https://github.com/jeffreycider/learnable-weight-decay/pull/47) | Cache immutable teacher logits when reuse is worthwhile | 50 |
| [48](https://github.com/jeffreycider/learnable-weight-decay/pull/48) | Cache source checkpoints without mutating recorded optimizer state | 21 |
| [49](https://github.com/jeffreycider/learnable-weight-decay/pull/49) | Keep the corrected design and writing references in project memory | 0 |
| [50](https://github.com/jeffreycider/learnable-weight-decay/pull/50) | Protect original run provenance during rejected resumes | 6 |
| [51](https://github.com/jeffreycider/learnable-weight-decay/pull/51) | Qualify compiled teacher caching; leave it opt-in | 0 |

| [52](https://github.com/jeffreycider/learnable-weight-decay/pull/52) | Audit trained private probes from every distributed checkpoint owner | 19 |

| [53](https://github.com/jeffreycider/learnable-weight-decay/pull/53) | Measure full optimizer steps, memory, and compiler reuse | 27 |
| [54](https://github.com/jeffreycider/learnable-weight-decay/pull/54) | Restore complete checkpoints before later shards | 13 |
| [55](https://github.com/jeffreycider/learnable-weight-decay/pull/55) | Expose the progress of checkpoint-private fitting | 9 |
| [56](https://github.com/jeffreycider/learnable-weight-decay/pull/56) | Preserve ordering across complete teacher-logit tie groups | 34 |
| [57](https://github.com/jeffreycider/learnable-weight-decay/pull/57) | Initialize shared Chisel weights from a hash-pinned checkpoint | 13 |
| [58](https://github.com/jeffreycider/learnable-weight-decay/pull/58) | Stage recordings locally and copy complete states asynchronously | 104 |

| [59](https://github.com/jeffreycider/learnable-weight-decay/pull/59) | Schedule offline preservation pressure and private-probe learning rates | 56 |
| [60](https://github.com/jeffreycider/learnable-weight-decay/pull/60) | Continue staged recordings and reclaim only verified local copies | 98 |
| [61](https://github.com/jeffreycider/learnable-weight-decay/pull/61) | Initialize trainable private probes with paired attention directions | 62 |

| [62](https://github.com/jeffreycider/learnable-weight-decay/pull/62) | Keep native forwards with optional compensated linear gradients | 121 |

| [63](https://github.com/jeffreycider/learnable-weight-decay/pull/63) | Retain preservation and norm measurements for every audited checkpoint | 8 |
| [64](https://github.com/jeffreycider/learnable-weight-decay/pull/64) | Record independent seeds in disjoint GPU groups with storage backpressure | 42 |
| [65](https://github.com/jeffreycider/learnable-weight-decay/pull/65) | Fit held-out checkpoint probes on support examples before disjoint-query audits | 75 |
| [66](https://github.com/jeffreycider/learnable-weight-decay/pull/66) | Improve private fitting, preservation pressure and held-out image audits | 98 |

| [69](https://github.com/jeffreycider/learnable-weight-decay/pull/69) | Fit and restore independent checkpoint preservation multipliers | 57 |

**The active tree contains the shared harness and its supported components.** Obsolete trainers are deleted from this branch. Useful SOAP, kernel and storage research is retained on [the archival branch](https://github.com/jeffreycider/learnable-weight-decay/tree/archive/research-20261115). Running jobs retain frozen source releases.

## Verification

```bash
python -m pytest -q tests/harness
```

Tests cover block parity, literal MuonH arithmetic, global gradient/activation reduction, immutable capture, deterministic augmentation across worker counts, private optimizer persistence, exact one-step VJPs, pending-credit restart, and capture → offline fitting → online execution. CUDA tests additionally check compilation reuse and a tiny CIFAR pipeline using real images. The current CPU suite has 98 successful tests and nine CUDA-only skips. A100 checks include the capture/offline/online pipeline, two-GPU offline fitting, and repeated cached versus uncached preservation gradients. Commands and environments are in [qualification.yaml](docs/qualification.yaml). These are correctness checks, not evidence of a validation-loss improvement.

A GitHub Actions workflow could not be published with the current OAuth token's scopes. The commands above were run directly; no green CI status is being claimed.
