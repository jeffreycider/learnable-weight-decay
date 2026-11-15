# Pinned PR323 baseline

This is a minimal extraction of Konstantin Mishchenko's PR323 Muon baseline,
from upstream commit `452cc392f6af4f29834d1af64edeac55d1315397`. The untouched
upstream script and rules are in `vendor/`. The muoff copy used for discovery is
also preserved: it differs from upstream by one trailing blank line only.
The model module was copied from muoff's modular harness; tests compare its
outputs and every parameter gradient bit-for-bit against the upstream classes.

The parent's later user clarification supersedes the handoff's 3000-step cap:
**the full comparison uses the original 3250-step PR323 schedule and settings**.
Short smoke/pilot runs retain `schedule_steps: 3250`; they do not compress the
learning-rate schedule. They cannot establish the published endpoint result.

Run from the repository root:

```bash
torchrun --standalone --nproc_per_node=1 -m baseline.train \
  --config baseline/config.json --mode baseline --output runs/baseline_seed0
```

Set `train_data` and `val_data` to locally staged files. Dataset identity is
`kjj0/fineweb10B-gpt2` at revision
`889765ea1f903759787add96995d81171b632d0c`. Sorted shard order and the original
shard-tail dropping rule are retained. Each run hashes the shards it will use
against the pinned Hugging Face LFS SHA256 values. The model does not shuffle
sequences or consume the controller's RNG. `seed` is set immediately before the
original parameter initialization loop. Pair seeds on the same GPU and software
stack; CUDA normal initialization need not match across hardware.

Protocol: 12 layers, width 768, vocabulary 50304; 162354816 parameters, 74
matrices, 162201600 matrix parameters, and 268032 total row/column coordinates.
Each optimizer step consumes 524288 tokens (512 sequences of length 1024).
Microbatches contain 64 sequences; one rank therefore executes 8 prescribed
forward/backward microbatches per step. Validation uses 10485760 tokens and the
live weights. No EMA or pre-forward lookahead is present. Validation is observed
only by the logger, never passed to the rewriter.

Muon hidden matrices: LR .025, weight decay .05, momentum .95, Nesterov,
12 Newton–Schulz iterations. Auxiliary AdamW: embedding LR .7, output-head LR
.004, one-dimensional parameter LR .015; betas (.8,.95), epsilon 1e-10,
weight decay .001. LR is stable for 30% of training then decays linearly over
70%. Gradients are **token sums**, with SUM all-reduction. Controller mean-loss
probes must divide by 524288 without modifying the base optimizer's gradient.
Muon mutates `.grad` in-place and keeps rank-sharded state under `momentum`;
AdamW's first moment is `exp_avg`.

`train.py` adds logging and four read/write integration hooks:
`before_step`, `before_microbatch`, `after_backward` (before optimizer mutation),
and `after_step` (after complete base update, before gradient clearing). Modes
`shadow` and `learned` load `rewriter.integration.RewriterAdapter`. They currently
fail closed for more than one rank until distributed controller synchronization
has been validated. The baseline itself supports 1, 2, 4, or 8 ranks as upstream.
No extra training pass is performed for compilation. The saved counters count
prescribed training microbatches; validation passes are separate.

Each run writes resolved configuration, environment, dataset checksums, full
Python/JSON source record, JSONL metrics, optional offline W&B records, and a
terminal checkpoint containing base/controller optimizer and RNG state. Atomic
`latest_rankN.pt` checkpoints are written every 250 completed iterations by
default (`checkpoint_every`, zero disables periodic checkpoints). Restart with
`--resume runs/previous/latest_rank0.pt` (or a directory for rank-local files).
The checkpoint stores the exact next-data cursor, clocks, RNGs, optimizer state,
and pending controller replay inputs. Restarting performs no previous GPT pass.
Research configuration, including seed, schedule, model, microbatching, controller
settings and terminal iteration, must match; only local data paths and logging/
checkpoint cadence may change. Static optimizer group settings are checked before
loading so stored betas/decay cannot silently override changed code. The source
record for a restart gets a separate filename and existing JSONL metrics append.
State is resumed only within the same trial, never imported across seeds.

The official pinned rules are preserved in `vendor/PR323_RULES.md`; meeting
those rules and beating a matched baseline are separate claims. This extraction
is not evidence of either result.


For sensor-heavy runs, `compile: false, compile_loss: true` keeps model and hooks
eager while compiling only the pure float conversion, softcap, and summed
cross-entropy. The compiler cannot enter sensor hooks in this mode, and
`fullgraph=True` prevents silent graph-break fallback inside the loss. There are
no warmup model evaluations. This mode uses the same architecture and loss
units, but compiler fusion can change floating-point rounding; compare against
a control using the same execution and microbatch settings. `compile_loss` is
off by default and is incompatible with whole-model `compile: true`. CPU tests
compare outputs and gradients with the pinned equations, including large logits
and an end-to-end hooked GPT. GPU memory/performance still require measurement.

For matched deterministic training, set `deterministic: true` and launch Python
with `PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8`. Before CUDA initializes,
this applies the unchanged `muoff/boxlogs/jstudy/determinism_policy.py` recipe
vendored here: strict deterministic algorithms (not warn-only), deterministic
Inductor, autotuning disabled, deterministic cuDNN with benchmarking disabled,
and Flash-only SDPA. The policy deliberately requires PyTorch 2.11.x; other
versions need separate qualification. It does not alter optimizer hyperparameters
or TF32 choices. The global controller's float32 attention requires its own local
math-SDPA context; GPT training retains Flash-only SDPA outside that context.

The source evidence is muoff's
`hippocampus/results/RES-DETERMINISM-RECIPE-PRINTED-REPEAT-20260928.md`: two fresh
runs matched full raw-byte states at updates 0/1/2/3/5/10/20/30 on the audited
8xA100 torch2.11 configuration. This is evidence for that test, not a universal
GPU determinism or resume guarantee. Every run records active determinism,
SDPA, TF32, threading, and launch-environment flags in `environment.json`.

`fingerprint_steps: [0, 64]` optionally logs SHA256s of every parameter and buffer
at initialization (before validation) and after the complete specified update.
The combined hash includes names, shapes, dtypes, and exact bytes; per-tensor
hashes locate disagreements. These checks copy model state to CPU and are
intentionally occasional. They add no GPT evaluation.
