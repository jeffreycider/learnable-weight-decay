This directory pins the actual source embedded in accepted Track 3 record #46
(PR328), at official repository commit
`4ea6b937337a4889b8cfe3f38a93d120048d8f71`. `pin.json` records its SHA256 and log
path. Importing `upstream.py` starts distributed training.

The embedded source includes dormant experimental branches. Its defaults alone
are not the accepted run. `scripts/train_soap46.py` clears those environment
options and forces the recorded settings: RowFloor on, Frobenius restoration
off, rho1, CWD.025 after the radius pin, tail EMA tau150/start2400/end2900 and
nonembedding readout blend.6. The original model, data, effective batch524288,
sequence1024, 2900-step schedules, initialization, Adam paths, Muon polynomial,
SOAP covariance updates, attention trust gates and displacement-EMA wrapper
remain. No PR395/Shampoo, spectral annealing or head eigen-Adam is used.

The two learned variants replace SOAP bases with additive full dense residuals:

- `inner-outer`: `(QL+XL) [((QL+XL)^T D (QR+XR)) / sqrt(V)] (QR+XR)^T`.
- `outer-only`: `(QL+XL) [(QL^T D QR) / sqrt(V)] (QR+XR)^T`.

Division/square root are elementwise. V is updated using unperturbed Q and D,
exactly as in record46. Raw-G covariance and sorted QR basis updates occur
**after** the weight step, every step. The first pass has no SOAP basis and no
executed X-dependent action, so its next loss does not advance meta Adam.
Residuals start at zero each trial. With the default `--residual-frame basis`,
they persist in their stored matrix coordinates as Q refreshes; no state
transport or off-run initialization occurs.

Schatten-4 magnitude matching, the trust gate, finite 12-stage BF16 Muon
polynomial, radial brake, RowFloor, parameter update, radius pin and CWD are all
included in the delayed derivative. Each owner contracts this optimizer-only
map with the next ordinary task gradient; there is one pending record, no
extra GPT pass, HVP or gradient ring. Old observations are detached copies.
The displacement lookahead cotangent is `1+beta_next*(1-.99)` while active;
tail EMA is evaluation-only. No validation signal enters learning.

Example (fresh run):

```bash
PYTHONPATH=. OMP_NUM_THREADS=4 torchrun --standalone --nproc_per_node=8 \
  scripts/train_soap46.py --variant inner-outer --meta-lr 1e-5 \
  --base-lr .0375 --cwd .025 --meta-betas .9 .99 --meta-clip 1 \
  --steps 2900 --microbatch 64 --checkpoint-every 500 --output RUN_DIRECTORY
```

Use meta LR0 for frozen-X control. Base-LR changes scale the hidden decay
coefficient by the same factor, preserving cooldown onset. All other Adam
schedules stay fixed. Checkpoints preserve model, base optimizers, wrapper
clocks, tail EMA, data cursor, RNG, residual Adam and pending action snapshots.
Resume with `--resume RUN_DIRECTORY/step0500` and identical numeric settings.

`--compile-map` is currently blocked for training: isolated H100 qualification
found 1–8% gradient differences and failed weight parity even with
`emulate_precision_casts=True`. Eager zero-X was bit-exact to the pinned
upstream helpers on both tested full-sized shapes and variants. The isolated
qualification script preserves the compiler investigation. Its implementation
builds one persistent callable; changing LR, step, CWD and
observed tensor values are tensor inputs, not scalar specialization keys.
Static shape/attention/variant and grad/no-grad modes can each compile once.
Compilation uses an FP32 multiply/add path for tensor learning rates; it may
round differently from eager add(alpha=...). Qualify it separately before a
scientific run. Counts and per-component timings are logged.

```bash
PYTHONPATH=. python scripts/qualify_soap46.py --output QUALIFICATION --compile
OMP_NUM_THREADS=2 PYTHONPATH=. pytest -q tests/test_soap46_map.py tests/test_soap46_online.py tests/test_lag_one.py
```

The GPU script evaluates only optimizer tensors (default shapes768×768 and
3072×768), checks zero-X against the pinned upstream helpers, compares eager
and compiled gradients, and checks graph reuse across changed states/schedules.
It performs zero GPT forwards/backwards. GPU results must be read from its
artifact; CPU tests alone are not a full-model trajectory qualification.

### Identity warmup with real zero-residual credit

The warmed experiment uses `--warmup-steps 1000 --warmup-meta moments` with the
original 2900-step schedule. Actions 0 through 999 use exactly zero residuals.
After each subsequent ordinary gradient, the controller differentiates the
**executed zero-residual optimizer action**. Adam advances its moments and step
counter with learning rate zero; the residual parameters remain bit-zero. This
warms a gradient filter at the baseline trajectory. It does not evaluate or train
an unexecuted nonzero residual off that trajectory.

At ordinary iteration 1000, the gradient credits action 999. Adam's requested
learning rate is restored for this update, and action 1000 can first use nonzero
residuals. The initial basis-free action provides no residual derivative and is
skipped. Thus a 1000-step moment warmup has 998 moment-only Adam updates before
activation; action 999's feedback arrives at activation. `--warmup-meta off`
skips earlier VJPs/Adam updates but retains action 999 to supply that first credit.

The log separates `warmup_moment_updates`, `residual_updates`, total `meta_steps`,
`nominal_meta_lr`, and per-update `meta_effective_lr`. Warmup is not implemented
as a zero action multiplier, which would incorrectly erase its derivative.
Checkpoint/resume pins the policy and preserves Adam state and pending credit.
Fresh full runs are the supported warmed comparison; cross-variant prefix forks
are not supported because A and B have different zero-residual hypergradients.
Existing runs with `warmup_steps=0` are early-start diagnostics, not measurements
of this warmed design.

### Delayed ramp and direct residual decay

The new ramp experiment uses `--warmup-steps 500 --meta-ramp-steps 500
--warmup-meta moments --residual-decay 0.10`. For zero-based iteration `t`, its
actual meta LR is `peak * clamp((t - 500) / 500, 0, 1)`. It is exactly zero at
`t=500`, first positive at `t=501`, reaches the configured peak at `t=1000`, and
then stays there. With the logger's completed-step convention, the first nonzero
proposal can therefore appear in row502. This is distinct from the earlier
1000-step identity warmup followed by an immediate LR jump.

After **all pending action-time credit has been consumed**, active-phase
residuals are multiplied by `1 - residual_decay`, then Adam applies its update.
The shrink is independent of learning rate: decay0.10 means `X *= 0.9` on each
active step, including the zero-LR ramp boundary. It never decays GPT weights,
SOAP statistics, or Adam moments. Before the active phase, residual decay is0.
Logs include `applied_residual_decay`, `meta_effective_lr`,
`warmup_moment_updates`, `zero_lr_moment_updates`, and `residual_updates`.

Defaults `meta_ramp_steps=0` and `residual_decay=0` retain the previous schedule
and no-decay behavior. Resume requires the same source and all schedule/decay
settings; incompatible or missing checkpoint policy metadata is rejected
explicitly. A frozen reference can use a positive nominal peak with
`warmup_steps=2900`, retaining the same moment-computation machinery while
executing zero residuals throughout the run.

### Optional conditioning experiments

`--residual-frame parameter` is an explicit outer-only experiment. It stores
full dense `ZL,ZR` and forms `XL=ZL@QL`, `XR=ZR@QR` using frozen action-time bases.
For each fixed orthogonal Q this spans the same additive-residual family, but
Z persists in parameter coordinates across basis refreshes. Coordinatewise Adam
therefore follows a different learning trajectory. This leaves baseline Q/V
measurement updates unchanged; it is not a transport of baseline optimizer
state. `--meta-epsilon` exposes residual Adam's denominator floor (default1e-8).
Both choices are recorded and checked on resume. The existing numerical-source
0950ad0 experiments use basis coordinates and epsilon1e-8 throughout.

Separate CPU tests check zero-residual bitwise identity, consistent basis-column
permutations, full-map finite differences, delayed credit, and exact pending
resume. Run `scripts/qualify_soap46.py --variants outer-only --residual-frame
parameter --output PATH` for isolated GPU map qualification before training the
new parameterization. CPU correctness alone does not establish its GPU runtime
or learning benefit.
