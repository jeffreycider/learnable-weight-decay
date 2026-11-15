The corrected default follows the published PR395 SOAP–Shampoo update, with
learned additive left/right residuals and **no extra polar transform**.

Our earlier description incorrectly said the published path contained a final
polar. We consequently added an unintended operation. The user intended the
published baseline; their reply did not authorize that change. All earlier
SOAP results in `reports/soap/` compare versions of this **historical extra-polar
variant**, and do not answer the intended PR395 comparison. Their original
sources, checkpoints and results remain unchanged. No corrected-path GPU
training result has been established yet.

The untouched source is pinned to Git commit
`20dfc9150e0e99f83881c5a468fc2d8d0106e9b8`; `upstream.py` has SHA256
`c852286ca93859902ffb8c120fde013e088a1886d7a8293dcf8c2dbf9cbe917f`.
Importing that file would start distributed training. The reviewed driver
generates a patched copy and records it with each run.

From the repository root, with the prescribed FineWeb shards under
`data/fineweb10B`, a full learned run is:

```bash
OMP_NUM_THREADS=4 torchrun --standalone --nproc_per_node=8 \
  scripts/train_soap_hyper.py --seed 0 --steps 2900 --meta-lr 1e-5 \
  --microbatch 64 --checkpoint-every 500 --output RUN_DIRECTORY
```

Use `--meta-lr 0` for the published-path fixed control. The driver defaults to
`--variant pr395`. Learning rates `1e-5` and `1e-4` were tested only in the
historical extra-polar variant; neither is a demonstrated corrected-path
setting. Each fresh run resets model and optimizer state. The default
driver length remains 256 for short qualifications, so specify `--steps 2900`
for a full run. Effective batch is 524,288 tokens, sequence length 1,024. The
original model, initialization, non-hidden optimizer groups, 2,900-step schedule,
and fixed live/tail-EMA evaluation procedure are retained. Validation never
enters hyperparameter learning. The measured execution stack is PyTorch
2.11.0+cu128 on eight H100 80GB HBM3 GPUs; upstream's SDPA policy is retained.

For SOAP-processed momentum S and frozen Shampoo factors L/R, the intended raw
update is `(L + s_L H_L) S (R + s_R H_R)`. To preserve the upstream numerical
baseline, we evaluate its direct polynomial action `B = ShampooRaw(S)` using
the untouched pinned helper, then add only the residual expansion:

`B + Q_L S R + L S Q_R + Q_L S Q_R`, with `Q_L=s_L H_L`, `Q_R=s_R H_R`.

At zero residual, the baseline is not replaced by explicitly multiplying
`L S R`; that would change floating-point association. Explicit factors are
used only in the residual terms. Their association remains a declared
floating-point implementation choice for nonzero residuals. There is no SVD
or polar in this corrected action. The original Frobenius normalization, SPA,
radial scaling, row floor, radius pin and CWD follow. The parameter update
preserves upstream `add(..., alpha=-lr)` arithmetic and FP32 radius statistics.

H_L/H_R are full dense matrices for all 72 hidden weights: 297,271,296
zero-initialized parameters, sharded by the original matrix owners. Each
detached scale is the inverse fourth root of the mean damped Gram eigenvalue.
Only an exactly zero Gram uses a 1e-20 positive denominator. Embedding, head,
bias and normalization optimizer paths remain the original ones.

`--variant legacy-extra-polar` explicitly selects the historical map. To
reproduce historical runs or restore their checkpoints, use their immutable
source archive below; current source hashes deliberately reject those old
checkpoints. New checkpoints additionally reject crossing optimizer variants.

The next ordinary GPT gradient is divided by the global token count **only for
hypergradient credit**. Base gradients retain token-SUM units. Owners replay
frozen optimizer algebra one matrix at a time, then take one Adam update with
betas (0.9, 0.99), epsilon 1e-8, zero decay and global gradient clip 1. There is
one pending record, no gradient ring, GPT replay, HVP or task counterfactual.
Q=0 comparisons every 16 steps use optimizer algebra only.

Both actual wrappers are included. With upcoming displacement lookahead beta,
its EMA rate gamma, tail-anchor strength a and direct tail-EMA derivative c,
the committed-weight cotangent multiplier is
`J = (1+a)*(1+beta*(1-gamma)) - a*c`. Here c is 1 at lazy initialization, 1/tau
while updating and 0 after freezing. Tests cover every transition and compare
against differentiation through the pinned wrapper. This is direct one-step
credit with older optimizer state frozen, not a full-trajectory derivative.

Checkpoints contain every base optimizer, external head/embedding state, tail
EMA, loader cursor, rank-local RNG, H/Adam state and pending frozen record.
`--resume DIR` rejects different source hashes, seeds, world sizes, learning
rates, microbatches or endpoints. `scripts/train_soap_frozen.py --freeze-at 1000`
adds an explicit within-run diagnostic: it restores that checkpoint and holds
H fixed, while Q=sH still follows ordinary detached scale changes. Its distinct
checkpoint schema prevents accidental resumption with learning enabled.

The historical `scripts/launch_soap_full.py`, `poll_soap_full.py` and
`archive_soap_full.py` scripts remain pinned to the historical source archive;
they are not corrected-path experiment launchers. They launch only on an explicitly supplied existing lease, mirror complete metrics
atomically and preserve all rank shards with verified hashes. Source archives,
configuration, data hashes, offline W&B and raw logs accompany every run.
`scripts/report_soap_full.py` generates the report and downloadable reproduction
bundle. Results and limits belong in that report rather than being implied by
successful implementation tests.

```bash
OMP_NUM_THREADS=2 PYTHONPATH=. pytest -q \
  tests/test_lag_one.py tests/test_soap_residual.py tests/test_soap_pr395.py \
  tests/test_soap_envelope.py tests/test_soap_checkpoint.py \
  tests/test_soap_freeze.py
```

The corrected-path CPU qualification compares Q=0 bit-for-bit with extracted
pinned upstream code, including rectangular/square FP32/BF16 weights, SPA,
row-floor alternatives and a production-sized 768×768 matrix. It also checks
nonzero-residual finite differences, zero-initialization trainability, the
complete wrapper cotangent, Gram updates, and pending-credit resume. GPU
parity and corrected full training remain unmeasured.

The historical full-run training source was frozen at
`46dcd37`; its exact archive SHA256 is
`fa81f85494944f2e023a6ee1460100270bcd659cf1cc7f86ba0dcdec18e92ed1`.
