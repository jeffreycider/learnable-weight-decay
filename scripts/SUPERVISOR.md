# SUPERSEDED — execution disabled

The user superseded this learned full-run plan with frozen-checkpoint objective work.
Both legacy `--execute` entrypoints now refuse execution. The text below is
historical; use only the separately reviewed baseline keeper for the two existing baselines.

# Exact six-run supervisor

The plan is fixed: baseline/reference/K2/tolerance10x at seed0, then baseline and
the predeclared reference at seed1. No best-seed0-arm selection occurs. All runs
use the frozen numerical source at `102ab99`, its existing deterministic archive,
and 3250 prescribed steps. Seed1 changes only the GPT seed; all model/controller/
optimizer state is initialized afresh, with the same predeclared independent RNG
stream settings. There is no checkpoint transfer between trials.

Review without contacting or changing leases:

```bash
.venv/bin/python scripts/supervise_six_runs.py
python scripts/launch_seed1_pair.py
```

Start the durable local supervisor only after review:

```bash
mkdir -p runs/six_run_supervisor
nohup .venv/bin/python -u scripts/supervise_six_runs.py --execute \
  > runs/six_run_supervisor/console.log 2>&1 < /dev/null &
```

The process holds `runs/six_run_supervisor.lock`; another instance fails. Its
original 18-hour deadline is saved in state and survives a restart. CLI calls
have explicit timeouts. It polls every 120 seconds and touches only the four
lease IDs in `runs/panel_leases.json`. It allocates no GPU. Finished baseline0's
lease runs reference1; finished K2's lease runs baseline1. Reference1 does not
wait for baseline1. Those lease windows extend from launch by 12 and 4 hours,
respectively (at most 16 additional H100-hours if retained for their entire
windows), subject to each pod's hard lifetime. Unused completed leases release
as soon as their artifacts are verified.

Before reusing/releasing a lease the supervisor requires a completed 3250-step
record, endpoint validation, closed trainer, final checkpoint, full source record,
configuration, environment, data hashes, and offline W&B artifacts. It copies the
final checkpoint once to `/scratch/checkpoints/learnable-weight-decay/20261108-six-run-plan/`.
The copied bytes are hashed and synced. Compact source/config/data/metrics/stdout/
W&B bundles go to the matching `/scratch/artifacts/` subtree and are downloaded
locally and verified before release. Large checkpoints are never downloaded to
this EC2 host. Original `/persist` outputs remain intact.

The supervisor writes `runs/six_run_supervisor/state.json` and a JSONL event log.
Each successful run is preserved under `runs/six_run_supervisor/runs/<label>`.
After all six complete it invokes `summarize_runs.py --checkpoint 3250 --plot`
and writes `runs/six_run_supervisor/report/`. At its deadline it creates a partial
report and launches nothing further; existing leases retain their bounded expiry.

Any failed or ambiguous launch/mutation is recorded as `manual_required`.
It is **not automatically replayed**, including after supervisor restart. Other
already-running trials can still be observed. Inspect local launch logs, remote
processes and artifacts before manually resolving a state error; do not simply
remove a launch-attempt marker. The supervisor never retries a failed training
trial or changes its configuration based on validation.

For manual launch only, these commands perform readiness checks and skip a busy
predecessor without interruption:

```bash
python scripts/launch_seed1_pair.py --execute --arm reference
python scripts/launch_seed1_pair.py --execute --arm baseline
```

Do not combine manual launching with an active supervisor. The seed1 launcher
records separate outputs/maps, verifies the frozen archive SHA256 and numerical
source files, and registers the new offline run. It is dry-run by default.
