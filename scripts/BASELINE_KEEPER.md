# Baseline-only keeper

The obsolete learned six-run launchers are disabled. This keeper manages only
existing baseline0 on `6470a9a9-c41d-47cc-9048-9048caa100b1` and baseline1 on
`e3e0f6ab-e6bf-4c60-9f32-cca58afb0447`. It launches no training and allocates no GPU.

Review with `.venv/bin/python scripts/keep_baselines.py`. After review:

Launch from the repository root using a detached Python subprocess. In this
terminal tool context, a shell `nohup ... &` did not survive the tool shell's
exit; `start_new_session=True` with all three streams redirected did.

```bash
.venv/bin/python - <<'PYTHON'
import subprocess
from pathlib import Path
root = Path.cwd().resolve()
output = root / 'runs/baseline_keeper'
output.mkdir(parents=True, exist_ok=True)
with (output / 'console.log').open('ab') as log:
    process = subprocess.Popen(
        [str(root / '.venv/bin/python'), '-u',
         str(root / 'scripts/keep_baselines.py'), '--execute'],
        cwd=root, stdin=subprocess.DEVNULL, stdout=log,
        stderr=subprocess.STDOUT, start_new_session=True,
    )
print(process.pid)
PYTHON
```

For a restart, leave `runs/baseline_keeper/state.json` intact. The keeper retains
its original deadline and successful archival records; the single-instance lock
rejects a second active keeper. Do not clear ambiguous mutation markers merely
to restart. The event log uses `time` for observation time and `source_time` for
a checkpoint manifest's timestamp.

A single-instance lock prevents duplicate keepers. Polling is every60seconds and
the original deadline is6hours, persisted across restarts. Explicit CLI timeouts
are capped by that deadline. Read-only transient errors retry at the next poll;
failed or ambiguous mutations retain an attempt/error marker and require manual
inspection rather than automatic replay.

Seed1 rolling checkpoints250/500/750/1000 are copied through an open file handle,
so the trainer's atomic replacement cannot change an in-progress copy. The
copied checkpoint's actual seed, mode, step and data cursor are checked. Each
saved file is hashed, synced and rehashed under:

`/scratch/checkpoints/learnable-weight-decay/baseline_keeper/seed1_stepN.pt`

Step250 was separately captured and verified before the keeper existed; the
keeper verifies and adopts that manifest. Missed rolling checkpoints are reported
explicitly rather than reconstructed from future state.

For each finished3250-step baseline, the keeper requires the endpoint validation,
closed training process, final checkpoint and complete source/config/environment/
data/stdout/offline-W&B records. Final checkpoints and compact record bundles go
under the corresponding scratch `baseline_keeper/final/<label>` directories.
Compact bundles are downloaded and every file verified locally before that GPU
can be released. Large checkpoints are never downloaded to EC2. Original
`/persist` records remain intact. W&B declaration failures block release pending
manual inspection.

Local records, checkpoint manifests and event logs live under
`runs/baseline_keeper/`. A failed/unfinished trainer is never restarted, stopped,
or released automatically. The keeper touches no other lease.
