"""Run the pinned record-46 source; monitor its existing validation output.

This CPU supervisor adds no model calls. Run it on a prepared eight-GPU node.
The source is extracted verbatim from the published seed-0 log.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


RECORD_ENV = dict(ROWFLOOR='1', ROWFLOOR_FROB='0', ROWFLOOR_RHO='1.0',
                  CWD='0.025', CWD_POST='1', TAILEMA_TAU='150', TAILEMA_LAMBDA='0.6')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--steps', type=int, default=2900)
    a = p.parse_args()
    if not 1 <= a.steps <= 2900:
        raise ValueError('stop must fit the original 2900-step schedule')
    a.output.mkdir(parents=True, exist_ok=False)
    work = Path('/tmp/soap46-data')
    data = work / 'data/fineweb10B'
    data.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in [f'fineweb_train_{i:06d}.bin' for i in range(1, 17)] + ['fineweb_val_000000.bin']:
        src = Path('/scratch/datasets/fineweb10B') / name
        dst = data / name
        if not dst.exists():
            shutil.copy2(src, dst)
        if dst.stat().st_size != src.stat().st_size:
            raise ValueError('dataset size mismatch: ' + name)
        manifest[name] = {'bytes': dst.stat().st_size, 'sha256': hashlib.sha256(dst.read_bytes()).hexdigest()}
    source = a.source.read_text()
    # Logging cadence only; no numerical or optimizer operation changes.
    assert source.count('TRAIN_PROGRESS_INTERVAL = 0') == 1
    source = source.replace('TRAIN_PROGRESS_INTERVAL = 0', 'TRAIN_PROGRESS_INTERVAL = 25')
    target = a.output.resolve() / 'executed.py'
    target.write_text(source)
    env = dict(os.environ, **RECORD_ENV, STOP_STEP=str(a.steps), OMP_NUM_THREADS='4',
               TORCHINDUCTOR_COMPILE_THREADS='2', WANDB_MODE='offline')
    config = dict(seed=a.seed, steps=a.steps, schedule_steps=2900, record_env=RECORD_ENV,
                  upstream_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),
                  executed_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  effective_batch_tokens=524288, sequence_length=1024, world_size=8,
                  evaluation_readout='published tail EMA', extra_training_evaluations=0)
    (a.output / 'config.json').write_text(json.dumps(config, indent=2))
    (a.output / 'data_manifest.json').write_text(json.dumps(manifest, indent=2))
    import wandb
    run = wandb.init(project='learnable-weight-decay', name=a.output.name,
                     dir=str(a.output.resolve()), mode='offline', config=config)
    cmd = ['/opt/venv211/bin/torchrun', '--standalone', '--nproc_per_node=8', str(target), '--seed', str(a.seed)]
    start = time.time()
    with (a.output / 'stdout.log').open('w') as log, (a.output / 'metrics.jsonl').open('w') as metrics:
        process = subprocess.Popen(cmd, cwd=work, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        (a.output / 'launch.json').write_text(json.dumps(dict(pid=process.pid, command=cmd, started_at_unix=start)))
        for line in process.stdout:
            log.write(line); log.flush()
            if not line.startswith('step:'):
                continue
            row = {key: float(value) for key, value in re.findall(r'(val_loss|val_avg_loss|val_ema_loss|train_time|step_avg):([\d.]+)', line)}
            row.update(step=int(line.split(':', 1)[1].split('/')[0]), elapsed_seconds=time.time()-start)
            if 'val_ema_loss' in row:
                row['objective'] = row['val_ema_loss']; row['event'] = 'validation'
            else:
                row['event'] = 'progress'
            metrics.write(json.dumps(row)+'\n'); metrics.flush(); run.log(row)
        code = process.wait()
    (a.output / 'finished.json').write_text(json.dumps(dict(exit_code=code, elapsed_seconds=time.time()-start)))
    run.finish(exit_code=code)
    if code:
        raise SystemExit(code)


if __name__ == '__main__':
    main()
