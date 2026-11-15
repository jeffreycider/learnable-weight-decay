"""Run a predeclared SOAP panel on one existing lease, preserving every trial.

Launch this CPU supervisor with detached subprocess.Popen(start_new_session=True). It never changes a running trial's
hyperparameters or stops a trial based on validation. One supervisor per lease.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def command(args, timeout=420):
    # Only status reads and atomic metrics copies can be replayed safely.
    # A launch/extension/archive mutation is never retried on ambiguous errors.
    retryable = ((args[0] == 'strangeloop' and args[1:3] == ['gpu', 'status']) or
                 (len(args) > 1 and Path(args[1]).name in
                  ('poll_soap46.py', 'snapshot_soap46_metrics.py')))
    for attempt in range(3):
        try:
            result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            if not retryable or attempt == 2:
                raise
        else:
            if not result.returncode:
                return result.stdout
            detail = result.stderr or result.stdout
            transient = any(s in detail.lower() for s in
                ('http 502', 'http 503', 'http 504', 'connection reset', 'network unreachable', 'timed out'))
            if not retryable or not transient or attempt == 2:
                raise RuntimeError(f'{args[:3]}: {detail}')
        print(json.dumps({'event': 'retry_status_transport', 'command': args[:3],
                          'attempt': attempt + 1}), flush=True)
        time.sleep(5 * (attempt + 1))


def cli(*args):
    return command(['strangeloop', *args])


def script(name, *args):
    return command([sys.executable, 'scripts/' + name, *map(str, args)])


def write(path, data):
    temporary = path.with_suffix('.incomplete')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def pause_at_boundary(plan_path, gpu, progress, archive_pending):
    """Explicit handoff only after a finished run and verified preservation."""
    marker=plan_path.with_suffix('.pause.json')
    if not marker.exists():return
    for prior in archive_pending:
        print(script('archive_soap46.py','--gpu',gpu,'--name',prior,'--execute'),flush=True)
    archive_pending.clear()
    acknowledgement=plan_path.with_suffix('.paused.json')
    write(acknowledgement,dict(gpu=gpu,completed=list(progress['completed']),
        preserved=True,paused_at=time.time(),request=json.loads(marker.read_text())))
    print(json.dumps({'event':'paused_after_preservation','gpu':gpu}),flush=True)
    # Time passing is not authorization to overlap a successor benchmark.
    while marker.exists():time.sleep(5)
    acknowledgement.unlink(missing_ok=True)
    print(json.dumps({'event':'explicit_pause_released','gpu':gpu}),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    import fcntl
    lock = args.plan.with_suffix('.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    progress_path = args.plan.with_suffix('.progress.json')
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {'completed': []}
    progress['pid'] = os.getpid()
    archive_pending = [name for name in progress['completed']
                       if not (ROOT / 'runs/soap46-v1' / name / 'archive_result.json').exists()]
    gpu = plan['gpu']
    for trial in plan['trials']:
        if trial['name'] in progress['completed']:
            continue
        status = json.loads(cli('gpu', 'status', gpu, '--json'))
        if status['status'] != 'ready':
            raise RuntimeError('Lease is not ready; no automatic replacement')
        if status['expires_in_seconds'] < 7200:
            cli('gpu', 'extend', gpu, '--hours', '3')
        output = status['artifact_dir'] + '/' + trial['name']
        # A launch record is the adoption boundary; never blindly replay a launch.
        record = ROOT / 'runs/soap46-v1' / (trial['name'] + '-launch.json')
        if record.exists():
            actual = json.loads(record.read_text())
            for key, value in {**trial, 'gpu': gpu, 'sha256': plan['sha256'], 'source_commit': plan['source_commit']}.items():
                if actual.get(key) != value:
                    raise RuntimeError(f'Cannot adopt mismatching launch: {key}')
        else:
            launch = ['--gpu', gpu, '--name', trial['name'], '--variant', trial['variant'],
                      '--meta-lr', trial['meta_lr'], '--warmup-steps', trial['warmup_steps'],
                      '--warmup-meta', 'moments', '--meta-ramp-steps', trial['meta_ramp_steps'],
                      '--residual-decay', trial['residual_decay'], '--steps', 2900,
                      '--checkpoint-every', trial.get('checkpoint_every', 1000), '--archive', plan['archive'],
                      '--sha256', plan['sha256'], '--source-commit', plan['source_commit'], '--execute']
            for key in ('seed', 'base_lr', 'cwd', 'meta_clip', 'meta_epsilon', 'residual_frame'):
                if key in trial:
                    launch += ['--' + key.replace('_', '-'), trial[key]]
            if trial.get('reset_meta_at_activation',False):launch.append('--reset-meta-at-activation')
            if 'meta_betas' in trial:
                launch += ['--meta-betas', *trial['meta_betas']]
            if progress['completed'] or plan.get('already_mounted'):
                launch.append('--reuse-mounted-volume')
            print(script('launch_soap46.py', *launch), flush=True)
        progress['active'] = trial['name']
        write(progress_path, progress)
        # Archive the preceding closed run while the next run uses the GPUs.
        for prior in archive_pending:
            print(script('archive_soap46.py', '--gpu', gpu, '--name', prior, '--execute'), flush=True)
        archive_pending.clear()
        registered = False
        last_snapshot = -1
        started = time.monotonic()
        while True:
            state = json.loads(script('poll_soap46.py', '--gpu', gpu, '--output', output))
            row = state.get('latest_train') or {}
            write(args.plan.with_suffix('.status.json'), state)
            if not registered and state['wandb']:
                cli('gpu', 'declare', gpu, '--name', trial['name'], '--wandb-path', str(Path(state['wandb'][0]).parent))
                registered = True
            step = row.get('step', 0)
            if step // 125 > last_snapshot and step:
                print(script('snapshot_soap46_metrics.py', '--gpu', gpu, '--name', trial['name']), flush=True)
                print(json.dumps({'run': trial['name'], 'step': step, 'validation': state.get('latest_validation'),
                                  'residual_norm': row.get('basis_residual_norm'), 'effective_lr': row.get('meta_effective_lr')}), flush=True)
                last_snapshot = step // 125
            # Training has resumed past this step only after every rank's
            # atomic checkpoint and complete marker are closed. Copy these
            # immutable shards while the GPUs continue the current trial.
            saved = progress.setdefault('checkpoint_archives', {}).setdefault(trial['name'], [])
            checkpoint_every = trial.get('checkpoint_every', 1000)
            checkpoint_steps = range(checkpoint_every, 2900, checkpoint_every) if checkpoint_every else ()
            for checkpoint_step in checkpoint_steps:
                if step > checkpoint_step and checkpoint_step not in saved:
                    marker = ROOT / 'runs/soap46-v1' / trial['name'] / f'step{checkpoint_step:04d}' / 'archive_result.json'
                    if not marker.exists():
                        print(script('archive_soap46.py', '--gpu', gpu, '--name', trial['name'],
                                     '--checkpoint-step', checkpoint_step, '--execute'), flush=True)
                    saved.append(checkpoint_step)
                    write(progress_path, progress)
            if state['finished']:
                if state['finished']['completed_steps'] != 2900:
                    raise RuntimeError('Unexpected endpoint')
                break
            if state.get('launcher_alive') is False:
                progress['error'] = {'run': trial['name'], 'stdout_tail': state.get('stdout_tail')}
                write(progress_path, progress)
                cli('gpu', 'down', gpu, '--note', f"SOAP panel stopped: {trial['name']} exited without completion; intermediate files remain in durable storage.")
                raise RuntimeError('Training process exited without completion; stopped lane and released lease')
            if time.monotonic() - started > 4200:
                raise RuntimeError('Trial has not finished within 70 minutes; inspect before continuing')
            time.sleep(45)
        # Wait for distributed workers to release the GPUs before the next launch.
        for _ in range(30):
            active = cli('gpu', 'exec', gpu, '--', 'nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader').strip()
            if not active:
                break
            time.sleep(2)
        else:
            raise RuntimeError('Finished workers still occupy GPUs')
        progress['completed'].append(trial['name'])
        progress['active'] = None
        write(progress_path, progress)
        archive_pending.append(trial['name'])
        pause_at_boundary(args.plan,gpu,progress,archive_pending)
    for prior in archive_pending:
        print(script('archive_soap46.py', '--gpu', gpu, '--name', prior, '--execute'), flush=True)
    progress['finished'] = True
    write(progress_path, progress)
    cli('sync', '--note', f"SOAP ramp/decay panel completed on {gpu}: {progress['completed']}; results require paired-seed analysis.")
    if plan.get('release_on_completion', True):
        cli('gpu', 'down', gpu)


if __name__ == '__main__':
    main()
