"""Explicitly reviewed replacement lease for the interrupted SOAP seed-2 pair.

This never retries an ambiguous launch. The original incomplete attempt remains
separately recorded; only the unchanged two-trial recovery plan may execute.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_soap46_followup as followup
from scripts import run_soap46_panel as panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--lane', choices=['A', 'B'], required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    assert len(plan['trials']) == 2
    assert all(t['seed'] == 2 for t in plan['trials'])
    assert plan['release_on_completion'] is False
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    state_path = args.plan.with_name(f'followup-{args.lane}-v1.json')
    lock = state_path.with_suffix('.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = json.loads(state_path.read_text())
    assert state['recovery']['new_gpu'] == plan['gpu']
    state['pid'] = os.getpid()
    state['gpu'] = plan['gpu']
    panel.write(state_path, state)
    try:
        with args.plan.with_suffix('.log').open('a') as log:
            subprocess.run([sys.executable, 'scripts/run_soap46_panel.py',
                            '--plan', str(args.plan.resolve()), '--execute'],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=3 * 3600)
        state['confirmation_differences'] = [
            followup.endpoint(f'soap46-{args.lane}-finalist-seed{seed}-v1')
            - followup.endpoint(f'soap46-{args.lane}-frozen-seed{seed}-v1')
            for seed in (1, 2)
        ]
        state['confirmation_completed'] = True
        state['finished'] = True
        panel.write(state_path, state)
        panel.cli('sync', '--note', f'SOAP46 {args.lane} replacement completed the unchanged seed2 pair; the interrupted 324-step attempt is excluded and preserved separately.')
        followup.release_or_handoff(plan['gpu'], state, state_path)
    except Exception as error:
        state['error'] = repr(error)
        panel.write(state_path, state)
        raise


if __name__ == '__main__':
    main()
