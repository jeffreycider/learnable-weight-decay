"""Same-source coordinate-frame × Adam-memory study after the preserved round2 queue.

The predecessor retains each clean lease explicitly. This supervisor never
interrupts its jobs or leases hardware. All choices use completed archived runs.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import statistics

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_soap46_panel as panel
from scripts.run_soap46_round2 import claim,locked_state,paired_trials
from scripts.run_soap46_followup import endpoint
DIRECTORY=ROOT/'runs/soap46-conditioning'
PREVIOUS=ROOT/'runs/soap46-round2'


def initial_state(config):
    jobs=[]
    for i,cell in enumerate(config['cells']):
        trials=[{**config['fixed'],**cell,'seed':seed,
            'name':f'soap46-r3-{cell["residual_frame"]}-b{str(cell["meta_betas"][1]).replace(".","")}-seed{seed}-v1'}
            for seed in config['development_seeds']]
        jobs.append(dict(id=f'factorial-{i}',status='pending',trials=trials))
    return dict(phase='factorial',jobs=jobs,phase_history=[],workers={},created_at=time.time(),
        config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest())


def advance(state,config,score=endpoint):
    if any(j['status']!='complete' for j in state['jobs']):return
    state['phase_history'].append(dict(phase=state['phase'],jobs=state['jobs']))
    if state['phase']=='factorial':
        trials=[t for j in state['jobs'] for t in j['trials']]
        means={j['id']:statistics.mean(score(t['name']) for t in j['trials']) for j in state['jobs']}
        best=min(state['jobs'],key=lambda j:(means[j['id']],j['id']))
        winner=best['trials'][0]
        state['factorial_scores']={t['name']:score(t['name']) for t in trials}
        state['factorial_mean_scores']=means
        state['winner']=winner;state['phase']='confirmation'
        state['jobs']=[dict(id=f'confirmation-{seed}',status='pending',trials=[
            {**winner,'seed':seed,'name':f'soap46-r3-finalist-seed{seed}-v1'},
            {**winner,'seed':seed,'name':f'soap46-r3-frozen-seed{seed}-v1','warmup_steps':2900,'meta_lr':1e-5}])
            for seed in config['confirmation_seeds']]
    elif state['phase']=='confirmation':
        state['confirmation_differences']=[score(j['trials'][0]['name'])-score(j['trials'][1]['name']) for j in state['jobs']]
        state['phase']='complete';state['finished_at']=time.time()
    else:raise RuntimeError('Unknown conditioning phase')


def predecessor_ready(gpu):
    path=PREVIOUS/(gpu+'.handoff.json')
    if not path.exists():return False
    record=json.loads(path.read_text());state=json.loads((PREVIOUS/'coordinator.json').read_text())
    if record.get('gpu')!=gpu or not record.get('preserved') or not record.get('queue_complete'):
        raise RuntimeError('Invalid predecessor handoff')
    if state['phase']!='complete':raise RuntimeError('Predecessor has not completed declared queue')
    return True


def execute_job(job,config,gpu,mounted):
    path=DIRECTORY/(job['id']+'.json')
    plan={**config['numeric_source'],'gpu':gpu,'trials':job['trials'],
        'already_mounted':mounted,'release_on_completion':False}
    if path.exists():raise RuntimeError('Existing job plan requires explicit adoption review')
    panel.write(path,plan)
    with path.with_suffix('.log').open('a') as log:
        subprocess.run([sys.executable,'scripts/run_soap46_panel.py','--plan',str(path),'--execute'],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3*3600)
    for t in job['trials']:endpoint(t['name'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/soap46_conditioning_round3.json')
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    config=json.loads(a.config.read_text());deadline=datetime.fromisoformat(config['deadline_utc']).timestamp()
    if not a.execute:print(json.dumps(dict(maximum_runs=len(config['cells'])*len(config['development_seeds'])+2*len(config['confirmation_seeds']),config=config),indent=2));return
    if (DIRECTORY/'held_by_user_20261110_0348.json').exists():
        raise RuntimeError('User held unlaunched coordinate-frame/Adam-memory study; explicit scope review required before any restart')
    DIRECTORY.mkdir(exist_ok=True)
    with (DIRECTORY/(a.gpu+'.worker.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not predecessor_ready(a.gpu):
            if time.time()>deadline:raise TimeoutError('No predecessor handoff before deadline; no GPU mutation')
            time.sleep(45)
        # Share the predecessor's lease lock to prevent another owner restart.
        with (PREVIOUS/(a.gpu+'.worker.lock')).open('a') as lease_lock:
            fcntl.flock(lease_lock,fcntl.LOCK_EX)
            state_path=DIRECTORY/'coordinator.json'
            with (DIRECTORY/'initialize.lock').open('a') as initialize:
                fcntl.flock(initialize,fcntl.LOCK_EX)
                if not state_path.exists():panel.write(state_path,initial_state(config))
            mounted=False
            try:
                while time.time()<deadline:
                    with locked_state(state_path) as state:
                        if state['config_sha256']!=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest():
                            raise RuntimeError('Conditioning configuration changed')
                        if state['phase']=='complete':break
                        advance(state,config)
                        if state['phase']=='complete':break
                        # Two-run confirmation jobs need two full trial slots.
                        pending=next((j for j in state['jobs'] if j['status']=='pending'),None)
                        if pending and deadline-time.time()<config['minimum_launch_remaining_seconds']*len(pending['trials']):
                            raise RuntimeError('Insufficient declared time to start a complete job; preserve state for owner review')
                        job=claim(state,a.gpu)
                        state['workers'][a.gpu]=dict(pid=os.getpid(),last_seen=time.time(),job=None if job is None else job['id'])
                    if job is None:time.sleep(45);continue
                    print(json.dumps(dict(event='job_start',job=job['id'],gpu=a.gpu)),flush=True)
                    execute_job(job,config,a.gpu,mounted);mounted=True
                    with locked_state(state_path) as state:
                        current=next(j for j in state['jobs'] if j['id']==job['id'])
                        current.update(status='complete',finished_at=time.time())
                    print(json.dumps(dict(event='job_complete',job=job['id'],gpu=a.gpu)),flush=True)
                else:raise TimeoutError('Declared conditioning deadline reached')
                panel.write(DIRECTORY/(a.gpu+'.handoff.json'),dict(gpu=a.gpu,preserved=True,queue_complete=True,
                    time=time.time(),owner='SOAP execution owner',next_action='Review paired evidence and next bounded branch before08:41UTC'))
                print(json.dumps(dict(event='conditioning_complete_node_retained',gpu=a.gpu)),flush=True)
            except Exception as error:
                with locked_state(state_path) as state:
                    state.setdefault('errors',[]).append(dict(gpu=a.gpu,error=repr(error),time=time.time()))
                raise


if __name__=='__main__':main()
