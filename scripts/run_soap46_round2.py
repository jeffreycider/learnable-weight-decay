"""Bounded shared SOAP queue; selection occurs only between complete full runs.

Workers own already handed-over leases. No allocations, interrupted training,
validation-dependent within-run changes, or ambiguous launch retries occur here.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_soap46_panel as panel
from scripts.run_soap46_followup import endpoint, replacement_supervisor_alive

DIRECTORY=ROOT/'runs/soap46-round2'
KEYS=('variant','meta_lr','meta_clip','meta_betas','warmup_steps','warmup_meta',
      'meta_ramp_steps','residual_decay','seed','base_lr','cwd','checkpoint_every')


def initial_state(config,adopt_plan):
    return dict(phase='screen',jobs=[dict(id='initial-lane1',trials=config['lane1_trials'],
        status='external',external_plan=str(adopt_plan.resolve()))]+[
        dict(id='screen-'+str(i),trials=[t],status='pending') for i,t in enumerate(config['lane2_trials'])],
        workers={},created_at=time.time(),phase_history=[],config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest())


def historical_reference():
    name='soap46-B-ramp500-1000-d10-lr1e-5-seed0-v1'
    cfg=json.loads((ROOT/'runs/soap46-v1'/name/'config.json').read_text())
    trial={k:cfg[k] for k in KEYS};trial.update(name=name,checkpoint_every=0)
    return trial


def signature(trial):
    return json.dumps({k:v for k,v in trial.items() if k not in ('name','seed','checkpoint_every')},sort_keys=True)


def select_candidates(state,reference,score=endpoint):
    candidates=[reference]+[t for j in state['jobs'] for t in j['trials']
        if t['variant']=='outer-only' and t['warmup_steps']==500]
    unique={}
    for t in candidates:
        key=signature(t)
        if key not in unique or score(t['name'])<score(unique[key]['name']):unique[key]=t
    return sorted(unique.values(),key=lambda t:(score(t['name']),t['name']))[:2]


def paired_trials(candidate,label,seed):
    treatment={**candidate,'name':f'soap46-r2-{label}-seed{seed}-v1','seed':seed,'checkpoint_every':0}
    control={**candidate,'name':f'soap46-r2-{label}-frozen-seed{seed}-v1','seed':seed,
        'warmup_steps':2900,'meta_lr':1e-5,'checkpoint_every':0}
    return [treatment,control]


def advance(state,reference,score=endpoint):
    if any(j['status']!='complete' for j in state['jobs']):return
    old=state['phase'];state['phase_history'].append(dict(phase=old,jobs=state['jobs']))
    if old=='screen':
        candidates=select_candidates(state,reference,score)
        if len(candidates)!=2:raise RuntimeError('Expected two distinct eligible B candidates')
        state['selected_seed0']=candidates;state['phase']='development'
        state['jobs']=[dict(id=f'development-{i}',trials=paired_trials(t,f'dev{i}',3),status='pending') for i,t in enumerate(candidates)]
    elif old=='development':
        comparisons=[]
        for i,j in enumerate(state['jobs']):
            treatment,control=j['trials']
            comparisons.append(dict(candidate=state['selected_seed0'][i],
                difference=score(treatment['name'])-score(control['name']),treatment=treatment,control=control))
        winner=min(comparisons,key=lambda x:(x['difference'],x['candidate']['name']))
        state['development_comparisons']=comparisons;state['selected_finalist']=winner['candidate']
        state['phase']='confirmation'
        state['jobs']=[dict(id=f'confirmation-{seed}',trials=paired_trials(winner['candidate'],'finalist',seed),status='pending') for seed in (4,5)]
    elif old=='confirmation':
        state['confirmation_differences']=[score(j['trials'][0]['name'])-score(j['trials'][1]['name']) for j in state['jobs']]
        state['phase']='complete';state['finished_at']=time.time()
        state['next_action']='Interpret fixed-setting fresh-seed results and hypergradient health before choosing the next bounded experiment; this queue is not project completion.'
    else:raise RuntimeError('Unknown phase')


@contextmanager
def locked_state(path):
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=json.loads(path.read_text())
        yield state
        panel.write(path,state)


def refresh_external(state):
    for j in state['jobs']:
        if j['status']!='external':continue
        progress_path=Path(j['external_plan']).with_suffix('.progress.json')
        if not progress_path.exists():continue
        progress=json.loads(progress_path.read_text())
        if progress.get('error'):raise RuntimeError('Adopted panel reports failure')
        if progress.get('finished'):
            for t in j['trials']:endpoint(t['name'])
            j.update(status='complete',finished_at=time.time())


def claim(state,gpu):
    if any(j.get('gpu')==gpu and j['status']=='running' for j in state['jobs']):
        raise RuntimeError('Existing running claim requires manual review before worker restart')
    available=next((j for j in state['jobs'] if j['status']=='pending'),None)
    if available:available.update(status='running',gpu=gpu,worker_pid=os.getpid(),started_at=time.time())
    return available


def execute_job(job,config,gpu,mounted):
    plan={**config['numeric_source'],'gpu':gpu,'trials':job['trials'],
        'already_mounted':mounted,'release_on_completion':False}
    path=DIRECTORY/(job['id']+'.json')
    if path.exists() and json.loads(path.read_text())!=plan:raise RuntimeError('Existing job plan differs')
    panel.write(path,plan)
    if path.with_suffix('.progress.json').exists():raise RuntimeError('Existing job progress requires explicit adoption review')
    with path.with_suffix('.log').open('a') as log:
        subprocess.run([sys.executable,'scripts/run_soap46_panel.py','--plan',str(path),'--execute'],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3*3600)
    for t in job['trials']:endpoint(t['name'])


def validate_adoption(job, config, gpu, path, progress, command):
    """Adopt only the exact surviving CPU panel; never launch a replacement GPT."""
    plan=json.loads(path.read_text())
    if (job.get('gpu')!=gpu or plan.get('gpu')!=gpu or
        plan.get('trials')!=job['trials'] or
        any(plan.get(k)!=v for k,v in config['numeric_source'].items())):
        raise RuntimeError('Adoption source, GPU, or trial mismatch')
    if progress.get('error'):raise RuntimeError('Cannot adopt a failed panel')
    if progress.get('finished'):return
    if not command or 'scripts/run_soap46_panel.py' not in command:
        raise RuntimeError('Expected surviving panel process is absent')
    i=command.index('--plan') if '--plan' in command else -1
    if i<0 or i+1>=len(command) or Path(command[i+1]).resolve()!=path.resolve():
        raise RuntimeError('Surviving panel command belongs to another plan')


def adopt_running_job(job, config, gpu, deadline):
    path=DIRECTORY/(job['id']+'.json')
    while time.time()<deadline:
        progress=json.loads(path.with_suffix('.progress.json').read_text())
        process=Path('/proc')/str(progress.get('pid'))/'cmdline'
        command=process.read_bytes().decode().rstrip('\0').split('\0') if process.exists() else []
        validate_adoption(job,config,gpu,path,progress,command)
        if progress.get('finished'):
            for t in job['trials']:
                endpoint(t['name'])
                if not (ROOT/'runs/soap46-v1'/t['name']/'archive_result.json').exists():
                    raise RuntimeError('Finished adopted trial lacks its archive record')
            return
        time.sleep(45)
    raise TimeoutError('Adopted running panel exceeded worker deadline')


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',required=True);p.add_argument('--adopt-plan',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/soap46_tuning_round2.json')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--adopt-running',action='store_true',help='Explicitly adopt the exact surviving CPU panel after an owner-supervisor restart')
    p.add_argument('--retain-on-completion',action='store_true',help='Return a preserved clean node to the authorized successor owner instead of releasing it')
    a=p.parse_args()
    config=json.loads(a.config.read_text());reference=historical_reference()
    if not a.execute:print(json.dumps(dict(max_total_runs=20,config=config),indent=2));return
    if (ROOT/'runs/soap46-conditioning/held_by_user_20261110_0348.json').exists():
        raise RuntimeError('Future round2 jobs held after user scope correction; current standalone panels preserve already-running comparisons')
    DIRECTORY.mkdir(exist_ok=True);state_path=DIRECTORY/'coordinator.json'
    with (DIRECTORY/'initialize.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not state_path.exists():panel.write(state_path,initial_state(config,a.adopt_plan))
    with (DIRECTORY/(a.gpu+'.worker.lock')).open('a') as worker_lock:
        fcntl.flock(worker_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        own_external=json.loads(a.adopt_plan.read_text())['gpu']==a.gpu
        deadline=json.loads(state_path.read_text())['created_at']+10*3600
        if own_external:
            # The original six-run supervisor keeps exclusive GPU ownership.
            while True:
                progress=json.loads(a.adopt_plan.with_suffix('.progress.json').read_text())
                if progress.get('finished'):break
                pid=progress.get('pid')
                if not Path(f'/proc/{pid}').exists() and not replacement_supervisor_alive(a.adopt_plan):raise RuntimeError('Adopted supervisor died; review its active run before restarting')
                if time.time()>deadline:raise TimeoutError('Adopted supervisor exceeded worker deadline')
                time.sleep(45)
        mounted=own_external
        try:
            with locked_state(state_path) as state:
                existing=[j for j in state['jobs'] if j.get('gpu')==a.gpu and j['status']=='running']
                if existing and not a.adopt_running:
                    raise RuntimeError('Existing running claim requires explicit --adopt-running')
                if len(existing)>1:raise RuntimeError('Multiple running claims for one GPU')
                if existing:existing[0]['worker_pid']=os.getpid()
            if existing:
                adopt_running_job(existing[0],config,a.gpu,deadline);mounted=True
                with locked_state(state_path) as state:
                    current=next(j for j in state['jobs'] if j['id']==existing[0]['id'])
                    current.update(status='complete',finished_at=time.time())
                print(json.dumps({'event':'adopted_job_complete','job':existing[0]['id'],'gpu':a.gpu}),flush=True)
            while time.time()<deadline:
                with locked_state(state_path) as state:
                    if state['config_sha256']!=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest():raise RuntimeError('Round configuration changed')
                    refresh_external(state)
                    if state['phase']=='complete':break
                    advance(state,reference)
                    if state['phase']=='complete':break
                    job=claim(state,a.gpu)
                    state['workers'][a.gpu]=dict(pid=os.getpid(),last_seen=time.time(),job=None if job is None else job['id'])
                if job is None:
                    time.sleep(45);continue
                print(json.dumps({'event':'job_start','job':job['id'],'gpu':a.gpu}),flush=True)
                execute_job(job,config,a.gpu,mounted);mounted=True
                with locked_state(state_path) as state:
                    current=next(j for j in state['jobs'] if j['id']==job['id'])
                    current.update(status='complete',finished_at=time.time())
                print(json.dumps({'event':'job_complete','job':job['id']}),flush=True)
            else:raise TimeoutError('Ten-hour worker deadline exceeded')
            if a.retain_on_completion:
                panel.write(DIRECTORY/(a.gpu+'.handoff.json'),dict(gpu=a.gpu,time=time.time(),
                    queue_complete=True,preserved=True,owner='SOAP execution owner',
                    reason='User-authorized six-hour adaptive tuning continues after this bounded queue'))
                print(json.dumps({'event':'queue_complete_node_retained_for_successor','gpu':a.gpu}),flush=True)
            else:
                panel.cli('gpu','down',a.gpu)
                print(json.dumps({'event':'queue_complete_node_released','gpu':a.gpu}),flush=True)
        except Exception as error:
            with locked_state(state_path) as state:state.setdefault('errors',[]).append(dict(gpu=a.gpu,error=repr(error),time=time.time()))
            raise  # Keep any ambiguous active GPU process for explicit review.


if __name__=='__main__':main()
