"""Bounded original-A/B learning-rate and decay study on clean retained leases."""
import argparse
from datetime import datetime
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
from scripts.run_soap46_round2 import claim,locked_state
from scripts.run_soap46_followup import endpoint
DIRECTORY=ROOT/'runs/soap46-lr-cwd'


def initial_state(config):
    jobs=[]
    for i,(lr,cwd) in enumerate(config['neighbors']):
        for variant in config['variants']:
            method='A' if variant=='inner-outer' else 'B'
            t={**config['fixed'],'variant':variant,'base_lr':lr,'cwd':cwd,
                'name':f'soap46-lrwd-{method}-cell{i}-seed0-v1'}
            jobs.append(dict(id=f'screen-{method}-{i}',status='pending',trials=[t]))
    return dict(phase='screen',jobs=jobs,phase_history=[],workers={},created_at=time.time(),
        config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest())


def pair(candidate,label,seed,config):
    return [{**candidate,'seed':seed,'name':f'soap46-lrwd-{label}-seed{seed}-v1'},
        {**candidate,**config['published_control'],'seed':seed,'warmup_steps':2900,
         'name':f'soap46-lrwd-{label}-frozen-seed{seed}-v1'}]


def control_for(job):
    if 'shared_control' in job:
        if len(job['trials'])!=1:raise ValueError('Shared control must not also be queued')
        return job['shared_control']
    if len(job['trials'])!=2:raise ValueError('Expected treatment and explicit control')
    return job['trials'][1]


def validate_shared_control(job,config):
    """Reuse a real archived zero-X run, never invent a duplicate observation."""
    if 'shared_control' not in job:return
    control=control_for(job);treatment=job['trials'][0]
    endpoint(control['name'])  # Includes every step's exact-zero residual check.
    directory=ROOT/'runs/soap46-v1'/control['name']
    actual=json.loads((directory/'config.json').read_text())
    launch=json.loads((directory/'launch_plan.json').read_text())
    if (actual['seed']!=treatment['seed'] or actual['steps']!=config['gpt_steps']
            or actual['microbatch']!=64 or actual['warmup_steps']<actual['steps']
            or actual['base_lr']!=config['published_control']['base_lr']
            or actual['cwd']!=config['published_control']['cwd']
            or launch['sha256']!=config['numeric_source']['sha256']
            or launch['source_commit']!=config['numeric_source']['source_commit']):
        raise RuntimeError('Shared control source, seed, execution, or baseline settings differ')


def activate_extension(state,config):
    """Owner-approved, conditional LR/CWD extension after both development losses fail."""
    extension=config['conditional_extension']
    if (state['phase']!='review' or state.get('extension_attempted') or
            len(state.get('development_comparisons',[]))!=2 or
            any(r['difference']<0 for r in state['development_comparisons'])):
        raise RuntimeError('Conditional extension gate is not satisfied')
    candidate=next(t for t in state['selected'] if t['variant']=='inner-outer')
    trials=[{**candidate,'seed':0,'base_lr':lr,'cwd':cwd,
             'name':f'soap46-lrwd-A-extension{i}-seed0-v1'}
            for i,(lr,cwd) in enumerate(extension['cells'])]
    state.update(extension_attempted=True,phase='extension-screen',
                 extension_incumbent=candidate,
                 jobs=[dict(id='extension-A-screen',status='pending',trials=trials)])
    state.pop('finished_at',None)


def advance(state,config,score=endpoint):
    if any(j['status']!='complete' for j in state['jobs']):return
    state['phase_history'].append(dict(phase=state['phase'],jobs=state['jobs']))
    if state['phase']=='screen':
        selected=[]
        for variant in config['variants']:
            trials=[t for j in state['jobs'] for t in j['trials'] if t['variant']==variant]
            selected.append(min(trials,key=lambda t:(score(t['name']),t['name'])))
        state['screen_scores']={t['name']:score(t['name']) for j in state['jobs'] for t in j['trials']}
        state['selected']=selected;state['phase']='development'
        state['jobs']=[dict(id=f'development-{i}',status='pending',trials=pair(t,f'dev{i}',config['development_seed'],config)) for i,t in enumerate(selected)]
    elif state['phase']=='development':
        results=[dict(candidate=state['selected'][i],treatment=j['trials'][0],
                      control=control_for(j),shared_control='shared_control' in j,
                      difference=score(j['trials'][0]['name'])-score(control_for(j)['name']))
                 for i,j in enumerate(state['jobs'])]
        state['development_comparisons']=results
        eligible=[r for r in results if r['difference']<0]
        if not eligible:
            state.update(phase='review',jobs=[],finished_at=time.time(),reason='Neither selected method improved its development paired control; owner review before further trials')
            return
        winner=min(eligible,key=lambda r:(r['difference'],r['candidate']['name']))['candidate']
        state['winner']=winner;state['phase']='confirmation'
        state['jobs']=[dict(id=f'confirmation-{seed}',status='pending',trials=pair(winner,'finalist',seed,config)) for seed in config['confirmation_seeds']]
    elif state['phase']=='confirmation':
        state['confirmation_differences']=[score(j['trials'][0]['name'])-score(j['trials'][1]['name']) for j in state['jobs']]
        state.update(phase='complete',finished_at=time.time())
    elif state['phase']=='extension-screen':
        candidates=[state['extension_incumbent']]+[t for j in state['jobs'] for t in j['trials']]
        winner=min(candidates,key=lambda t:(score(t['name']),t['name']))
        state['extension_screen_scores']={t['name']:score(t['name']) for t in candidates}
        state['extension_selected']=winner
        if winner['name']==state['extension_incumbent']['name']:
            state.update(phase='review',jobs=[],finished_at=time.time(),reason='Extension did not beat its incumbent; its failed development result is not rerun')
            return
        treatment={**winner,'seed':config['development_seed'],'name':'soap46-lrwd-A-extension-dev-seed3-v1'}
        control=config['conditional_extension']['shared_control']
        state.update(phase='extension-development',jobs=[dict(id='extension-development',
            status='pending',trials=[treatment],shared_control=control)])
    elif state['phase']=='extension-development':
        job=state['jobs'][0]
        difference=score(job['trials'][0]['name'])-score(control_for(job)['name'])
        state['extension_development_comparison']=dict(candidate=state['extension_selected'],
            treatment=job['trials'][0],control=control_for(job),shared_control=True,difference=difference)
        if difference>=0:
            state.update(phase='review',jobs=[],finished_at=time.time(),reason='Extension failed development comparison; fresh seeds remain untouched')
            return
        winner=state['extension_selected'];state.update(winner=winner,phase='confirmation')
        state['jobs']=[dict(id=f'confirmation-{seed}',status='pending',
            trials=pair(winner,'extension-finalist',seed,config)) for seed in config['confirmation_seeds']]
    else:raise RuntimeError('Unknown phase')


def execute(job,config,gpu):
    validate_shared_control(job,config)
    path=DIRECTORY/(job['id']+'.json')
    if path.exists():raise RuntimeError('Existing plan requires explicit adoption, never replay an ambiguous launch')
    panel.write(path,{**config['numeric_source'],'gpu':gpu,'trials':job['trials'],
        'already_mounted':True,'release_on_completion':False})
    with path.with_suffix('.log').open('a') as log:
        subprocess.run([sys.executable,'scripts/run_soap46_panel.py','--plan',str(path),'--execute'],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3*3600)
    for t in job['trials']:
        endpoint(t['name'])
        if not (ROOT/'runs/soap46-v1'/t['name']/'archive_result.json').exists():raise RuntimeError('Missing final archive')


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/soap46_lr_cwd_round3.json')
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    config=json.loads(a.config.read_text());deadline=datetime.fromisoformat(config['deadline_utc']).timestamp()
    if not a.execute:print(json.dumps(dict(maximum_runs=21 if config.get('conditional_extension') else 18,config=config),indent=2));return
    DIRECTORY.mkdir(exist_ok=True)
    with (ROOT/'runs/soap46-round2'/f'{a.gpu}.worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state_path=DIRECTORY/'coordinator.json'
        with (DIRECTORY/'initialize.lock').open('a') as initialize:
            fcntl.flock(initialize,fcntl.LOCK_EX)
            if not state_path.exists():panel.write(state_path,initial_state(config))
        try:
            while time.time()<deadline:
                with locked_state(state_path) as state:
                    if state['config_sha256']!=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest():raise RuntimeError('Configuration changed')
                    if state['phase'] in ('complete','review'):break
                    advance(state,config)
                    if state['phase'] in ('complete','review'):break
                    pending=next((j for j in state['jobs'] if j['status']=='pending'),None)
                    if pending and deadline-time.time()<config['minimum_launch_remaining_seconds']*len(pending['trials']):raise RuntimeError('Insufficient review-window time for complete job')
                    job=claim(state,a.gpu)
                    state['workers'][a.gpu]=dict(pid=os.getpid(),last_seen=time.time(),job=None if job is None else job['id'])
                if job is None:time.sleep(45);continue
                print(json.dumps(dict(event='job_start',job=job['id'],gpu=a.gpu)),flush=True)
                execute(job,config,a.gpu)
                with locked_state(state_path) as state:
                    next(j for j in state['jobs'] if j['id']==job['id']).update(status='complete',finished_at=time.time())
                print(json.dumps(dict(event='job_complete',job=job['id'],gpu=a.gpu)),flush=True)
            else:raise TimeoutError('Review deadline reached; no new trials launched')
            panel.write(DIRECTORY/(a.gpu+'.handoff.json'),dict(gpu=a.gpu,preserved=True,time=time.time(),next_action='Owner review or release; no automatic method expansion'))
        except Exception as error:
            with locked_state(state_path) as state:state.setdefault('errors',[]).append(dict(gpu=a.gpu,error=repr(error),time=time.time()))
            raise


if __name__=='__main__':main()
