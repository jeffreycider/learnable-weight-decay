"""Prospective two-host B residual-LR/decay screen; numerical source stays fixed."""
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
from scripts.run_soap46_followup import endpoint
from scripts.run_soap46_round2 import locked_state
DIRECTORY=ROOT/'runs/soap46-xlr-decay'


def frozen(config,seed,name):
    return {**config['fixed'],'meta_lr':1e-5,'residual_decay':.1,
            'warmup_steps':2900,'seed':seed,'name':name}


def initial_state(config):
    jobs=[]
    for lane,cells in enumerate(config['lanes'],1):
        control=frozen(config,3,f'soap46-xlrdecay-lane{lane}-frozen-seed3-v1')
        treatments=[{**config['fixed'],'seed':3,'meta_lr':cell['peak_lr'],
                     'residual_decay':cell['decay'],'name':cell['name']} for cell in cells]
        jobs.append(dict(id=f'screen-lane{lane}',lane=lane,status='pending',
                         trials=[control,*treatments]))
    return dict(phase='screen',jobs=jobs,phase_history=[],workers={},errors=[],
                created_at=time.time(),config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest())


def advance(state,config,score=endpoint):
    if any(j['status']!='complete' for j in state['jobs']):return
    state['phase_history'].append(dict(phase=state['phase'],jobs=state['jobs']))
    if state['phase']=='screen':
        controls=[dict(lane=j['lane'],name=j['trials'][0]['name'],loss=score(j['trials'][0]['name']),gpu=j.get('gpu')) for j in state['jobs']]
        rows=[]
        for job,control in zip(state['jobs'],controls):
            for trial in job['trials'][1:]:
                loss=score(trial['name']);difference=loss-control['loss']
                eligible=(difference<=-config['selection']['minimum_improvement'] and loss<min(c['loss'] for c in controls))
                rows.append(dict(trial=trial,lane=job['lane'],control=control['name'],
                                 loss=loss,difference=difference,eligible=eligible))
        state.update(controls=controls,screen_results=rows,
                     control_repeat_difference=controls[1]['loss']-controls[0]['loss'])
        candidates=[r for r in rows if r['eligible']]
        if not candidates:
            state.update(phase='review',jobs=[],finished_at=time.time(),reason='No cell passed the frozen practical screen; no automatic grid expansion')
            return
        winner=min(candidates,key=lambda r:(r['difference'],r['loss'],r['trial']['name']))
        state['selected']=winner;state['phase']='confirmation';state['jobs']=[]
        for lane,seed in enumerate(config['confirmation_seeds'],1):
            treatment={**winner['trial'],'seed':seed,'name':f'soap46-xlrdecay-finalist-seed{seed}-v1'}
            control=frozen(config,seed,f'soap46-xlrdecay-frozen-seed{seed}-v1')
            state['jobs'].append(dict(id=f'confirmation-seed{seed}',lane=lane,status='pending',trials=[treatment,control]))
    elif state['phase']=='confirmation':
        pairs=[]
        for job in state['jobs']:
            treatment,control=job['trials'];a,b=score(treatment['name']),score(control['name'])
            pairs.append(dict(seed=treatment['seed'],treatment=treatment['name'],control=control['name'],treatment_loss=a,control_loss=b,difference=a-b))
        state.update(phase='complete',confirmation_pairs=pairs,
                     primary_confirmation_mean=sum(r['difference'] for r in pairs)/len(pairs),finished_at=time.time())
    else:raise RuntimeError('Unknown phase')


def claim_lane(state,lane,gpu):
    if any(w['gpu']==gpu and int(k)!=lane for k,w in state['workers'].items()):
        raise RuntimeError('Two prospective lanes cannot share one host')
    old=state['workers'].get(str(lane))
    if old and old['gpu']!=gpu:raise RuntimeError('Prospective host assignment changed')
    job=next((j for j in state['jobs'] if j['lane']==lane),None)
    if job is None or job['status']=='complete':return None
    if job['status']!='pending':raise RuntimeError('Ambiguous existing running claim requires review')
    job.update(status='running',gpu=gpu,worker_pid=os.getpid(),started_at=time.time())
    state['workers'][str(lane)]=dict(gpu=gpu,pid=os.getpid(),last_seen=time.time())
    return job


def validate_trial(trial,config,gpu):
    folder=ROOT/'runs/soap46-v1'/trial['name']
    actual=json.loads((folder/'config.json').read_text());launch=json.loads((folder/'launch_plan.json').read_text())
    if any(actual.get(k)!=v for k,v in trial.items() if k!='name'):
        raise RuntimeError('Executed recipe differs from frozen declaration')
    if any(launch[k]!=config['numeric_source'][k] for k in ('sha256','source_commit')) or launch['gpu']!=gpu:
        raise RuntimeError('Wrong numerical source or assigned host')
    return endpoint(trial['name'])


def execute(job,config,gpu,mounted):
    plan_path=DIRECTORY/(job['id']+'.json')
    if plan_path.exists():raise RuntimeError('Existing plan needs explicit adoption; no ambiguous relaunch')
    panel.write(plan_path,{**config['numeric_source'],'gpu':gpu,'trials':job['trials'],
                          'already_mounted':mounted,'release_on_completion':False})
    with plan_path.with_suffix('.log').open('a') as log:
        subprocess.run([sys.executable,'scripts/run_soap46_panel.py','--plan',str(plan_path),'--execute'],
                       cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=4*3600)
    for trial in job['trials']:validate_trial(trial,config,gpu)


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',required=True);p.add_argument('--lane',type=int,choices=(1,2),required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/soap46_xlr_decay.json');p.add_argument('--already-mounted',action='store_true');p.add_argument('--execute',action='store_true')
    args=p.parse_args();config=json.loads(args.config.read_text())
    if not args.execute:print(json.dumps(initial_state(config),indent=2));return
    # This stage cannot overlap or condition on an incomplete current fresh pair.
    from scripts.run_soap46_b_uncertainty import validate_records
    previous=json.loads((ROOT/'configs/soap46_b_uncertainty.json').read_text())
    if not (ROOT/'runs/soap46-b-uncertainty/summary.json').exists():raise RuntimeError('Prior four-run protocol is not archived and complete')
    validate_records(previous)
    DIRECTORY.mkdir(exist_ok=True);deadline=datetime.fromisoformat(config['launch_deadline_utc']).timestamp()
    with (ROOT/'runs/soap46-round2'/f'{args.gpu}.worker.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state_path=DIRECTORY/'coordinator.json'
        with (DIRECTORY/'initialize.lock').open('a') as initialize:
            fcntl.flock(initialize,fcntl.LOCK_EX)
            if not state_path.exists():
                for path in (ROOT/'runs/soap46-v1').glob('*/config.json'):
                    old=json.loads(path.read_text())
                    if old.get('baseline')=='accepted-record46' and old.get('seed') in config['confirmation_seeds']:
                        raise RuntimeError('Declared confirmation seeds already used')
                panel.write(state_path,initial_state(config))
        mounted=args.already_mounted
        try:
            while time.time()<deadline:
                with locked_state(state_path) as state:
                    if state['errors']:raise RuntimeError('Other lane failed; preserve and review')
                    if state['config_sha256']!=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest():raise RuntimeError('Configuration changed')
                    if state['phase'] in ('complete','review'):break
                    advance(state,config)
                    if state['phase'] in ('complete','review'):break
                    job=claim_lane(state,args.lane,args.gpu)
                if job is None:time.sleep(30);continue
                print(json.dumps(dict(event='job_start',job=job['id'],gpu=args.gpu)),flush=True)
                execute(job,config,args.gpu,mounted);mounted=True
                with locked_state(state_path) as state:
                    next(j for j in state['jobs'] if j['id']==job['id']).update(status='complete',finished_at=time.time())
                print(json.dumps(dict(event='job_complete',job=job['id'])),flush=True)
            else:raise TimeoutError('No new jobs after launch deadline')
            panel.write(DIRECTORY/f'node{args.lane}_handoff.json',dict(gpu=args.gpu,preserved=True,time=time.time(),next_action='Owner review or release; no automatic scope expansion'))
        except Exception as error:
            with locked_state(state_path) as state:state['errors'].append(dict(lane=args.lane,error=repr(error),time=time.time()))
            raise


if __name__=='__main__':main()
