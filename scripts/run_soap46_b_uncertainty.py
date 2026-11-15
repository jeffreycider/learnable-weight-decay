"""Predeclared fixed-B effect estimation, separate from failed tuning gates."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_soap46_followup import endpoint
from scripts.run_soap46_panel import write

DIRECTORY=ROOT/'runs/soap46-b-uncertainty'


def trials(config):
    fixed=config['fixed']
    if (config['seeds']!=[4,5] or config['gpt_steps']!=2900 or fixed['variant']!='outer-only'
            or fixed['base_lr']!=.0375 or fixed['cwd']!=.025
            or fixed['meta_betas']!=[.9,.99] or fixed['meta_lr']!=1e-5
            or fixed['residual_decay']!=.1 or fixed['meta_clip']!=1.
            or fixed['warmup_steps']!=500 or fixed['meta_ramp_steps']!=500):
        raise ValueError('Fixed-B uncertainty manifest differs from authorization')
    result=[]
    for seed in config['seeds']:
        result.extend([
            {**fixed,'seed':seed,'name':f'soap46-B-uncertainty-seed{seed}-v1'},
            {**fixed,'seed':seed,'warmup_steps':2900,
             'name':f'soap46-B-uncertainty-frozen-seed{seed}-v1'}])
    return result


def activation_decision(state):
    if state.get('errors'):raise RuntimeError('Prior SOAP queue has unresolved errors')
    if state['phase'] in ('confirmation','complete'):return 'defer_to_A_confirmation'
    if not state.get('extension_attempted'):return 'wait'
    if state['phase']!='review':return 'wait'
    selected=state.get('extension_selected')
    if not selected:raise RuntimeError('Extension review lacks its declared selection')
    if selected['name']==state['extension_incumbent']['name']:
        return 'launch_separate_uncertainty_study'
    comparison=state.get('extension_development_comparison')
    if comparison is None or comparison['difference']<0:
        raise RuntimeError('Extension failure was not established')
    return 'launch_separate_uncertainty_study'


def summarize(config,score=endpoint):
    declared=trials(config);pairs=[]
    for i,seed in enumerate(config['seeds']):
        treatment,control=declared[2*i:2*i+2]
        a,b=score(treatment['name']),score(control['name'])
        pairs.append(dict(seed=seed,treatment=treatment['name'],control=control['name'],
                          treatment_loss=a,control_loss=b,difference=a-b))
    return dict(purpose=config['purpose'],pairs=pairs,
        primary_mean_difference=sum(row['difference'] for row in pairs)/len(pairs),
        independent_seed_pairs=2,positive_means='learned B has higher loss',
        scope='Both fixed fresh-seed outcomes, regardless sign; no historical or development rows pooled into the primary mean. Not winner confirmation of the failed LR/CWD selection.')


def validate_records(config):
    for trial in trials(config):
        folder=ROOT/'runs/soap46-v1'/trial['name']
        actual=json.loads((folder/'config.json').read_text())
        launch=json.loads((folder/'launch_plan.json').read_text())
        if any(actual.get(k)!=v for k,v in trial.items() if k!='name'):
            raise RuntimeError('Executed uncertainty recipe differs from frozen manifest')
        if any(launch[k]!=config['numeric_source'][k] for k in ('sha256','source_commit')):
            raise RuntimeError('Uncertainty source differs')
        endpoint(trial['name'])


def preserve_study_record(gpu,name,record):
    code="""import json,pathlib,subprocess,sys
p=pathlib.Path('/scratch/artifacts/learnable-weight-decay/soap46-b-uncertainty')/sys.argv[1]
p.parent.mkdir(parents=True,exist_ok=True)
assert not p.exists(),str(p)
p.write_text(json.dumps(json.loads(sys.argv[2]),indent=2))
subprocess.run(['sync','/scratch'],check=True)
print(str(p))
"""
    subprocess.run(['strangeloop','gpu','exec',gpu,'--timeout','120','--',
        '/opt/venv211/bin/python','-c',code,name,json.dumps(record)],
        cwd=ROOT,check=True,timeout=150)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/soap46_b_uncertainty.json')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();config_bytes=args.config.read_bytes();config=json.loads(config_bytes)
    source_bytes=Path(__file__).read_bytes();declared=trials(config)
    if not args.execute:print(json.dumps(dict(config=config,trials=declared),indent=2));return
    DIRECTORY.mkdir(exist_ok=True)
    with (DIRECTORY/'supervisor.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        deadline=datetime.fromisoformat(config['activation_deadline_utc']).timestamp()
        while time.time()<deadline:
            state=json.loads((ROOT/'runs/soap46-lr-cwd/coordinator.json').read_text())
            decision=activation_decision(state)
            if decision=='defer_to_A_confirmation':
                write(DIRECTORY/'deferred.json',dict(time=time.time(),reason=decision,
                    next_action='A confirmation has priority; coordinate scheduling before this separate fixed-B study.'))
                return
            if decision!='wait':break
            time.sleep(30)
        else:raise TimeoutError('Conditional activation window expired; no GPU work launched')
        if args.config.read_bytes()!=config_bytes or Path(__file__).read_bytes()!=source_bytes:
            raise RuntimeError('Frozen study manifest or supervisor source changed while waiting')
        lease_lock=ROOT/'runs/soap46-round2'/f'{args.gpu}.worker.lock'
        with lease_lock.open('a') as lease:
            for _ in range(30):
                try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:time.sleep(2)
            else:raise RuntimeError('Prior SOAP owner still holds lease; no duplicate launch')
            state=json.loads((ROOT/'runs/soap46-lr-cwd/coordinator.json').read_text())
            if activation_decision(state)!='launch_separate_uncertainty_study':
                raise RuntimeError('Conditional decision changed at resource boundary')
            for phase in state['phase_history']:
                if phase['phase'] not in ('extension-screen','extension-development'):continue
                for job in phase['jobs']:
                    for trial in job['trials']:endpoint(trial['name'])
            observed=[]
            for path in (ROOT/'runs/soap46-v1').glob('*/config.json'):
                prior=json.loads(path.read_text())
                if prior.get('baseline')=='accepted-record46' and prior.get('seed') in config['seeds']:
                    observed.append(path.parent.name)
            if observed:raise RuntimeError('Fresh-seed status changed: '+repr(observed))
            plan_path=DIRECTORY/'panel.json'
            if plan_path.exists():raise RuntimeError('Existing study requires explicit adoption; never replay ambiguous launch')
            usage=json.loads(subprocess.run(['strangeloop','usage','--json'],cwd=ROOT,
                capture_output=True,text=True,check=True,timeout=120).stdout)
            remaining=usage['limit_usd']-usage['usage_usd']
            if usage.get('blocked') or usage.get('compute_locked') or remaining<config['estimated_study_cost_upper_usd']:
                raise RuntimeError('Study does not fit available allowance; preserve and request resource review')
            write(DIRECTORY/'activation.json',dict(time=time.time(),gpu=args.gpu,
                purpose=config['purpose'],config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                prior_failure=state.get('reason'),usage=usage,remaining_allowance_usd=remaining,
                fresh_seeds_observed_before_launch=observed,primary_outcome=config['primary_outcome']))
            write(plan_path,{**config['numeric_source'],'gpu':args.gpu,'trials':declared,
                             'already_mounted':True,'release_on_completion':False})
            preserve_study_record(args.gpu,'study_manifest.json',dict(config=config,
                config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                trials=declared,supervisor_source=source_bytes.decode(),
                supervisor_sha256=hashlib.sha256(source_bytes).hexdigest(),
                activation=json.loads((DIRECTORY/'activation.json').read_text())))
            with (DIRECTORY/'panel.log').open('a') as log:
                subprocess.run([sys.executable,'scripts/run_soap46_panel.py','--plan',str(plan_path),'--execute'],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3*3600)
            validate_records(config)
            summary={**summarize(config),'completed_at':time.time()}
            write(DIRECTORY/'summary.json',summary)
            preserve_study_record(args.gpu,'summary.json',summary)
            write(DIRECTORY/'handoff.json',dict(gpu=args.gpu,time=time.time(),preserved=True,
                next_action='Report all fixed outcomes; owner review or release, no automatic further sweep.'))


if __name__=='__main__':main()
