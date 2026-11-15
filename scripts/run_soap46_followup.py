"""Bounded post-panel research: four local neighbors, then two held-out seeds.

Orchestration only; all GPU runs retain the pinned published-precision source.
A lane selects on completed seed-0 endpoints, never on intermediate validation.
It confirms a candidate only if its seed-0 endpoint beats its frozen reference;
that selection result is not evidence of generalization or a benchmark win.
"""
from __future__ import annotations
import argparse
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


def endpoint(name,*,allow_changed_base=False):
    directory=ROOT/'runs/soap46-v1'/name
    if not (directory/'archive_result.json').exists():
        raise RuntimeError(f'Unarchived run: {name}')
    rows=[json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
    config=json.loads((directory/'config.json').read_text())
    train=[r for r in rows if r.get('event')=='train']
    if [r['step'] for r in train]!=list(range(1,2901)):
        raise RuntimeError(f'Incomplete ordinary training accounting: {name}')
    if config['warmup_steps']>=2900:
        if ((not allow_changed_base and (config['base_lr']!=.0375 or config['cwd']!=.025))
                or config['warmup_meta']!='moments' or config['meta_lr']<=0):
            raise RuntimeError('Frozen control does not use the declared published-base moment-warmup configuration')
        if any(r['basis_residual_norm']!=0 or r['residual_updates']!=0 for r in train):
            raise RuntimeError('Frozen control executed a nonzero residual')
        if train[-1]['zero_lr_moment_updates']<=0:
            raise RuntimeError('Frozen control did not accumulate Adam moments')
    values=[r['val_loss'] for r in rows if r.get('event')=='validation' and r.get('step')==2900]
    if len(values)!=1 or not __import__('math').isfinite(values[0]):
        raise RuntimeError(f'Expected one finite fixed endpoint: {name}')
    return values[0]


def best_trial(trials, score=endpoint):
    eligible=[t for t in trials if t['warmup_steps']<2900]
    if not eligible:raise RuntimeError('No learned candidates')
    return min(eligible,key=lambda t:(score(t['name']),t['name']))


def neighbors(winner,lane):
    result=[]
    for tag,lr,cwd in [('lr080',.03,.025),('lr120',.045,.025),('cwd050',.0375,.0125),('cwd150',.0375,.0375)]:
        result.append({**winner,'name':f'soap46-{lane}-neighbor-{tag}-seed0-v1','base_lr':lr,'cwd':cwd,'seed':0})
    return result


def confirmations(winner,lane):
    result=[]
    for seed in (1,2):
        result.append({**winner,'name':f'soap46-{lane}-finalist-seed{seed}-v1','seed':seed})
        result.append({**winner,'name':f'soap46-{lane}-frozen-seed{seed}-v1','seed':seed,
                       'warmup_steps':2900,'meta_lr':1e-5,'base_lr':.0375,'cwd':.025})
    return result


def attribution_controls(winner,lane):
    return [{**winner,'name':f'soap46-{lane}-frozen-finalist-base-seed{seed}-v1','seed':seed,
             'warmup_steps':2900,'meta_lr':1e-5} for seed in (1,2)]


def needs_attribution(winner,differences):
    changed=winner.get('base_lr',.0375)!=.0375 or winner.get('cwd',.025)!=.025
    return changed and len(differences)==2 and all(difference<0 for difference in differences)


def run_stage(plan_path,base,trials):
    plan={**base,'trials':trials,'release_on_completion':False,'already_mounted':True}
    if plan_path.exists():
        if json.loads(plan_path.read_text())!=plan:raise RuntimeError('Refuse changed stage plan')
    else:panel.write(plan_path,plan)
    progress_path=plan_path.with_suffix('.progress.json')
    if progress_path.exists():
        # A replacement outer supervisor adopts an already running child. It
        # never starts a second child that could race the same GPU or lock.
        deadline=time.monotonic()+5*3600
        while True:
            progress=json.loads(progress_path.read_text())
            if progress.get('finished'):return
            if progress.get('error'):raise RuntimeError('Stage supervisor reports failure')
            pid=progress.get('pid');cmdline=Path(f'/proc/{pid}/cmdline')
            if not cmdline.exists():raise RuntimeError('Existing stage supervisor died; review before retrying')
            if str(plan_path.resolve()) not in cmdline.read_bytes().decode():raise RuntimeError('Existing stage PID is unrelated')
            if time.monotonic()>deadline:raise RuntimeError('Existing stage exceeded five hours')
            time.sleep(45)
    # The child owns its own exclusive plan lock. A failure stops this manager;
    # ambiguous launcher mutations are never replayed automatically.
    with plan_path.with_suffix('.log').open('a') as log:
        subprocess.run([sys.executable,'scripts/run_soap46_panel.py','--plan',str(plan_path),'--execute'],
                       cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=5*3600)


def replacement_supervisor_alive(plan,proc_root=Path('/proc')):
    """Accept the explicitly recorded restart during its first adoption poll."""
    pidfile=plan.with_suffix('.pid')
    if not pidfile.exists():return False
    cmdline=proc_root/pidfile.read_text().strip()/'cmdline'
    return cmdline.exists() and str(plan.resolve()) in cmdline.read_bytes().decode()


def release_or_handoff(gpu,state,state_path):
    marker=state_path.with_name(f"handoff-{state['lane']}-v1.json")
    if marker.exists():
        handoff=json.loads(marker.read_text())
        if handoff.get('gpu')!=gpu:raise RuntimeError('Handoff GPU does not match the owned lease')
        archive=Path(handoff.get('source_archive',''))
        ready=(handoff.get('ready') is True and handoff.get('next_owner') and
               Path(handoff.get('launcher','')).is_file() and archive.is_file() and
               hashlib.sha256(archive.read_bytes()).hexdigest()==handoff.get('source_sha256'))
        if ready:
            state['handoff_ready']=handoff;panel.write(state_path,state)
            print(json.dumps({'event':'clean_node_handoff_ready','gpu':gpu,'next_owner':handoff['next_owner']}),flush=True)
            return
        state['handoff_skipped']='Successor command/source was not ready at clean completion.'
    panel.cli('gpu','down',gpu)
    state['lease_released']=True;panel.write(state_path,state)


def main():
    p=argparse.ArgumentParser();p.add_argument('--prior-plan',type=Path,required=True)
    p.add_argument('--lane',choices=['A','B'],required=True);p.add_argument('--execute',action='store_true');args=p.parse_args()
    old=json.loads(args.prior_plan.read_text())
    if old.get('release_on_completion',True):raise RuntimeError('Prior supervisor must retain its lease for this explicit successor')
    if not args.execute:
        print(json.dumps({'lane':args.lane,'gpu':old['gpu'],'max_followup_runs':10,
                          'neighbors':{'base_lr':[.03,.045],'cwd':[.0125,.0375]},'confirmation_seeds':[1,2]},indent=2));return
    import fcntl
    state_path=args.prior_plan.with_name(f'followup-{args.lane}-v1.json');lock=state_path.with_suffix('.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=json.loads(state_path.read_text()) if state_path.exists() else {'started_at':time.time(),'lane':args.lane,'gpu':old['gpu']}
    if state.get('error'):
        state.setdefault('reviewed_restart_errors',[]).append(state.pop('error'))
    state['pid']=os.getpid();panel.write(state_path,state)
    print(json.dumps({'event':'supervisor_started','lane':args.lane,'pid':os.getpid(),'attribution_controls_max':2}),flush=True)
    deadline=state['started_at']+7*3600
    try:
        while True:
            progress=json.loads(args.prior_plan.with_suffix('.progress.json').read_text())
            if progress.get('error'):raise RuntimeError('Prior panel reports failure')
            if progress.get('finished'):
                pid=progress['pid']
                if not Path(f'/proc/{pid}').exists():break
            elif not Path(f"/proc/{progress['pid']}").exists():
                if not replacement_supervisor_alive(args.prior_plan):
                    raise RuntimeError('Prior supervisor died; manual review required')
            if time.time()>deadline:raise RuntimeError('Seven-hour follow-up deadline exceeded')
            time.sleep(45)
        winner=best_trial(old['trials']);control=[t for t in old['trials'] if t['warmup_steps']==2900]
        if len(control)!=1:raise RuntimeError('Expected exactly one matched frozen seed-0 control')
        state['residual_selection']={'trial':winner,'endpoint':endpoint(winner['name']),'control_endpoint':endpoint(control[0]['name'])}
        panel.write(state_path,state)
        nearby=neighbors(winner,args.lane)
        run_stage(args.prior_plan.with_name(f'panel-{args.lane}-neighbors-v1.json'),old,nearby)
        finalist=best_trial([winner,*nearby]);score=endpoint(finalist['name']);reference=endpoint(control[0]['name'])
        state['finalist_selection']={'trial':finalist,'endpoint':score,'control_endpoint':reference,'seed0_difference':score-reference,
                                    'selection_only_not_improvement_evidence':True}
        panel.write(state_path,state)
        if score<reference:
            run_stage(args.prior_plan.with_name(f'panel-{args.lane}-confirmation-v1.json'),old,confirmations(finalist,args.lane))
            state['confirmation_completed']=True
            differences=[endpoint(f'soap46-{args.lane}-finalist-seed{seed}-v1')-endpoint(f'soap46-{args.lane}-frozen-seed{seed}-v1') for seed in (1,2)]
            state['confirmation_differences']=differences;panel.write(state_path,state)
            if needs_attribution(finalist,differences):
                diagnostics=attribution_controls(finalist,args.lane)
                run_stage(args.prior_plan.with_name(f'panel-{args.lane}-attribution-v1.json'),old,diagnostics)
                state['same_base_differences']=[endpoint(f'soap46-{args.lane}-finalist-seed{seed}-v1')-endpoint(f'soap46-{args.lane}-frozen-finalist-base-seed{seed}-v1',allow_changed_base=True) for seed in (1,2)]
        else:
            state['confirmation_completed']=False;state['reason']='No tuned seed-0 candidate improved the frozen endpoint; no promising finalist to confirm.'
        state['finished']=True;panel.write(state_path,state)
        panel.cli('sync','--note',f'Bounded SOAP46 lane {args.lane} follow-up completed; selection-only seed0 and held-out paired confirmations are recorded separately.')
        release_or_handoff(old['gpu'],state,state_path)
    except Exception as error:
        state['error']=repr(error);panel.write(state_path,state)
        # Preserve the lease and any running child on ambiguous failure. Parent
        # monitoring sees this explicit error; never kill/replay a live learner.
        raise


if __name__=='__main__':main()
