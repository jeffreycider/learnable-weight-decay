"""Bounded supervisor for four seed0 arms plus fixed reference/control seed1.

Default is dry-run. --execute never allocates GPUs, changes configurations,
resumes a failed trial, or chooses an endpoint from validation. Only the four
recorded lease IDs may be touched. A failed/ambiguous mutation sets a persisted
manual-required error; subsequent loops only observe that run. Restarting does
not clear errors or reset the original 18-hour deadline.
"""
from __future__ import annotations
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1]
PYTHON='/opt/venv211/bin/python'
EXPECTED_SEED0={'baseline','reference','k2','tolerance10x'}
FOLLOW_ON={'baseline_seed0':'reference','k2_seed0':'baseline'}
FINAL_LABELS={x+'_seed0' for x in EXPECTED_SEED0}|{'baseline_seed1','reference_seed1'}
SCRATCH_ROOT='learnable-weight-decay/20261108-six-run-plan'
COMMAND_DEADLINE=None


def stamp(): return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    os.replace(temporary,path)


def cli(*args,timeout=280):
    if COMMAND_DEADLINE is not None:
        timeout=min(timeout,max(1,COMMAND_DEADLINE-time.time()))
    result=subprocess.run(['strangeloop',*args],text=True,capture_output=True,timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'CLI failed ({result.returncode}): {args[:3]}: {result.stderr[-4000:]}')
    return result.stdout


OBSERVE=r'''
import pathlib,json
out=pathlib.Path(OUTPUT)
names=['config.json','environment.json','data_manifest.json','metrics.jsonl']
files={name:(out/name).read_text() for name in names if (out/name).is_file()}
# A reader may catch the final write mid-line; only complete JSONL records count.
text=files.get('metrics.jsonl','');text=text[:text.rfind('\n')+1] if text else ''
files['metrics.jsonl']=text
rows=[json.loads(line) for line in text.splitlines() if line]
active=[]
for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
    try: args=p.read_bytes().decode(errors='replace').split('\x00')
    except (FileNotFoundError,PermissionError,ProcessLookupError): continue
    if '--output' in args and args[args.index('--output')+1]==str(out): active.append(p.parent.name)
finished=[x for x in rows if x.get('event')=='finished']
trains=[x for x in rows if x.get('event')=='train']
vals=[x for x in rows if x.get('event')=='validation']
stdout=pathlib.Path(str(out)+'.log')
tail=''
if stdout.exists():
    with stdout.open('rb') as f:
        f.seek(max(0,stdout.stat().st_size-16384));tail=f.read().decode(errors='replace')
print(json.dumps(dict(files=files,active=active,finished=finished[-1] if finished else None,
     train=trains[-1] if trains else None,validation=vals[-1] if vals else None,
     wandb=[str(x.parent) for x in (out/'wandb').glob('offline-run-*/*.wandb')],stdout_tail=tail)))
'''


PRESERVE=r'''
import pathlib,json,hashlib,shutil,tarfile,subprocess,os
out=pathlib.Path(OUTPUT);label=LABEL
rows=[json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines() if line]
if not any(r.get('event')=='finished' and r.get('steps')==3250 for r in rows):
    raise RuntimeError('cannot preserve/release an unfinished trial')
if not any(r.get('event')=='validation' and r.get('step')==3250 for r in rows):
    raise RuntimeError('missing fixed-endpoint validation')
required=['config.json','environment.json','data_manifest.json','source_record.txt','final_rank0.pt']
if any(not (out/name).is_file() or (out/name).stat().st_size==0 for name in required):
    raise RuntimeError('missing/empty reproduction artifact')
if not list((out/'wandb').glob('offline-run-*/*.wandb')):
    raise RuntimeError('missing offline W&B record')
# The trainer must have closed W&B and checkpoint files before inventory/export.
for proc in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
    try: args=proc.read_bytes().decode(errors='replace').split('\x00')
    except (FileNotFoundError,PermissionError,ProcessLookupError): continue
    if '--output' in args and args[args.index('--output')+1]==str(out):
        raise RuntimeError('trainer still active; defer preservation')
artifact=pathlib.Path('/scratch/artifacts')/SCRATCH_ROOT/label
checkpoint_dir=pathlib.Path('/scratch/checkpoints')/SCRATCH_ROOT/label
artifact.mkdir(parents=True,exist_ok=True);checkpoint_dir.mkdir(parents=True,exist_ok=True)
manifest_path=artifact/'preservation_manifest.json'
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()
if manifest_path.exists():
    saved=json.loads(manifest_path.read_text())
    for item in (saved['checkpoint'],saved['bundle']):
        if digest(pathlib.Path('/scratch')/item['scratch_path'])!=item['sha256']:
            raise RuntimeError('previous preservation verification failed')
    print(json.dumps(saved));raise SystemExit(0)
source=out/'final_rank0.pt';destination=checkpoint_dir/'final_rank0.pt'
if not destination.exists() and shutil.disk_usage('/scratch').free < source.stat().st_size + 1024**3:
    raise RuntimeError('insufficient scratch space for final checkpoint and compact run record')
source_hash=digest(source)
if destination.exists():
    if digest(destination)!=source_hash: raise RuntimeError('existing durable checkpoint differs; refusing overwrite')
else:
    temporary=checkpoint_dir/'final_rank0.pt.partial'
    if temporary.exists(): raise RuntimeError('ambiguous prior checkpoint copy; inspect manually')
    shutil.copyfile(source,temporary)
    if digest(temporary)!=source_hash: raise RuntimeError('checkpoint copy checksum mismatch')
    os.replace(temporary,destination)
bundle=artifact/'run_record.tar.gz'
if bundle.exists(): raise RuntimeError('unmanifested prior bundle; inspect manually')
files=[p for p in out.rglob('*') if p.is_file() and not p.is_symlink() and p.suffix not in {'.pt','.tmp'}]
stdout=pathlib.Path(str(out)+'.log')
if not stdout.is_file(): raise RuntimeError('missing trainer stdout log')
inventory={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in files}
with tarfile.open(bundle,'w:gz') as tar:
    for path in files: tar.add(path,arcname=str(path.relative_to(out)),recursive=False)
    tar.add(stdout,arcname='stdout.log',recursive=False)
manifest=dict(label=label,lease=GPU,source_output=str(out),source_commit='102ab99',
    checkpoint=dict(scratch_path=str(destination.relative_to('/scratch')),bytes=destination.stat().st_size,sha256=source_hash),
    bundle=dict(scratch_path=str(bundle.relative_to('/scratch')),bytes=bundle.stat().st_size,sha256=digest(bundle)),
    files=inventory,stdout=dict(bytes=stdout.stat().st_size,sha256=digest(stdout)))
manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
subprocess.run(['sync','/scratch'],check=True)
# A final read confirms the exact bytes that will survive releasing the lease.
if digest(destination)!=source_hash or digest(bundle)!=manifest['bundle']['sha256']:
    raise RuntimeError('post-sync preservation checksum mismatch')
print(json.dumps(manifest))
'''


class Supervisor:
    def __init__(self,args):
        self.args=args
        self.directory=ROOT/args.state_dir
        self.directory.mkdir(parents=True,exist_ok=True)
        self.state_file=self.directory/'state.json'
        self.logfile=self.directory/'supervisor.jsonl'
        self.local_runs=self.directory/'runs'
        previous=json.loads((ROOT/args.predecessors).read_text())
        if {r['label'] for r in previous}!=EXPECTED_SEED0 or len(previous)!=4:
            raise ValueError('requires exactly the four declared seed0 arms')
        self.allowed={r['gpu'] for r in previous}
        if len(self.allowed)!=4: raise ValueError('seed0 plan must have four distinct leases')
        if self.state_file.exists():
            self.state=json.loads(self.state_file.read_text())
            if set(self.state['allowed_leases'])!=self.allowed:
                raise ValueError('refusing changed lease plan on supervisor restart')
        else:
            self.state=dict(created_at=stamp(),deadline=time.time()+18*3600,allowed_leases=sorted(self.allowed),
                jobs={r['label']+'_seed0':dict(r,label=r['label']+'_seed0',seed=0,phase='running') for r in previous},
                released=[],errors={},launch_attempts={},report_complete=False)
        self.save()

    def save(self): atomic_json(self.state_file,self.state)

    def log(self,event,**details):
        record=dict(time=stamp(),event=event,**details)
        with self.logfile.open('a') as f: f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)

    def fail(self,key,error):
        self.state['errors'][key]=dict(time=stamp(),error=str(error),manual_required=True)
        self.save();self.log('manual_required',key=key,error=str(error))

    def assert_lease(self,gpu):
        if gpu not in self.allowed: raise ValueError('attempt to touch lease outside exact four-lease plan')

    def observe(self,job):
        self.assert_lease(job['gpu'])
        program='OUTPUT='+repr(job['output'])+'\n'+OBSERVE
        result=json.loads(cli('gpu','exec',job['gpu'],'--',PYTHON,'-c',program,timeout=120))
        local=self.local_runs/job['label'];local.mkdir(parents=True,exist_ok=True)
        for name,text in result['files'].items(): (local/name).write_text(text)
        (local/'stdout_tail.log').write_text(result['stdout_tail'])
        atomic_json(local/'lease.json',job)
        job['latest']=dict(train=result['train'],validation=result['validation'],finished=result['finished'])
        if result['wandb'] and job.get('wandb_path')!=result['wandb'][0] and job['label'] not in self.state['errors']:
            cli('gpu','declare',job['gpu'],'--name',job['name'],'--wandb-path',result['wandb'][0],timeout=60)
            job['wandb_path']=result['wandb'][0]
        self.log('progress',label=job['label'],step=(result['train'] or {}).get('step'),
                 finished=bool(result['finished']),active=bool(result['active']))
        if not result['active'] and not result['finished']:
            raise RuntimeError('trainer exited without finished event; no automatic restart')
        return bool(result['finished']) and not result['active']

    def preserve(self,job):
        program='OUTPUT='+repr(job['output'])+'\nLABEL='+repr(job['label'])+'\nGPU='+repr(job['gpu'])+'\nSCRATCH_ROOT='+repr(SCRATCH_ROOT)+'\n'+PRESERVE
        manifest=json.loads(cli('gpu','exec',job['gpu'],'--',PYTHON,'-c',program,timeout=280))
        local=self.local_runs/job['label'];local.mkdir(parents=True,exist_ok=True)
        bundle=self.directory/(job['label']+'-run_record.tar.gz')
        if not bundle.exists():
            cli('scratch','get',manifest['bundle']['scratch_path'],str(bundle),timeout=280)
        if hashlib.sha256(bundle.read_bytes()).hexdigest()!=manifest['bundle']['sha256']:
            raise RuntimeError('local compact bundle hash mismatch')
        with tarfile.open(bundle) as tar: tar.extractall(local,filter='data')
        atomic_json(local/'preservation_manifest.json',manifest)
        required=['source_record.txt','metrics.jsonl','config.json','environment.json','data_manifest.json','stdout.log']
        if any(not (local/p).is_file() for p in required): raise RuntimeError('local run record incomplete')
        # Verify every small-file byte, not only a successful transfer exit code.
        for name,entry in manifest['files'].items():
            path=local/name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
                raise RuntimeError('local extracted artifact hash mismatch: '+name)
        job.update(phase='preserved',preservation=manifest)
        self.save();self.log('preserved',label=job['label'],checkpoint=manifest['checkpoint'],bundle=manifest['bundle'])

    def launch_follow_on(self,job):
        arm=FOLLOW_ON[job['label']];label=arm+'_seed1'
        if label in self.state['jobs'] or label in self.state['errors']: return
        if label in self.state['launch_attempts']:
            raise RuntimeError('prior launch attempt is ambiguous; inspect seed1 launch record manually')
        # Save BEFORE invoking a mutation. A timeout/crash can never auto-replay it.
        self.state['launch_attempts'][label]=dict(time=stamp(),predecessor=job['label'])
        self.save()
        result=subprocess.run([sys.executable,str(ROOT/'scripts/launch_seed1_pair.py'),'--execute','--arm',arm,
                               '--predecessors',self.args.predecessors],cwd=ROOT,text=True,capture_output=True,timeout=max(1,min(600,self.state['deadline']-time.time())))
        (self.directory/(label+'-launch.log')).write_text(result.stdout+'\n'+result.stderr)
        if result.returncode: raise RuntimeError('seed1 launch failed/ambiguous; inspect '+label+'-launch.log')
        record=json.loads((ROOT/'runs/seed1_launch'/f'{arm}.json').read_text())
        if record.get('state')!='launched' or record['gpu']!=job['gpu']:
            raise RuntimeError('seed1 launch did not confirm the expected reused lease')
        self.assert_lease(record['gpu'])
        self.state['jobs'][label]=dict(record,phase='running')
        self.save();self.log('follow_on_started',label=label,gpu=job['gpu'])

    def release_if_unused(self,job):
        gpu=job['gpu'];self.assert_lease(gpu)
        if gpu in self.state['released']: return
        if job['label'] in FOLLOW_ON:
            successor=FOLLOW_ON[job['label']]+'_seed1'
            if successor not in self.state['jobs'] or self.state['jobs'][successor]['phase']!='preserved': return
        if any(r['gpu']==gpu and r['phase']!='preserved' for r in self.state['jobs'].values()): return
        if any(k in self.state['errors'] for k,r in self.state['jobs'].items() if r['gpu']==gpu): return
        final_job=next((r for r in self.state['jobs'].values() if r['gpu']==gpu and r.get('seed')==1),job)
        latest=final_job.get('latest',{}).get('validation') or {}
        args=['gpu','down',gpu,'--summary','Completed fixed3250-step planned runs; verified final checkpoints in scratch and compact source/log bundles locally.']
        if latest.get('step')==3250: args+=['--metric','val_loss='+str(latest['val_loss'])]
        self.state.setdefault('release_attempts',{})[gpu]=stamp();self.save()
        cli(*args,timeout=180)
        self.state['released'].append(gpu);self.save();self.log('released',gpu=gpu)

    def report(self,complete):
        paths=[self.local_runs/label for label in sorted(FINAL_LABELS)]
        # Missing planned runs get explicit config placeholders and empty metrics.
        for path in paths:
            if not path.exists():
                path.mkdir(parents=True)
                (path/'metrics.jsonl').write_text('')
                (path/'config.json').write_text('{}\n');(path/'environment.json').write_text('{}\n')
        output=self.directory/('report' if complete else 'partial_report')
        command=[sys.executable,str(ROOT/'scripts/summarize_runs.py'),*[str(x) for x in paths],
                 '--output',str(output),'--checkpoint','3250','--plot']
        result=subprocess.run(command,cwd=ROOT,text=True,capture_output=True,timeout=180)
        (self.directory/'summarizer.log').write_text(result.stdout+'\n'+result.stderr)
        if result.returncode: raise RuntimeError('summarizer failed; see summarizer.log')
        self.state['report_complete']=complete;self.save();self.log('report',complete=complete,path=str(output))

    def run(self):
        global COMMAND_DEADLINE
        COMMAND_DEADLINE=self.state['deadline']
        self.log('supervisor_started',deadline=self.state['deadline'],allowed_leases=sorted(self.allowed))
        while time.time()<self.state['deadline']:
            # No mutable run operations are retried after a recorded error.
            for label in list(self.state['jobs']):
                job=self.state['jobs'][label]
                if job['gpu'] in self.state['released']: continue
                if job['phase']=='running':
                    try:
                        finished=self.observe(job);self.save()
                        if finished and label not in self.state['errors']: self.preserve(job)
                    except Exception as error:
                        if label not in self.state['errors']: self.fail(label,error)
                if job['phase']=='preserved' and label not in self.state['errors']:
                    if label in FOLLOW_ON:
                        successor=FOLLOW_ON[label]+'_seed1'
                        try: self.launch_follow_on(job)
                        except Exception as error: self.fail(successor,error)
                    try:
                        if job['gpu'] not in self.state.get('release_attempts',{}) or job['gpu'] in self.state['released']:
                            self.release_if_unused(job)
                    except Exception as error: self.fail(label,error)
            if set(self.state['jobs'])==FINAL_LABELS and all(j['phase']=='preserved' for j in self.state['jobs'].values()):
                self.report(True)
                self.log('complete',released=self.state['released'],errors=self.state['errors']);return
            remaining=self.state['deadline']-time.time()
            if remaining<=0: break
            time.sleep(min(self.args.poll_seconds,remaining))
        self.log('deadline_reached',message='No new work launched; unfinished leases remain under their existing expiry bounds.')
        COMMAND_DEADLINE=None  # only local partial-report generation remains
        self.report(False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--predecessors',default='runs/panel_leases.json')
    parser.add_argument('--state-dir',default='runs/six_run_supervisor')
    parser.add_argument('--poll-seconds',type=int,default=120)
    args=parser.parse_args()
    if args.execute:
        raise SystemExit('SUPERSEDED: the 64-step-warmup six-run plan is disabled. Use only the reviewed baseline keeper; no learned full runs are authorized by this launcher.')
    if not 30<=args.poll_seconds<=600: raise ValueError('poll interval must be30..600seconds')
    if not args.execute:
        print(json.dumps(dict(dry_run=True,seed0=sorted(EXPECTED_SEED0),follow_on=FOLLOW_ON,
                             max_wall_hours=18,new_gpu_allocations=0,state_dir=args.state_dir),indent=2));return
    directory=ROOT/args.state_dir;directory.mkdir(parents=True,exist_ok=True)
    with (ROOT/'runs/six_run_supervisor.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit('another six-run supervisor owns the lock')
        lock.seek(0);lock.truncate();lock.write(str(os.getpid())+'\n');lock.flush()
        supervisor=Supervisor(args)
        try: supervisor.run()
        except Exception as error:
            supervisor.fail('supervisor',error);raise


if __name__=='__main__': main()
