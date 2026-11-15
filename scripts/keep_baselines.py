"""Preserve only the two already-running baselines; never launch any training.

Dry-run by default. Reviewed --execute polls every60s, captures seed1 rolling
250/500/750/1000 checkpoints, exports final3250 artifacts, then releases only
the corresponding finished/closed baseline GPU. Original deadline is6hours.
Read-only transient failures retry next poll; mutation failures require manual
inspection and are never automatically replayed.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1]
PYTHON='/opt/venv211/bin/python'
JOBS={
 'baseline_seed0':dict(gpu='6470a9a9-c41d-47cc-9048-9048caa100b1',seed=0,
    output='/persist/data/runs/6470a9a9-c41d-47cc-9048-9048caa100b1/full-baseline',name='lwd-full-baseline-seed0'),
 'baseline_seed1':dict(gpu='e3e0f6ab-e6bf-4c60-9f32-cca58afb0447',seed=1,
    output='/persist/data/runs/e3e0f6ab-e6bf-4c60-9f32-cca58afb0447/warmup1000-baseline-seed1',name='lwd-warmup1000-baseline-seed1')}
TARGETS={250,500,750,1000}
COMMAND_DEADLINE=None
_spec=importlib.util.spec_from_file_location('_archive_helpers',ROOT/'scripts/supervise_six_runs.py')
_archive=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_archive)
OBSERVE=_archive.OBSERVE
# Baseline1 includes a thread-pinning execution wrapper; record actual source
# bytes instead of falsely claiming its entire source is exactly102ab99.
PRESERVE=_archive.PRESERVE.replace("source_commit='102ab99',",'')


def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');os.replace(temporary,path)


def cli(*args,timeout=280):
    if COMMAND_DEADLINE is not None:
        remaining=COMMAND_DEADLINE-time.time()
        if remaining<=0:raise TimeoutError('keeper deadline reached')
        timeout=min(timeout,remaining)
    result=subprocess.run(['strangeloop',*args],text=True,capture_output=True,timeout=timeout)
    if result.returncode:raise RuntimeError(f'CLI {args[:3]} failed: {result.stderr[-4000:]}')
    return result.stdout


CAPTURE=r'''
import pathlib,torch,hashlib,json,os,subprocess,time
src=pathlib.Path(OUTPUT)/'latest_rank0.pt'
root=pathlib.Path('/scratch/checkpoints/learnable-weight-decay/baseline_keeper')
root.mkdir(parents=True,exist_ok=True)
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
results=[]
for step in WANTED:
 manifest=root/('seed1_step'+str(step)+'.json')
 if manifest.exists():
  row=json.loads(manifest.read_text());dest=pathlib.Path(row['destination'])
  if row['seed']!=1 or row['step']!=step or digest(dest)!=row['sha256']:
   raise RuntimeError('existing archived rolling checkpoint failed verification')
  results.append(row)
if not src.exists():
 print(json.dumps(dict(preserved=results,latest_step=None)));raise SystemExit()
state=torch.load(src,map_location='cpu',weights_only=False,mmap=True)
latest=int(state['step'])
assert state['config']['seed']==1 and state['config']['mode']=='baseline'
assert state['data_cursor']['batches']==latest
del state
remaining=set(WANTED)-{r['step'] for r in results}
if latest not in remaining:
 print(json.dumps(dict(preserved=results,latest_step=latest)));raise SystemExit()
if list(root.glob('.incoming_seed1_*.pt')):
 raise RuntimeError('ambiguous incomplete rolling checkpoint copy; inspect manually')
tmp=root/('.incoming_seed1_'+str(os.getpid())+'.pt');h=hashlib.sha256()
# Open inode is stable if the trainer atomically replaces latest during copying.
with src.open('rb') as inp,tmp.open('xb') as out:
 for block in iter(lambda:inp.read(8*1024*1024),b''):
  h.update(block);out.write(block)
saved=torch.load(tmp,map_location='cpu',weights_only=False,mmap=True)
actual=int(saved['step'])
assert saved['config']['seed']==1 and saved['config']['mode']=='baseline'
assert saved['data_cursor']['batches']==actual
del saved
if actual not in remaining:
 tmp.unlink();print(json.dumps(dict(preserved=results,latest_step=actual)));raise SystemExit()
dest=root/('seed1_step'+str(actual)+'.pt')
if dest.exists():raise RuntimeError('refusing to overwrite unmanifested rolling checkpoint')
os.replace(tmp,dest)
row=dict(step=actual,seed=1,source=str(src),destination=str(dest),sha256=h.hexdigest(),bytes=dest.stat().st_size,time=time.time())
(root/('seed1_step'+str(actual)+'.json')).write_text(json.dumps(row,indent=2)+'\n')
subprocess.run(['sync','/scratch'],check=True)
if digest(dest)!=row['sha256']:raise RuntimeError('rolling checkpoint copy checksum mismatch')
results.append(row)
print(json.dumps(dict(preserved=results,latest_step=actual)))
'''


class Keeper:
    def __init__(self,directory):
        self.directory=directory;directory.mkdir(parents=True,exist_ok=True)
        self.path=directory/'state.json'
        self.state=json.loads(self.path.read_text()) if self.path.exists() else dict(
            created=time.time(),deadline=time.time()+6*3600,jobs={k:dict(v) for k,v in JOBS.items()},
            rolling={},errors={},attempts={},released=[],preserved={})
        if self.state['jobs']!=JOBS:raise ValueError('keeper state differs from exact two-baseline allowlist')
        self.save()

    def save(self):atomic_json(self.path,self.state)

    def log(self,event,**values):
        if 'time' in values:
            values=dict(values)
            values['source_time']=values.pop('time')
        row=dict(time=time.time(),event=event,**values)
        with (self.directory/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)

    def error(self,key,error):
        self.state['errors'][key]=str(error);self.save();self.log('manual_required',key=key,error=str(error))

    def remote(self,label,program,timeout=280):
        if label not in JOBS:raise ValueError('outside the exact two-baseline allowlist')
        remaining=self.state['deadline']-time.time()
        if remaining<=0:raise TimeoutError('keeper deadline reached')
        return json.loads(cli('gpu','exec',JOBS[label]['gpu'],'--',PYTHON,'-c',program,timeout=max(1,min(timeout,remaining))))

    def observe(self,label):
        job=JOBS[label]
        result=self.remote(label,'OUTPUT='+repr(job['output'])+'\n'+OBSERVE,timeout=90)
        directory=self.directory/'runs'/label;directory.mkdir(parents=True,exist_ok=True)
        for name,text in result['files'].items():(directory/name).write_text(text)
        (directory/'stdout_tail.log').write_text(result['stdout_tail'])
        if 'config.json' in result['files']:
            config=json.loads(result['files']['config.json'])
            if config['mode']!='baseline' or config['seed']!=job['seed'] or config['max_steps']!=3250:
                raise ValueError('running job identity does not match approved baseline')
        self.log('progress',label=label,step=(result['train'] or {}).get('step'),finished=bool(result['finished']))
        return result

    def mutate(self,key,action):
        if key in self.state['errors']:return None
        if key in self.state['attempts']:
            self.error(key,'prior mutation may have completed; inspect before manually clearing attempt');return None
        self.state['attempts'][key]=time.time();self.save()
        try:result=action()
        except Exception as error:self.error(key,error);return None
        del self.state['attempts'][key];self.save()
        return result

    def capture_rolling(self):
        wanted=sorted(TARGETS-{int(x) for x in self.state['rolling']})
        if not wanted:return
        label='baseline_seed1'
        program='OUTPUT='+repr(JOBS[label]['output'])+'\nWANTED='+repr(wanted)+'\n'+CAPTURE
        result=self.mutate('capture_rolling',lambda:self.remote(label,program))
        if result is None:return
        for row in result['preserved']:
            self.state['rolling'][str(row['step'])]=row
            atomic_json(self.directory/'rolling'/f"seed1_step{row['step']}.json",row)
            self.log('rolling_preserved',**row)
        latest=result['latest_step']
        if latest is not None:
            missed=sorted(x for x in TARGETS if x<latest and str(x) not in self.state['rolling'])
            if missed:
                self.state['missed_rolling_checkpoints']=missed
                self.log('missed_rolling_checkpoint',steps=missed,latest=latest)
        self.save()

    def preserve_final(self,label):
        job=JOBS[label]
        program=('OUTPUT='+repr(job['output'])+'\nLABEL='+repr(label)+'\nGPU='+repr(job['gpu'])
                 +'\nSCRATCH_ROOT='+repr('learnable-weight-decay/baseline_keeper/final')+'\n'+PRESERVE)
        manifest=self.remote(label,program)
        bundle=self.directory/(label+'-final_record.tar.gz')
        if not bundle.exists():cli('scratch','get',manifest['bundle']['scratch_path'],str(bundle),timeout=280)
        if hashlib.sha256(bundle.read_bytes()).hexdigest()!=manifest['bundle']['sha256']:
            raise RuntimeError('local final bundle checksum mismatch')
        local=self.directory/'runs'/label
        with tarfile.open(bundle) as archive:archive.extractall(local,filter='data')
        for name,row in manifest['files'].items():
            if hashlib.sha256((local/name).read_bytes()).hexdigest()!=row['sha256']:
                raise RuntimeError('local final record checksum mismatch: '+name)
        atomic_json(local/'preservation_manifest.json',manifest)
        self.state['preserved'][label]=manifest;self.save();self.log('final_preserved',label=label,checkpoint=manifest['checkpoint'])
        return manifest

    def register_wandb(self,label,result):
        if not result['wandb']:return
        path=result['wandb'][0];key='wandb_'+label
        if self.state.get('wandb',{}).get(label)==path:return
        def register():
            cli('gpu','declare',JOBS[label]['gpu'],'--name',JOBS[label]['name'],'--wandb-path',path,timeout=90)
            self.state.setdefault('wandb',{})[label]=path;self.save();return True
        self.mutate(key,register)

    def release(self,label,result):
        if label not in self.state['preserved']:raise RuntimeError('cannot release before verified final archival')
        if result['active'] or not result['finished']:raise RuntimeError('cannot release active/unfinished baseline')
        val=result['validation'] or {}
        args=['gpu','down',JOBS[label]['gpu'],'--summary','Completed baseline3250; final checkpoint verified in scratch and source/log/data/W&B bundle verified locally.']
        if val.get('step')==3250:args+=['--metric','val_loss='+str(val['val_loss'])]
        cli(*args,timeout=180)
        self.state['released'].append(label);self.save();self.log('released',label=label,gpu=JOBS[label]['gpu'])
        return True

    def run(self):
        global COMMAND_DEADLINE
        COMMAND_DEADLINE=self.state['deadline']
        self.log('keeper_started',deadline=self.state['deadline'],jobs=JOBS)
        while time.time()<self.state['deadline']:
            for label in JOBS:
                if label in self.state['released']:continue
                try:result=self.observe(label)
                except Exception as error:
                    self.log('read_error_retry_next_poll',label=label,error=str(error));continue
                self.register_wandb(label,result)
                if label=='baseline_seed1':self.capture_rolling()
                if result['finished'] and not result['active']:
                    if label not in self.state['preserved']:
                        self.mutate('preserve_'+label,lambda:self.preserve_final(label))
                    # A rolling-copy failure blocks seed1 release until manually inspected.
                    blocked=('wandb_'+label in self.state['errors'] or
                             (label=='baseline_seed1' and 'capture_rolling' in self.state['errors']))
                    if label in self.state['preserved'] and not blocked:
                        self.mutate('release_'+label,lambda:self.release(label,result))
                elif not result['active'] and not result['finished']:
                    self.log('baseline_exited_without_finish',label=label,manual_required=True)
            if set(self.state['released'])==set(JOBS):
                self.log('keeper_complete',missed_rolling=self.state.get('missed_rolling_checkpoints',[]));return
            time.sleep(max(0,min(60,self.state['deadline']-time.time())))
        self.log('deadline_reached',message='No further mutations; no training was launched or interrupted.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(dry_run=True,jobs=JOBS,rolling_seed1=sorted(TARGETS),max_hours=6,poll_seconds=60,new_training_runs=0),indent=2));return
    directory=ROOT/'runs/baseline_keeper';directory.mkdir(parents=True,exist_ok=True)
    with (ROOT/'runs/baseline_keeper.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('another baseline keeper owns the lock')
        lock.seek(0);lock.truncate();lock.write(str(os.getpid())+'\n');lock.flush()
        Keeper(directory).run()


if __name__=='__main__':main()
