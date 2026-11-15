"""Explicit-spec launch/poll/archive for frozen-preservation prototypes.

No allocations, sweeps, retries of launches, GPU release, or configuration
selection. launch/archive are dry-run unless --execute; poll is read-only.
The reviewed spec pins one already-owned GPU, exact source/checkpoint hashes,
structured trainer arguments. The trainer's step count and existing lease bound
the run; an additional wall cap is used only when explicitly present in the spec. Independent specs may
be launched on separate idle GPUs by the coordinator.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
PYTHON='/opt/venv211/bin/python'
IMAGE='muoff-torch211-wandb-resume-20261011'
REQUIRED={'name','gpu','source_archive','source_sha256','checkpoint','checkpoint_sha256','parameters'}
RESERVED={'checkpoint','output','train_data','wandb'}


def validate_spec(spec):
    missing=REQUIRED-spec.keys()
    if missing:raise ValueError('missing spec fields: '+str(sorted(missing)))
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,100}',spec['name']):raise ValueError('unsafe run name')
    for key in ('source_sha256','checkpoint_sha256'):
        if not re.fullmatch('[0-9a-f]{64}',spec[key]):raise ValueError('invalid SHA256: '+key)
    if not spec['source_archive'].startswith('/scratch/staging/') or not spec['checkpoint'].startswith('/scratch/checkpoints/'):
        raise ValueError('source/checkpoint must use reviewed durable scratch locations')
    if not isinstance(spec['parameters'],dict):raise ValueError('parameters must be a structured mapping')
    for key,value in spec['parameters'].items():
        if key in RESERVED or not re.fullmatch('[a-z][a-z0-9_]*',key):raise ValueError('reserved/invalid parameter: '+key)
        if not isinstance(value,(str,int,float,bool)) or (isinstance(value,float) and not math.isfinite(value)):
            raise ValueError('parameters must be finite scalar values')
    if 'wall_time_seconds' in spec and (not isinstance(spec['wall_time_seconds'],int) or spec['wall_time_seconds']<60):
        raise ValueError('an explicit wall_time_seconds must be an integer of at least60')
    if not 1<=spec.get('stage_train_shards',18)<=103:raise ValueError('stage_train_shards must be1..103')
    return spec


def trainer_arguments(spec,work,output):
    args=[PYTHON,str(Path(work)/'scripts/train_frozen_preservation.py'),'--checkpoint',spec['checkpoint'],
          '--output',output,'--train-data','/tmp/fineweb10B/fineweb_train_*.bin','--wandb']
    for key,value in sorted(spec['parameters'].items()):
        if value is False:continue
        args.append('--'+key.replace('_','-'))
        if value is not True:args.append(str(value))
    return args


def cli(*args,timeout=280):
    result=subprocess.run(['strangeloop',*args],check=True,text=True,capture_output=True,timeout=timeout)
    return result.stdout


def local_dir(spec):return ROOT/'runs/frozen_managed'/spec['name']


def state_paths(spec,status):
    base=status['artifact_dir']
    return dict(output=base+'/frozen-'+spec['name'],work='/workspace/frozen-'+spec['name'],
                launch=base+'/frozen-'+spec['name']+'.launch.json',
                source=base+'/frozen-'+spec['name']+'.source.tar.gz',
                lock=base+'/frozen-'+spec['name']+'.launch.lock')


LAUNCH=r'''
import ast,hashlib,json,os,pathlib,shutil,subprocess,tarfile,time
p=PATHS;s=SPEC
out=pathlib.Path(p['output']);work=pathlib.Path(p['work'])
if out.exists() or pathlib.Path(p['lock']).exists():
 raise RuntimeError('existing output/launch marker: do not blindly replay this trial')
active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
if active:raise RuntimeError('GPU has active compute processes; refusing to disturb them: '+active)
def digest(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
if digest(s['source_archive'])!=s['source_sha256']:raise RuntimeError('source archive SHA256 mismatch')
if digest(s['checkpoint'])!=s['checkpoint_sha256']:raise RuntimeError('input checkpoint SHA256 mismatch')
# Atomic marker is deliberately never removed automatically, even after failure.
fd=os.open(p['lock'],os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.close(fd)
if work.exists():raise RuntimeError('refusing existing worktree; inspect failed staging manually')
work.mkdir(parents=True)
with tarfile.open(s['source_archive']) as archive:archive.extractall(work,filter='data')
runner=work/'scripts/train_frozen_preservation.py'
tree=ast.parse(runner.read_text());allowed=set()
for node in ast.walk(tree):
 if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument':
  allowed.update(a.value for a in node.args if isinstance(a,ast.Constant) and isinstance(a.value,str) and a.value.startswith('--'))
unknown={'--'+k.replace('_','-') for k in s['parameters']}-allowed
if unknown:raise RuntimeError('parameters absent from frozen runner: '+str(unknown))
source_hashes={str(x.relative_to(work)):digest(x) for parent in ('baseline','rewriter','scripts')
               for x in (work/parent).rglob('*.py')}
shutil.copyfile(s['source_archive'],p['source'])
if digest(p['source'])!=s['source_sha256']:raise RuntimeError('durable source-copy hash mismatch')
data=pathlib.Path('/tmp/fineweb10B');data.mkdir(parents=True,exist_ok=True)
for i in range(1,s.get('stage_train_shards',18)+1):
 name=f'fineweb_train_{i:06d}.bin';destination=data/name
 if not destination.exists():shutil.copyfile('/scratch/datasets/fineweb10B/'+name,destination)
env=dict(os.environ,PYTHONHASHSEED='0',CUBLAS_WORKSPACE_CONFIG=':4096:8')
local_checkpoint=pathlib.Path('/tmp')/('lwd-checkpoint-'+s['checkpoint_sha256']+'.pt')
if not local_checkpoint.exists():shutil.copyfile(s['checkpoint'],local_checkpoint)
if digest(local_checkpoint)!=s['checkpoint_sha256']:raise RuntimeError('local checkpoint SHA256 mismatch')
command=[str(local_checkpoint) if value==s['checkpoint'] else value for value in COMMAND]
if 'wall_time_seconds' in s:
 command=['timeout','--signal=TERM','--kill-after=30s',str(s['wall_time_seconds'])]+command
log_path=pathlib.Path(p['output']+'.log')
with log_path.open('xb') as log:
 proc=subprocess.Popen(command,cwd=work,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
manifest=dict(spec=s,paths=p,pid=proc.pid,command=command,source_hashes=source_hashes,
              launched_at=time.time(),benchmark_compliant=False)
pathlib.Path(p['launch']).write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest),flush=True)
'''


POLL=r'''
import json,pathlib,os
p=PATHS;out=pathlib.Path(p['output']);launch=pathlib.Path(p['launch'])
manifest=json.loads(launch.read_text()) if launch.exists() else None
files={name:(out/name).read_text() for name in ['config.json','metrics.jsonl','replay_provenance.json'] if (out/name).exists()}
text=files.get('metrics.jsonl','');text=text[:text.rfind('\n')+1] if text else ''
files['metrics.jsonl']=text
rows=[json.loads(x) for x in text.splitlines() if x]
active=[]
for proc in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
 try:args=proc.read_bytes().decode(errors='replace').split('\x00')
 except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 if '--output' in args and args[args.index('--output')+1]==p['output']:active.append(proc.parent.name)
stdout=pathlib.Path(p['output']+'.log');tail=''
if stdout.exists():
 with stdout.open('rb') as f:f.seek(max(0,stdout.stat().st_size-16384));tail=f.read().decode(errors='replace')
print(json.dumps(dict(manifest=manifest,files=files,active=active,latest=rows[-1] if rows else None,
 training=[x for x in rows if x.get('event')=='train'][-1:],
 evaluations=[x for x in rows if x.get('event')=='evaluation'][-4:],
 finished=any(x.get('event')=='finished' for x in rows),stdout_tail=tail,
 wandb=[str(x.parent) for x in (out/'wandb').glob('offline-run-*/*.wandb')])))
'''


ARCHIVE=r'''
import json,pathlib,hashlib,tarfile,subprocess
p=PATHS;s=SPEC;out=pathlib.Path(p['output'])
for proc in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
 try:args=proc.read_bytes().decode(errors='replace').split('\x00')
 except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 if '--output' in args and args[args.index('--output')+1]==p['output']:
  raise RuntimeError('prototype still active; poll instead of archiving partial files')
if not pathlib.Path(p['launch']).is_file():raise RuntimeError('no launch manifest; inspect ambiguous launch manually')
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
root=pathlib.Path('/scratch/artifacts/learnable-weight-decay/frozen_runs')/s['name']
root.mkdir(parents=True,exist_ok=True)
if (root/'archive_manifest.json').exists():raise RuntimeError('already archived; do not repeat mutation')
files=[(x,str(x.relative_to(out))) for x in out.rglob('*') if x.is_file() and not x.is_symlink()] if out.exists() else []
files += [(pathlib.Path(p['launch']),'launch_manifest.json'),(pathlib.Path(p['source']),'source_bundle.tar.gz'),
          (pathlib.Path(p['output']+'.log'),'stdout.log')]
if any(not x.is_file() for x,_ in files):raise RuntimeError('missing launch/source/stdout provenance')
inventory={name:dict(bytes=x.stat().st_size,sha256=digest(x)) for x,name in files}
artifacts={}
for kind in ('full','compact'):
 archive=root/(kind+'_record.tar.gz')
 with tarfile.open(archive,'x:gz') as tar:
  for path,name in files:
   if kind=='compact' and path.suffix in {'.pt','.bin','.safetensors'}:continue
   tar.add(path,arcname=name,recursive=False)
 artifacts[kind]=dict(scratch_path=str(archive.relative_to('/scratch')),bytes=archive.stat().st_size,sha256=digest(archive))
manifest=dict(name=s['name'],gpu=s['gpu'],files=inventory,archives=artifacts,spec=s,benchmark_compliant=False)
(root/'archive_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
subprocess.run(['sync','/scratch'],check=True)
for entry in artifacts.values():
 if digest(pathlib.Path('/scratch')/entry['scratch_path'])!=entry['sha256']:raise RuntimeError('post-sync archive checksum mismatch')
print(json.dumps(manifest))
'''


def remote(spec,program,paths,extra=''):
    return json.loads(cli('gpu','exec',spec['gpu'],'--',PYTHON,'-c',
                         'SPEC='+repr(spec)+'\nPATHS='+repr(paths)+'\n'+extra+program))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=['launch','poll','archive'])
    parser.add_argument('--spec',required=True,type=Path)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();spec=validate_spec(json.loads(args.spec.read_text()))
    if args.operation!='poll' and not args.execute:
        print(json.dumps(dict(dry_run=True,operation=args.operation,spec=spec,new_gpu_allocations=0),indent=2));return
    local=local_dir(spec);local.mkdir(parents=True,exist_ok=True)
    status=json.loads(cli('gpu','status',spec['gpu'],'--json',timeout=60))
    if status.get('status')!='ready' or status.get('gpu_count',1)!=1 or status.get('image')!=IMAGE:
        raise RuntimeError('requires one ready GPU in the approved Torch2.11 image')
    paths=state_paths(spec,status)
    if args.operation=='launch':
        attempt=local/'launch_attempt.json'
        if attempt.exists():raise RuntimeError('prior launch attempt exists; poll/inspect, never auto-replay')
        attempt.write_text(json.dumps(dict(spec=spec,paths=paths),indent=2)+'\n')
        cli('gpu','declare',spec['gpu'],'--name',spec['name'],'--description','Frozen finite-output-preservation prototype; additional GPT evaluations counted; not Track3 compliant.')
        result=remote(spec,LAUNCH,paths,'COMMAND='+repr(trainer_arguments(spec,paths['work'],paths['output']))+'\n')
        (local/'launch_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(name=spec['name'],pid=result['pid'],output=paths['output'])));return
    if args.operation=='poll':
        result=remote(spec,POLL,paths)
        for name,text in result['files'].items():(local/name).write_text(text)
        (local/'stdout_tail.log').write_text(result['stdout_tail'])
        if result['manifest']:(local/'launch_manifest.json').write_text(json.dumps(result['manifest'],indent=2)+'\n')
        # Reading metrics is separate from explicit offline-run registration.
        if result['wandb'] and args.execute:
            cli('gpu','declare',spec['gpu'],'--name',spec['name'],'--wandb-path',result['wandb'][0],timeout=60)
        summary={k:v for k,v in result.items() if k not in ('files','manifest','stdout_tail')}
        (local/'status.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary));return
    marker=local/'archive_attempt.json'
    if marker.exists():raise RuntimeError('prior archive attempt exists; inspect before retrying mutation')
    marker.write_text(json.dumps(dict(spec=spec,paths=paths),indent=2)+'\n')
    result=remote(spec,ARCHIVE,paths)
    bundle=local/'compact_record.tar.gz';entry=result['archives']['compact']
    cli('scratch','get',entry['scratch_path'],str(bundle))
    if hashlib.sha256(bundle.read_bytes()).hexdigest()!=entry['sha256']:raise RuntimeError('local compact archive hash mismatch')
    with tarfile.open(bundle) as tar:tar.extractall(local,filter='data')
    for name,metadata in result['files'].items():
        if Path(name).suffix in {'.pt','.bin','.safetensors'}:continue
        if hashlib.sha256((local/name).read_bytes()).hexdigest()!=metadata['sha256']:raise RuntimeError('local record mismatch: '+name)
    (local/'archive_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['archives']))


if __name__=='__main__':main()
