"""Launch one explicitly configured multi-state fit on an existing8-H100 lease.

No allocation, sweeps, mutation retries, or performance-based selection. Stages a
fixed committed prefix before launching torchrun; records hashes and timestamps.
"""
import argparse,hashlib,json,subprocess
from pathlib import Path

PYTHON='/opt/venv211/bin/python'


def command_arguments(spec,work,bank,data,output):
    args=['/opt/venv211/bin/torchrun','--standalone','--nproc_per_node='+str(spec['world_size']),
          str(Path(work)/'scripts/train_multistate_chisel.py'),'--state-bank',str(bank),
          '--train-data',str(Path(data)/'fineweb_train_*.bin'),'--output',str(output)]
    for key,value in spec['parameters'].items():
        if value is False:continue
        args.append('--'+key.replace('_','-'))
        if value is not True:args.append(str(value))
    return args


WORKER=r'''
import hashlib,importlib.util,json,os,pathlib,shutil,struct,subprocess,sys,tarfile,time
spec=SPEC;p=PATHS
record=pathlib.Path(p['record']);work=pathlib.Path(p['work']);bank=pathlib.Path(p['bank']);data=pathlib.Path(p['data'])
def event(name,**values):
 row=dict(event=name,time=time.time(),**values)
 with (record/'orchestration.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 print(json.dumps(row),flush=True)
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
event('staging_started')
try:
 source=pathlib.Path(SOURCE)
 if digest(source)!=SOURCE_SHA:raise RuntimeError('source archive SHA mismatch')
 work.mkdir(exist_ok=False,parents=True)
 with tarfile.open(source) as tar:tar.extractall(work,filter='data')
 shutil.copyfile(source,record/'source.tar.gz')
 incoming=pathlib.Path(spec['state_bank']);manifest=json.loads((incoming/'manifest.json').read_text())
 available=json.loads((incoming/'available.json').read_text())
 prefix=spec['parameters']['diagnostic_prefix_states']
 if available['completed_steps']<prefix:raise RuntimeError('requested prefix is not yet published/synced')
 # available marker is published only after the writer's declared sync boundary.
 entries=[json.loads(x) for x in (incoming/'index.jsonl').read_text().splitlines()[:prefix] if x]
 if len(entries)!=prefix or [x['completed_steps'] for x in entries]!=list(range(1,prefix+1)):
  raise RuntimeError('requested complete prefix is not published')
 bank.mkdir(exist_ok=False,parents=True)
 (bank/'manifest.json').write_text(json.dumps(manifest,indent=2))
 (bank/'index.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in entries))
 for item in entries:
  source=incoming/item['file'];dest=bank/item['file']
  shutil.copyfile(source,dest)
  if digest(dest)!=item['sha256']:raise RuntimeError('state record copy hash mismatch: '+item['file'])
 config=manifest['provenance']['training_config'];batch=config['batch_tokens']
 shared=pathlib.Path('/scratch/datasets/fineweb10B');paths=sorted(shared.glob('fineweb_train_*.bin'))
 def length(index):
  with paths[index].open('rb') as f:header=f.read(1024)
  magic,version,count=struct.unpack('<iii',header[:12])
  if (magic,version)!=(20240520,1):raise RuntimeError('invalid dataset header')
  return count
 index=position=0
 for _ in range(prefix):
  if position+batch+1>=length(index):index+=1;position=0
  position+=batch
 data.mkdir(exist_ok=False,parents=True)
 pins={x['path']:x['lfs']['oid'] for x in json.loads((work/'baseline/vendor/dataset_files.json').read_text()) if 'lfs' in x}
 staged=[]
 for path in paths[:index+1]:
  dest=data/path.name;shutil.copyfile(path,dest);actual=digest(dest)
  if actual!=pins[path.name]:raise RuntimeError('dataset LFS SHA mismatch')
  staged.append(dict(file=path.name,sha256=actual,bytes=dest.stat().st_size))
 (record/'staging_manifest.json').write_text(json.dumps(dict(source_sha256=SOURCE_SHA,available_marker=available,
  prefix_states=prefix,entries=entries,dataset=staged,source_bank=str(incoming),local_bank=str(bank)),indent=2))
 if importlib.util.find_spec('matplotlib') is None:
  subprocess.run([sys.executable,'-m','pip','install','matplotlib==3.10.7'],check=True)
 with (record/'pip_freeze.txt').open('w') as f:subprocess.run([sys.executable,'-m','pip','freeze'],check=True,stdout=f)
 event('prefix_staged',states=prefix,weight_and_observation_bytes=sum(x['file_bytes'] for x in entries))
 env=dict(os.environ,PYTHONHASHSEED='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',
          TORCH_LOGS='recompiles,graph_breaks',TORCHINDUCTOR_CACHE_DIR='/tmp/chisel-multistate-compiler-cache')
 with (record/'training.log').open('xb') as log:
  proc=subprocess.Popen(COMMAND,cwd=work,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  event('training_started',pid=proc.pid,command=COMMAND)
  result=proc.wait()
 if result:raise RuntimeError('torchrun failed with exit '+str(result))
 event('training_finished')
except BaseException as error:
 event('failed',error=repr(error));raise
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',required=True);p.add_argument('--spec',required=True,type=Path)
    p.add_argument('--source',required=True);p.add_argument('--source-sha256',required=True)
    p.add_argument('--name',default='chisel-multistate-prefix25-v1');p.add_argument('--execute',action='store_true')
    a=p.parse_args();spec=json.loads(a.spec.read_text())
    if spec['world_size']!=8 or spec['parameters']['diagnostic_prefix_states']<5:
        raise ValueError('this launcher requires the explicit8-GPU diagnostic prefix spec')
    if not a.name.replace('-','').replace('_','').isalnum():raise ValueError('unsafe name')
    if len(a.source_sha256)!=64 or any(c not in '0123456789abcdef' for c in a.source_sha256):raise ValueError('invalid source hash')
    status=json.loads(subprocess.check_output(['strangeloop','gpu','status',a.gpu,'--json'],text=True))
    if status['status']!='ready' or status['gpu_count']!=8:raise RuntimeError('requires ready8-GPU lease')
    paths=dict(record=status['artifact_dir']+'/'+a.name,work='/workspace/'+a.name,
               bank='/tmp/'+a.name+'-bank',data='/tmp/'+a.name+'-data',output=status['artifact_dir']+'/'+a.name+'/fit')
    command=command_arguments(spec,paths['work'],paths['bank'],paths['data'],paths['output'])
    manifest=dict(gpu=a.gpu,spec=spec,source=a.source,source_sha256=a.source_sha256,paths=paths,command=command)
    if not a.execute:print(json.dumps(dict(dry_run=True,**manifest),indent=2));return
    local=Path(__file__).resolve().parents[1]/'runs'/a.name;local.mkdir(exist_ok=True,parents=True)
    attempt=local/'launch_attempt.json'
    if attempt.exists():raise RuntimeError('prior launch attempt exists; inspect status, never automatically replay')
    attempt.write_text(json.dumps(manifest,indent=2)+'\n')
    # Reader refresh only before launching any process that opens the shared mount.
    subprocess.run(['strangeloop','gpu','refresh',a.gpu],check=True)
    assignments=dict(SPEC=spec,PATHS=paths,SOURCE=a.source,SOURCE_SHA=a.source_sha256,COMMAND=command)
    worker='\n'.join(k+'='+repr(v) for k,v in assignments.items())+'\n'+WORKER
    launch='''import json,os,pathlib,subprocess,time
p=PATHS;record=pathlib.Path(p['record'])
if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU compute already active')
record.mkdir(exist_ok=False,parents=True)
(record/'worker.py').write_text(WORKER)
with (record/'orchestrator.log').open('xb') as log:
 proc=subprocess.Popen(['/opt/venv211/bin/python',str(record/'worker.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
result=dict(pid=proc.pid,launched_at=time.time(),paths=p)
(record/'launch_manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
'''
    program='PATHS='+repr(paths)+'\nWORKER='+repr(worker)+'\n'+launch
    output=subprocess.check_output(['strangeloop','gpu','exec',a.gpu,'--',PYTHON,'-c',program],text=True,timeout=90)
    (local/'launch_manifest.json').write_text(output)
    subprocess.run(['strangeloop','gpu','declare',a.gpu,'--name',a.name,'--description','Fresh same-run25-state Chisel diagnostic,100epochs,300sharedAdamupdates; held-out states and sequences; extra GPT computation.'],check=True)
    print(output)


if __name__=='__main__':main()
