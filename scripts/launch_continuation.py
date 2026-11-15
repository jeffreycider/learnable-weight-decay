"""Launch one explicitly prepared one-shot continuation on an existing8-H100 lease.

Dry-run by default. No allocation, sweep, retry, or parameter selection. Fork
root must be atomically finalized and synced by its producer before execution.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess

REMOTE=r'''
import hashlib,json,os,pathlib,shutil,subprocess,tarfile,time
s=SPEC;b=pathlib.Path(ARTIFACT);work=pathlib.Path('/workspace/lwd-oneshot-'+s['arm']);out=b/('oneshot-'+s['arm']+'-seed0')
def digest(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
if out.exists() or work.exists():raise RuntimeError('existing output/workdir; inspect ambiguous launch manually')
names=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).splitlines()
if len(names)!=8 or any('H100' not in n for n in names):raise RuntimeError('requires exactly8H100')
active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
if active:raise RuntimeError('refusing occupied GPU node: '+active)
source=pathlib.Path(s['source']);root=pathlib.Path(s['fork_root']);fork=root/s['arm']
if digest(source)!=s['source_sha256']:raise RuntimeError('source archive SHA256 mismatch')
manifest=json.loads((root/'manifest.json').read_text())
if manifest['benchmark_compliant'] is not False or s['arm'] not in manifest['arms']:raise RuntimeError('incomplete/wrong fork root')
arm=json.loads((fork/'manifest.json').read_text())
if arm!=manifest['arms'][s['arm']] or arm['continuation_start_step']!=1750 or arm['continuation_end_step']!=3250:raise RuntimeError('fork manifest/clock mismatch')
required={'config.json'}|{f'latest_rank{r}.pt' for r in range(8)}
if set(arm['files'])!=required:raise RuntimeError('fork must include config and all8rank checkpoints')
work.mkdir(parents=True)
with tarfile.open(source) as t:t.extractall(work,filter='data')
local=pathlib.Path('/tmp/lwd-oneshot-'+s['arm']+'-fork');local.mkdir()
for name,record in arm['files'].items():
 src=fork/name;dst=local/name
 if src.stat().st_size!=record['bytes'] or digest(src)!=record['sha256']:raise RuntimeError('fork source file hash mismatch: '+name)
 shutil.copyfile(src,dst)
 if digest(dst)!=record['sha256']:raise RuntimeError('local fork copy hash mismatch: '+name)
config=json.loads((local/'config.json').read_text())
if config['fork_label']!=s['arm'] or config['world_size']!=8 or config['seed']!=0 or config['benchmark_compliant'] is not False:raise RuntimeError('fork config mismatch')
data=pathlib.Path('/tmp/fineweb10B');data.mkdir(exist_ok=True)
for name in ['fineweb_val_000000.bin']+[f'fineweb_train_{i:06d}.bin' for i in range(1,19)]:
 if not (data/name).exists():shutil.copyfile(pathlib.Path('/scratch/datasets/fineweb10B')/name,data/name)
out.mkdir(parents=True)
shutil.copyfile(source,out/'source_archive.tar.gz');shutil.copyfile(local/'config.json',out/'fork_config.json')
(out/'fork_manifest.json').write_text(json.dumps(arm,indent=2)+'\n')
cmd=['/opt/venv211/bin/torchrun','--standalone','--nproc_per_node=8','-m','baseline.launch','--config',str(local/'config.json'),'--mode','baseline','--resume',str(local),'--output',str(out)]
env=dict(os.environ,PYTHONHASHSEED='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',TORCHINDUCTOR_COMPILE_THREADS='2')
with (out/'stdout.log').open('xb') as f:
 proc=subprocess.Popen(cmd,cwd=work,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(spec=s,command=cmd,pid=proc.pid,started=time.time(),output=str(out),work=str(work),local_fork=str(local),fork_manifest_sha256=digest(fork/'manifest.json'),source_sha256=s['source_sha256'],benchmark_compliant=False,remaining_gpt_steps=1500)
(out/'launch_manifest.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
'''


ARCHIVE=r'''
import hashlib,json,pathlib,shutil,subprocess,tarfile
s=SPEC;out=pathlib.Path(ARTIFACT)/('oneshot-'+s['arm']+'-seed0')
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
for proc in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
 try:args=proc.read_bytes().decode(errors='replace').split('\0')
 except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 if '--output' in args and args[args.index('--output')+1]==str(out):raise RuntimeError('continuation still active')
rows=[json.loads(x) for x in (out/'metrics.jsonl').read_text().splitlines()]
if not any(r.get('event')=='finished' and r['steps']==3250 for r in rows):raise RuntimeError('no completed3250 endpoint')
launch=json.loads((out/'launch_manifest.json').read_text())
if launch['source_sha256']!=s['source_sha256'] or launch['spec']['arm']!=s['arm']:raise RuntimeError('launch provenance mismatch')
target=pathlib.Path('/scratch/checkpoints/learnable-weight-decay/one_shot_continuations')/s['arm']/'final3250'
target.mkdir(parents=True,exist_ok=False)
files={}
for rank in range(8):
 src=out/f'final_rank{rank}.pt';dst=target/src.name
 expected=digest(src);shutil.copyfile(src,dst)
 if digest(dst)!=expected:raise RuntimeError('final checkpoint copy hash mismatch')
 files[dst.name]=dict(bytes=dst.stat().st_size,sha256=expected)
compact=target/'compact_record.tar.gz';inventory={}
with tarfile.open(compact,'w:gz') as t:
 for p in sorted(out.rglob('*')):
  if p.is_file() and '.pt' not in p.suffixes and not p.is_symlink():
   name=str(p.relative_to(out));inventory[name]=dict(bytes=p.stat().st_size,sha256=digest(p));t.add(p,arcname=name)
manifest=dict(arm=s['arm'],source_sha256=s['source_sha256'],benchmark_compliant=False,checkpoints=files,compact_inventory=inventory,
 compact_record=dict(path=str(compact),sha256=digest(compact),bytes=compact.stat().st_size),
 final_validation=[r for r in rows if r.get('event')=='validation'][-1],finished=[r for r in rows if r.get('event')=='finished'][-1])
(target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
subprocess.run(['sync','/scratch'],check=True);print(json.dumps(dict(directory=str(target),manifest=manifest)),flush=True)
'''


def cli(*args,timeout=280):
    return subprocess.run(['strangeloop',*args],check=True,capture_output=True,text=True,timeout=timeout).stdout


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('gpu','arm','source','source-sha256','fork-root'):p.add_argument('--'+arg,required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--archive',action='store_true');a=p.parse_args();s=vars(a).copy();s.pop('execute');s.pop('archive')
    if not re.fullmatch('[a-z][a-z0-9_]*',a.arm) or not re.fullmatch('[0-9a-f]{64}',a.source_sha256):p.error('unsafe arm/hash')
    if not a.source.startswith('/scratch/staging/') or not a.fork_root.startswith('/scratch/checkpoints/'):p.error('expected durable source/fork paths')
    status=json.loads(cli('gpu','status',a.gpu,'--json'))
    if status['gpu_count']!=8 or status['gpu_type']!='H100':p.error('requires8H100 lease')
    name='lwd-oneshot-'+a.arm.replace('_','-')+'-seed0'
    if not a.execute:print(json.dumps(dict(spec=s,name=name,artifact_dir=status['artifact_dir'],execute=False),indent=2));return
    record=Path(__file__).resolve().parents[1]/'runs'/'continuations'/a.arm;record.mkdir(parents=True,exist_ok=True)
    if a.archive:
        program='SPEC='+repr(s)+'\nARTIFACT='+repr(status['artifact_dir'])+'\n'+ARCHIVE
        result=cli('gpu','exec',a.gpu,'--','/opt/venv211/bin/python','-c',program)
        (record/'archive_output.txt').write_text(result);print(result);return
    marker=record/'launch_attempt.json'
    with marker.open('x') as f:f.write(json.dumps(s,indent=2)+'\n')
    cli('gpu','refresh',a.gpu)
    program='SPEC='+repr(s)+'\nARTIFACT='+repr(status['artifact_dir'])+'\n'+REMOTE
    (record/'remote_launcher.py').write_text(program)
    result=cli('gpu','exec',a.gpu,'--','/opt/venv211/bin/python','-c',program)
    (record/'launch_output.txt').write_text(result)
    print(result)
    print(cli('gpu','declare',a.gpu,'--name',name,'--description','One-shot frozen rewrite diagnostic; common1750-step prefix,1500 furtherPR323 steps; extra GPT preservation training, nonbenchmark.'))

if __name__=='__main__':main()
