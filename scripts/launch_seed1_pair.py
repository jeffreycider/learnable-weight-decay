"""Start predeclared seed1 reference/control as finished seed0 leases become free.

Dry-run by default. Never allocates GPUs, resumes checkpoints, interrupts an
existing trainer, or modifies any seed0 source/output. Re-run after a predecessor
finishes: ready arms start independently, so reference1 never waits for baseline1.
"""
import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess

COMMIT = '102ab99'
ARCHIVE = '/scratch/staging/lwd-source-20261108-deterministic.tar.gz'
ARCHIVE_SHA = '13258068b9666c4fdc044296651d39678e9846b30adfd33171474742783a8db3'
IMAGE = 'muoff-torch211-wandb-resume-20261011'
PYTHON = '/opt/venv211/bin/python'
ARMS = {'reference':dict(predecessor='baseline',mode='learned',lease_hours=12),
        'baseline':dict(predecessor='k2',mode='baseline',lease_hours=4)}
ROOT = Path(__file__).resolve().parents[1]


def cli(*args):
    return subprocess.run(['strangeloop',*args],check=True,text=True,capture_output=True,timeout=300).stdout


def numerical_manifest():
    names = subprocess.check_output(['git','ls-tree','-r','--name-only',COMMIT],cwd=ROOT,text=True).splitlines()
    names = [n for n in names if ((n.startswith(('baseline/','rewriter/')) and n.endswith('.py'))
                                or n=='configs/full_reference.json')]
    return {name:hashlib.sha256(subprocess.check_output(['git','show',f'{COMMIT}:{name}'],cwd=ROOT)).hexdigest()
            for name in names}


def seed1_config():
    config=json.loads(subprocess.check_output(['git','show',f'{COMMIT}:configs/full_reference.json'],cwd=ROOT,text=True))
    assert config['seed']==0 and config['max_steps']==config['schedule_steps']==3250
    assert config['deterministic'] and config['wandb'] and config['fingerprint_steps']==[0,64]
    config['seed']=1
    return config


# Executed read-only before extending, and repeated immediately before staging.
QUALIFY = r'''
import json, pathlib, os
old = pathlib.Path(PREDECESSOR_OUTPUT)
metrics = old/'metrics.jsonl'
rows = [json.loads(line) for line in metrics.read_text().splitlines() if line] if metrics.exists() else []
finished = [r for r in rows if r.get('event')=='finished' and r.get('steps')==3250]
ready = bool(finished)
reason = 'predecessor has not completed all 3250 steps'
if ready:
    required=['final_rank0.pt','source_record.txt','config.json','environment.json','data_manifest.json']
    missing=[name for name in required if not (old/name).is_file()]
    if missing:
        raise RuntimeError('completed predecessor missing artifacts: '+str(missing))
    config=json.loads((old/'config.json').read_text())
    if config['seed']!=0 or config['max_steps']!=3250 or config['mode']!=PREDECESSOR_MODE:
        raise RuntimeError('predecessor identity mismatch')
    if not list((old/'wandb').glob('offline-run-*/*.wandb')):
        raise RuntimeError('completed predecessor has no offline W&B artifact')
    for proc in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
        try: args=proc.read_bytes().decode(errors='replace').split('\x00')
        except (FileNotFoundError,PermissionError,ProcessLookupError): continue
        if '--output' in args and args[args.index('--output')+1]==str(old):
            ready=False;reason='predecessor has finished metrics but trainer is still closing';break
    if ready: reason='predecessor finished; required artifacts preserved'
'''


def remote_args(predecessor):
    return ('PREDECESSOR_OUTPUT='+repr(predecessor['output'])+'\n'
            +'PREDECESSOR_MODE='+repr(predecessor['mode'])+'\n')


def attach_wandb(record):
    query="import pathlib,json; p=pathlib.Path("+repr(record['output'])+"); print(json.dumps([str(x.parent) for x in (p/'wandb').glob('offline-run-*/*.wandb')]))"
    paths=json.loads(cli('gpu','exec',record['gpu'],'--',PYTHON,'-c',query))
    if paths:
        cli('gpu','declare',record['gpu'],'--name',record['name'],'--wandb-path',paths[0])
        record['wandb_path']=paths[0]
    else:
        record['wandb_registration_pending']=True
    return record


def launch(label,predecessor,manifest):
    arm=ARMS[label];gpu=predecessor['gpu']
    status=json.loads(cli('gpu','status',gpu,'--json'))
    if status['status']!='ready' or status['gpu_type']!='H100' or status.get('gpu_count',1)!=1:
        return dict(label=label+'_seed1',state='waiting',reason='reuse lease unavailable; no new lease allocated',gpu=gpu)
    if status.get('image')!=IMAGE:
        raise RuntimeError('reuse image differs from frozen full panel')
    name='lwd-full-'+label+'-seed1'
    output=status['artifact_dir']+'/full-'+label+'-seed1'
    record=dict(label=label+'_seed1',arm=label,seed=1,gpu=gpu,name=name,output=output,
                config='full_'+label+'_seed1',mode=arm['mode'],source_commit=COMMIT,
                source_archive=ARCHIVE,source_archive_sha256=ARCHIVE_SHA,predecessor=predecessor,
                lease_hours_from_launch=arm['lease_hours'])
    # An earlier successful launch is not repeated; registration can be retried.
    record_file=ROOT/'runs/seed1_launch'/f'{label}.json'
    if record_file.exists():
        prior=json.loads(record_file.read_text())
        if prior.get('state')=='launched':
            return attach_wandb(prior)
    query=remote_args(predecessor)+QUALIFY+"\nprint(json.dumps(dict(ready=ready,reason=reason)))\n"
    qualification=json.loads(cli('gpu','exec',gpu,'--',PYTHON,'-c',query))
    if not qualification['ready']:
        return dict(record,state='waiting',reason=qualification['reason'])
    hard_end=dt.datetime.fromisoformat(status['pod_expires_at'].replace('Z','+00:00'))
    if (hard_end-dt.datetime.now(dt.timezone.utc)).total_seconds()<arm['lease_hours']*3600:
        return dict(record,state='waiting',reason='pod lifetime too short for declared lease window; no new lease allocated')
    cli('gpu','extend',gpu,'--hours',str(arm['lease_hours']))
    cli('gpu','declare',gpu,'--name',name,'--description',
        'Predeclared fresh seed1 '+label+'; exact frozen102ab99 PR3233250 schedule; '
        'reference selected before seed0 endpoints, not best-arm selection; no checkpoint reuse.')
    payload=dict(output=output,work='/workspace/lwd-seed1-'+label,archive=ARCHIVE,
                 archive_sha=ARCHIVE_SHA,manifest=manifest,config=seed1_config(),mode=arm['mode'],
                 label=label,python=PYTHON,artifact_dir=status['artifact_dir'])
    program=remote_args(predecessor)+QUALIFY+'\nP='+repr(payload)+r'''
if not ready: raise RuntimeError(reason)
import hashlib, shutil, subprocess, time
out=pathlib.Path(P['output']);work=pathlib.Path(P['work'])
if out.exists(): raise RuntimeError('refusing to overwrite existing seed1 output '+str(out))
archive=pathlib.Path(P['archive'])
if hashlib.sha256(archive.read_bytes()).hexdigest()!=P['archive_sha']:
    raise RuntimeError('deterministic archive hash changed')
inventory={str(p.relative_to(old)):p.stat().st_size for p in old.rglob('*') if p.is_file()}
manifest_path=pathlib.Path(P['artifact_dir'])/('seed1_'+P['label']+'_predecessor_inventory.json')
manifest_path.write_text(json.dumps(dict(predecessor=str(old),files=inventory,finished=finished[-1]),indent=2)+'\n')
subprocess.run(['sync',str(old)],check=True)
work.mkdir(parents=True,exist_ok=True)
subprocess.run(['tar','-xzf',str(archive),'-C',str(work)],check=True)
for name,expected in P['manifest'].items():
    if hashlib.sha256((work/name).read_bytes()).hexdigest()!=expected:
        raise RuntimeError('source differs from frozen102ab99: '+name)
# Derive from the archived reference, changing only the independent GPT seed.
c=json.loads((work/'configs/full_reference.json').read_text());c['seed']=1
if c!=P['config']: raise RuntimeError('seed1 configuration differs beyond seed')
config_path=work/'configs'/('full_'+P['label']+'_seed1.json')
config_path.write_text(json.dumps(c,indent=2)+'\n')
data=pathlib.Path('/tmp/fineweb10B');data.mkdir(parents=True,exist_ok=True)
for filename in [f'fineweb_train_{i:06d}.bin' for i in range(1,19)]+['fineweb_val_000000.bin']:
    destination=data/filename
    if not destination.exists(): shutil.copyfile('/scratch/datasets/fineweb10B/'+filename,destination)
# Reserve output atomically. Training itself creates no resumed state.
out.mkdir(parents=True,exist_ok=False)
env=dict(os.environ,PYTHONHASHSEED='0',CUBLAS_WORKSPACE_CONFIG=':4096:8')
cmd=['/opt/venv211/bin/torchrun','--standalone','--nproc_per_node=1','-m','baseline.train',
     '--config',str(config_path),'--mode',P['mode'],'--output',str(out)]
with pathlib.Path(str(out)+'.log').open('xb') as log:
    proc=subprocess.Popen(cmd,cwd=work,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(out/'launch.json').write_text(json.dumps(dict(pid=proc.pid,command=cmd,source_commit='102ab99',fresh_run=True),indent=2)+'\n')
for _ in range(30):
    if list((out/'wandb').glob('offline-run-*/*.wandb')): break
    if proc.poll() is not None: raise RuntimeError('fresh trainer exited during startup; inspect log, do not automatically replay')
    time.sleep(1)
print(json.dumps(dict(pid=proc.pid,inventory=str(manifest_path))),flush=True)
'''
    launched=json.loads(cli('gpu','exec',gpu,'--',PYTHON,'-c',program))
    record.update(launched,state='launched')
    record_file.parent.mkdir(parents=True,exist_ok=True)
    record_file.write_text(json.dumps(record,indent=2)+'\n')
    record=attach_wandb(record)
    record_file.write_text(json.dumps(record,indent=2)+'\n')
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=['reference','baseline','all'],default='all')
    parser.add_argument('--execute',action='store_true',help='extend free predecessor leases and start fresh runs')
    parser.add_argument('--predecessors',default='runs/panel_leases.json')
    args=parser.parse_args()
    if args.execute:
        raise SystemExit('SUPERSEDED: the 64-step-warmup six-run plan is disabled. Use only the reviewed baseline keeper; no learned full runs are authorized by this launcher.')
    labels=list(ARMS) if args.arm=='all' else [args.arm]
    previous={x['label']:x for x in json.loads((ROOT/args.predecessors).read_text())}
    plan=[dict(arm=label,predecessor=previous[ARMS[label]['predecessor']],mode=ARMS[label]['mode'],
               lease_hours=ARMS[label]['lease_hours'],source_commit=COMMIT,seed=1,max_steps=3250)
          for label in labels]
    if not args.execute:
        print(json.dumps(dict(dry_run=True,plan=plan,max_additional_h100_hours=sum(ARMS[x]['lease_hours'] for x in labels)),indent=2));return
    manifest=numerical_manifest()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(labels)) as pool:
        records=list(pool.map(lambda label:launch(label,previous[ARMS[label]['predecessor']],manifest),labels))
    path=ROOT/'runs/seed1_leases.json'
    saved={x['label']:x for x in json.loads(path.read_text())} if path.exists() else {}
    saved.update({x['label']:x for x in records})
    path.write_text(json.dumps(list(saved.values()),indent=2)+'\n')
    for record in records: print(json.dumps(record),flush=True)


if __name__=='__main__': main()
