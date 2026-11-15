"""Bounded caretaker for the explicitly authorized fast8 seed1→seed0 pair.

Run inside the existing eight-H100 lease. No allocation, parameter selection,
failed-launch retry, or lease release. Existing ambiguous archive/launch markers
fail closed. Final checkpoint shards are copied uncompressed and SHA256 checked.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

GPU='53480b53-afff-47f0-b2f8-8444064039ee'
BASE=Path('/persist/data/runs')/GPU
WORK=Path('/workspace/lwd-fast8')
SOURCE_SHA='97c0c32e904439d5e9764fcdc7edfa16d72932a68427d79f68d429e189f04c59'
STATE=BASE/'fast8_caretaker'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def log(event,**kwargs):
    with (STATE/'events.jsonl').open('a') as f:
        f.write(json.dumps(dict(time=time.time(),event=event,**kwargs))+'\n')


def observe(seed):
    out=BASE/f'baseline-8h100-seed{seed}'
    metrics=out/'metrics.jsonl'
    text=metrics.read_text() if metrics.exists() else ''
    rows=[json.loads(x) for x in text[:text.rfind('\n')+1].splitlines() if x]
    training=[r for r in rows if r.get('event')=='train']
    done=any(r.get('event')=='finished' and r['steps']==3250 for r in rows)
    active=[]
    for proc in Path('/proc').glob('[0-9]*/cmdline'):
        try:args=proc.read_bytes().decode(errors='replace').split('\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if '--output' in args and args[args.index('--output')+1]==str(out):active.append(proc.parent.name)
    return out,rows,training[-1] if training else None,done,active


def archive(seed,out,rows):
    target=Path('/scratch/checkpoints/learnable-weight-decay')/f'pr323_fast8_seed{seed}'/'final3250'
    target.mkdir(parents=True,exist_ok=True)
    marker=target/'archive_started.json'
    if marker.exists():raise RuntimeError('archive already attempted; inspect manually')
    marker.write_text(json.dumps(dict(started=time.time(),source=str(out)))+'\n')
    files={}
    for rank in range(8):
        src=out/f'final_rank{rank}.pt';dst=target/src.name
        if not src.is_file() or src.stat().st_size<100_000_000:raise RuntimeError('missing/truncated rank checkpoint')
        before=digest(src);shutil.copyfile(src,dst)
        if digest(dst)!=before:raise RuntimeError('checkpoint copy hash mismatch')
        files[dst.name]=dict(bytes=dst.stat().st_size,sha256=before)
    compact=target/'compact_record.tar.gz'
    inventory={}
    with tarfile.open(compact,'w:gz') as t:
        for p in sorted(out.rglob('*')):
            if p.is_file() and '.pt' not in p.suffixes and not p.is_symlink():
                name=str(p.relative_to(out));inventory[name]=dict(bytes=p.stat().st_size,sha256=digest(p));t.add(p,arcname=name)
    train=[r for r in rows if r.get('event')=='train']
    val=[r for r in rows if r.get('event')=='validation']
    manifest=dict(seed=seed,world_size=8,steps=3250,source_sha256=SOURCE_SHA,checkpoints=files,
                  compact_record=dict(path=str(compact),sha256=digest(compact),bytes=compact.stat().st_size),
                  compact_inventory=inventory,final_validation=val[-1],finished=rows[-1],
                  training_tokens=3250*524288,training_forwards_per_rank=train[-1]['training_forwards_per_rank'])
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    subprocess.run(['sync',str(target)],check=True)
    log('archived',seed=seed,directory=str(target),compact_sha256=manifest['compact_record']['sha256'])


def launch_seed0():
    out=BASE/'baseline-8h100-seed0'
    if out.exists():raise RuntimeError('seed0 output exists; never automatically replay launch')
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if active:raise RuntimeError('GPU still has active compute: '+active)
    source=BASE/'baseline-8h100-seed1'/'source_archive.tar.gz'
    if digest(source)!=SOURCE_SHA:raise RuntimeError('source archive mismatch')
    config=json.loads((WORK/'configs/baseline_8h100_seed1.json').read_text())
    config.update(seed=0,preserved_checkpoint_dir='/scratch/checkpoints/learnable-weight-decay/pr323_fast8_seed0')
    cfg=WORK/'configs/baseline_8h100_seed0.json';cfg.write_text(json.dumps(config,indent=2)+'\n')
    out.mkdir();shutil.copyfile(source,out/'source_archive.tar.gz');shutil.copyfile(__file__,out/'caretaker_launch.py')
    env=dict(os.environ,PYTHONHASHSEED='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',TORCHINDUCTOR_COMPILE_THREADS='2')
    cmd=['/opt/venv211/bin/torchrun','--standalone','--nproc_per_node=8','-m','baseline.launch',
         '--config','configs/baseline_8h100_seed0.json','--mode','baseline','--output',str(out)]
    with (out/'stdout.log').open('xb') as f:
        proc=subprocess.Popen(cmd,cwd=WORK,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(gpu=GPU,pid=proc.pid,command=cmd,source_sha256=SOURCE_SHA,config=config,started=time.time(),output=str(out))
    (out/'launch_manifest.json').write_text(json.dumps(record,indent=2)+'\n');log('launched',seed=0,**record)


def main():
    STATE.mkdir(exist_ok=True)
    fd=os.open(STATE/'started.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
    deadline=time.time()+3*3600
    (STATE/'deadline.json').write_text(json.dumps(dict(deadline=deadline))+'\n')
    seed=1
    try:
        while time.time()<deadline:
            out,rows,latest,done,active=observe(seed)
            log('poll',seed=seed,latest=latest,finished=done,active=active)
            if done and not active:
                archive(seed,out,rows)
                if seed==0:log('complete');return
                launch_seed0();seed=0
            elif not active:raise RuntimeError(f'seed{seed} stopped without finished3250 event')
            time.sleep(60)
        raise TimeoutError('three-hour caretaker deadline reached; no automatic mutation')
    except Exception as exc:
        log('error',error=repr(exc));raise

if __name__=='__main__':main()
