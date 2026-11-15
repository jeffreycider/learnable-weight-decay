"""Launch only the predeclared full2900-step SOAP residual pair.

Existing leases only. Each invocation launches one explicit arm, once; no retries
or allocations. Default is dry-run. All three use identical world8/data/schedule.
"""
import argparse
import json
import subprocess
from pathlib import Path

ARMS={'fixed':0.,'learned':1e-4,'low':1e-5,'frozen':1e-4}


def cli(*args):
    return subprocess.run(['strangeloop',*args],check=True,text=True,capture_output=True,timeout=300).stdout


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',required=True);p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--archive',required=True);p.add_argument('--sha256',required=True);p.add_argument('--source-commit',required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--resume');p.add_argument('--freeze-at',type=int);p.add_argument('--execute',action='store_true');args=p.parse_args()
    if args.arm=='frozen' and (not args.resume or args.freeze_at!=1000):raise ValueError('frozen diagnostic requires explicit step1000 resume')
    if args.arm!='frozen' and (args.resume or args.freeze_at is not None):raise ValueError('fresh comparisons cannot resume a different trial')
    status=json.loads(cli('gpu','status',args.gpu,'--json'))
    if status['status']!='ready' or status['gpu_count']!=8 or status['gpu_type']!='H100':raise ValueError('requires ready eight-H100 lease')
    name='soap-full-'+args.arm+'-seed'+str(args.seed)+'-v1';output=status['artifact_dir']+'/'+name
    plan={'gpu':args.gpu,'arm':args.arm,'output':output,'meta_lr':ARMS[args.arm],'steps':2900,'seed':args.seed,'microbatch':64,
          'resume':args.resume,'freeze_at':args.freeze_at,'source_archive':args.archive,'source_sha256':args.sha256,'source_commit':args.source_commit}
    print(json.dumps(plan,indent=2))
    if not args.execute:return
    cli('gpu','refresh',args.gpu)
    cli('gpu','declare',args.gpu,'--name',name,'--description','Matched full2900-iteration PR395 plus outer polar; fixed/learned dense residuals, original validation readout, no validation feedback')
    remote=r'''
import pathlib,hashlib,json,tarfile,shutil,subprocess,os,sys,time
plan=json.loads(sys.argv[1]); out=pathlib.Path(plan['output']); work=pathlib.Path('/tmp')/out.name
if out.exists() or work.exists():raise RuntimeError('refuse overwrite or ambiguous relaunch')
archive=pathlib.Path(plan['source_archive'])
assert hashlib.sha256(archive.read_bytes()).hexdigest()==plan['source_sha256']
work.mkdir();out.mkdir()
with tarfile.open(archive) as t:t.extractall(work)
shutil.copytree(work,out/'source_record')
if plan['resume']:
 checkpoint=pathlib.Path(plan['resume']);header=json.loads((checkpoint/'complete.json').read_text())
 assert header['completed_steps']==plan['freeze_at'] and header['seed']==plan['seed'] and header['meta_lr']==plan['meta_lr']
 assert len(list(checkpoint.glob('rank*.pt')))==8
 manifest=json.loads((checkpoint/'manifest.json').read_text())
 for name,row in manifest['files'].items():
  h=hashlib.sha256()
  with (checkpoint/name).open('rb') as f:
   for chunk in iter(lambda:f.read(8388608),b''):h.update(chunk)
  assert h.hexdigest()==row['sha256'],name
 (out/'ancestor_checkpoint_manifest.json').write_text(json.dumps(manifest,indent=2))
data=work/'data/fineweb10B';data.mkdir(parents=True)
manifest={}
for name in [f'fineweb_train_{i:06d}.bin' for i in range(1,17)]+['fineweb_val_000000.bin']:
 src=pathlib.Path('/scratch/datasets/fineweb10B')/name;dst=data/name;shutil.copy2(src,dst)
 manifest[name]={'bytes':dst.stat().st_size,'sha256':hashlib.sha256(dst.read_bytes()).hexdigest()}
(out/'data_manifest.json').write_text(json.dumps(manifest,indent=2))
(out/'launch_plan.json').write_text(json.dumps(plan,indent=2))
(out/'hardware.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
env=dict(os.environ,PYTHONPATH=str(work),OMP_NUM_THREADS='4',TORCHINDUCTOR_COMPILE_THREADS='2',WANDB_MODE='offline',SOAP_CHECKPOINT_ROOT='/scratch/checkpoints/learnable-weight-decay/'+out.name)
cmd=['/opt/venv211/bin/torchrun','--standalone','--nproc_per_node=8','scripts/train_soap_hyper.py','--seed',str(plan['seed']),'--steps','2900','--checkpoint-every','500','--meta-lr',str(plan['meta_lr']),'--microbatch','64','--output',str(out)]
if plan['resume']:
 cmd[3]='scripts/train_soap_frozen.py';cmd.extend(['--freeze-at',str(plan['freeze_at']),'--resume',plan['resume']])
log=(out/'stdout.log').open('w');process=subprocess.Popen(cmd,cwd=work,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record={'pid':process.pid,'started_at_unix':time.time(),'command':cmd,'work':str(work),**plan}
(out/'launch.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
'''
    result=cli('gpu','exec',args.gpu,'--','/opt/venv211/bin/python','-c',remote,json.dumps(plan))
    record_dir=Path('runs/soap-full-seed'+str(args.seed)+'-v1');record_dir.mkdir(parents=True,exist_ok=True)
    (record_dir/(args.arm+'-launch.json')).write_text(result)
    print(result)

if __name__=='__main__':main()
