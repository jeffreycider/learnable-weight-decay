"""Launch only the three predeclared 256-step SOAP residual diagnostic arms.

Existing leases only. Each invocation launches one explicit arm, once; no retries
or allocations. Default is dry-run. All three use identical world8/data/schedule.
"""
import argparse
import json
import subprocess
from pathlib import Path

ARMS={'fixed':0.,'low':1e-5,'high':1e-4}


def cli(*args):
    return subprocess.run(['strangeloop',*args],check=True,text=True,capture_output=True,timeout=300).stdout


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',required=True);p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--archive',required=True);p.add_argument('--sha256',required=True);p.add_argument('--source-commit',required=True)
    p.add_argument('--execute',action='store_true');args=p.parse_args()
    status=json.loads(cli('gpu','status',args.gpu,'--json'))
    if status['status']!='ready' or status['gpu_count']!=8 or status['gpu_type']!='H100':raise ValueError('requires ready eight-H100 lease')
    name='soap-hyper-'+args.arm+'-256-v1';output=status['artifact_dir']+'/'+name
    plan={'gpu':args.gpu,'arm':args.arm,'output':output,'meta_lr':ARMS[args.arm],'steps':256,'seed':0,'microbatch':64,
          'source_archive':args.archive,'source_sha256':args.sha256,'source_commit':args.source_commit}
    print(json.dumps(plan,indent=2))
    if not args.execute:return
    cli('gpu','refresh',args.gpu)
    cli('gpu','declare',args.gpu,'--name',name,'--description','Matched seed0 256-iteration diagnostic: PR395 plus outer polar; fixed/learned dense residuals, no validation feedback')
    remote=r'''
import pathlib,hashlib,json,tarfile,shutil,subprocess,os,sys,time
plan=json.loads(sys.argv[1]); out=pathlib.Path(plan['output']); work=pathlib.Path('/tmp')/out.name
if out.exists() or work.exists():raise RuntimeError('refuse overwrite or ambiguous relaunch')
archive=pathlib.Path(plan['source_archive'])
assert hashlib.sha256(archive.read_bytes()).hexdigest()==plan['source_sha256']
work.mkdir();out.mkdir()
with tarfile.open(archive) as t:t.extractall(work)
shutil.copytree(work,out/'source_record')
data=work/'data/fineweb10B';data.mkdir(parents=True)
manifest={}
for name in ['fineweb_train_000001.bin','fineweb_train_000002.bin','fineweb_val_000000.bin']:
 src=pathlib.Path('/scratch/datasets/fineweb10B')/name;dst=data/name;shutil.copy2(src,dst)
 manifest[name]={'bytes':dst.stat().st_size,'sha256':hashlib.sha256(dst.read_bytes()).hexdigest()}
(out/'data_manifest.json').write_text(json.dumps(manifest,indent=2))
(out/'launch_plan.json').write_text(json.dumps(plan,indent=2))
(out/'hardware.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
env=dict(os.environ,PYTHONPATH=str(work),OMP_NUM_THREADS='4',TORCHINDUCTOR_COMPILE_THREADS='2',WANDB_MODE='offline')
cmd=['/opt/venv211/bin/torchrun','--standalone','--nproc_per_node=8','scripts/train_soap_hyper.py','--seed','0','--steps','256','--meta-lr',str(plan['meta_lr']),'--microbatch','64','--output',str(out)]
log=(out/'stdout.log').open('w');process=subprocess.Popen(cmd,cwd=work,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record={'pid':process.pid,'started_at_unix':time.time(),'command':cmd,'work':str(work),**plan}
(out/'launch.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
'''
    result=cli('gpu','exec',args.gpu,'--','/opt/venv211/bin/python','-c',remote,json.dumps(plan))
    record_dir=Path('runs/soap-hyper-pilot-v1');record_dir.mkdir(parents=True,exist_ok=True)
    (record_dir/(args.arm+'-launch.json')).write_text(result)
    print(result)

if __name__=='__main__':main()
