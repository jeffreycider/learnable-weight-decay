"""Launch one explicit record46 trial on an existing idle eight-H100 lease."""
import argparse
import json
import math
from pathlib import Path
import subprocess


def cli(*args):
    return subprocess.run(['strangeloop', *args], check=True, text=True,
                          capture_output=True, timeout=300).stdout


REMOTE = r'''
import hashlib,json,os,pathlib,shutil,subprocess,sys,tarfile,time
p=json.loads(sys.argv[1]);out=pathlib.Path(p['output']);work=pathlib.Path('/tmp')/p['name']
if out.exists() or work.exists():raise RuntimeError('refuse overwrite or ambiguous relaunch')
# This launcher never shares GPU capacity with another training process.
active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
if active:raise RuntimeError('GPU training processes already active: '+active)
archive=pathlib.Path(p['archive'])
if p.get('reuse_mounted_volume'):
 prior=[json.loads(f.read_text()) for f in out.parent.glob('*/launch_plan.json')]
 assert any(x.get('archive')==p['archive'] and x.get('sha256')==p['sha256'] for x in prior),'reuse requires an archive verified by a previous launch on this node'
assert hashlib.sha256(archive.read_bytes()).hexdigest()==p['sha256'],'archive hash mismatch'
work.mkdir();out.mkdir(parents=True)
with tarfile.open(archive) as t:t.extractall(work,filter='data')
data=pathlib.Path('/tmp/soap46-data/data/fineweb10B');data.mkdir(parents=True,exist_ok=True)
manifest={}
for name in [f'fineweb_train_{i:06d}.bin' for i in range(1,17)]+['fineweb_val_000000.bin']:
 src=pathlib.Path('/scratch/datasets/fineweb10B')/name;dst=data/name
 if not dst.exists():shutil.copy2(src,dst)
 assert src.stat().st_size==dst.stat().st_size
 manifest[name]={'bytes':dst.stat().st_size,'sha256':hashlib.sha256(dst.read_bytes()).hexdigest()}
(work/'data').symlink_to(data.parent,target_is_directory=True)
(out/'data_manifest.json').write_text(json.dumps(manifest,indent=2))
(out/'launch_plan.json').write_text(json.dumps(p,indent=2))
env=dict(os.environ,PYTHONPATH=str(work),OMP_NUM_THREADS='4',TORCHINDUCTOR_COMPILE_THREADS='2',WANDB_MODE='offline',SOAP46_CHECKPOINT_ROOT='/scratch/checkpoints/learnable-weight-decay/'+p['name'])
cmd=['/opt/venv211/bin/torchrun','--standalone','--nproc_per_node=8','scripts/train_soap46.py',
 '--variant',p['variant'],'--seed',str(p['seed']),'--steps',str(p['steps']),
 '--meta-lr',str(p['meta_lr']),'--warmup-steps',str(p['warmup_steps']),'--warmup-meta',p['warmup_meta'],'--meta-ramp-steps',str(p['meta_ramp_steps']),'--residual-decay',str(p['residual_decay']),'--base-lr',str(p['base_lr']),'--cwd',str(p['cwd']),
 '--microbatch','64','--checkpoint-every',str(p['checkpoint_every']),'--output',str(out)]
cmd+=['--meta-clip',str(p['meta_clip']),'--meta-betas',*map(str,p['meta_betas'])]
if p.get('resume'):cmd+=['--resume',p['resume']]
if p.get('compile_map'):cmd+=['--compile-map']
if p.get('reset_meta_at_activation',False):cmd+=['--reset-meta-at-activation']
# Old immutable archives accept defaults implicitly; new options require a new explicit source.
if p.get('residual_frame','basis')!='basis':cmd+=['--residual-frame',p['residual_frame']]
if p.get('meta_epsilon',1e-8)!=1e-8:cmd+=['--meta-epsilon',str(p['meta_epsilon'])]
log=(out/'stdout.log').open('w')
proc=subprocess.Popen(cmd,cwd=work,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record={'pid':proc.pid,'started_at_unix':time.time(),'command':cmd,'work':str(work),**p}
(out/'launch.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
'''


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gpu',required=True);p.add_argument('--name',required=True)
    p.add_argument('--variant',choices=['inner-outer','outer-only'],required=True)
    p.add_argument('--meta-lr',type=float,required=True)
    p.add_argument('--meta-clip',type=float,default=1.)
    p.add_argument('--residual-frame',choices=['basis','parameter'],default='basis');p.add_argument('--meta-epsilon',type=float,default=1e-8)
    p.add_argument('--meta-betas',type=float,nargs=2,default=[.9,.99])
    p.add_argument('--warmup-steps',type=int,default=0);p.add_argument('--warmup-meta',choices=['moments','off'],default='moments')
    p.add_argument('--meta-ramp-steps',type=int,default=0);p.add_argument('--residual-decay',type=float,default=0.)
    p.add_argument('--base-lr',type=float,default=.0375);p.add_argument('--cwd',type=float,default=.025)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--steps',type=int,default=2900)
    p.add_argument('--checkpoint-every',type=int,default=500);p.add_argument('--resume')
    p.add_argument('--reset-meta-at-activation',action='store_true')
    p.add_argument('--compile-map',action='store_true')
    p.add_argument('--archive',required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--source-commit',required=True);p.add_argument('--execute',action='store_true')
    p.add_argument('--reuse-mounted-volume',action='store_true',help='Reuse an archive already SHA-verified by a prior launch on this node; never refresh during archival')
    a=p.parse_args()
    if not a.meta_clip>0 or not all(0<=b<1 for b in a.meta_betas):raise ValueError('invalid residual Adam settings')
    if not math.isfinite(a.meta_epsilon) or a.meta_epsilon<=0 or (a.residual_frame=='parameter' and a.variant!='outer-only'):raise ValueError('invalid residual conditioning')
    if not 1<=a.steps<=2900 or not a.name.replace('-','').replace('_','').isalnum():raise ValueError('invalid run name or length')
    if a.reset_meta_at_activation and not 2<=a.warmup_steps<a.steps:raise ValueError('invalid residual Adam reset boundary')
    status=json.loads(cli('gpu','status',a.gpu,'--json'))
    if status['status']!='ready' or status['gpu_count']!=8 or status['gpu_type']!='H100':raise ValueError('requires ready eight-H100 lease')
    plan={k:v for k,v in vars(a).items() if k!='execute'}
    plan['output']=status['artifact_dir']+'/'+a.name
    print(json.dumps(plan,indent=2))
    if not a.execute:return
    if not a.reuse_mounted_volume:cli('gpu','refresh',a.gpu)
    cli('gpu','declare',a.gpu,'--name',a.name,'--description',
        f'Accepted record46; {a.variant}; residualAdamLR{a.meta_lr}; seed{a.seed}; {a.steps}steps, schedule2900; no extra GPT training evaluations')
    result=cli('gpu','exec',a.gpu,'--','/opt/venv211/bin/python','-c',REMOTE,json.dumps(plan))
    records=Path('runs/soap46-v1');records.mkdir(parents=True,exist_ok=True)
    (records/(a.name+'-launch.json')).write_text(result)
    print(result)


if __name__=='__main__':main()
