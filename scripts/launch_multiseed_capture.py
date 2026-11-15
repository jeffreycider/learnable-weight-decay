"""Launch one authorized PR323 state-bank capture from the unchanged seed0 source."""
import argparse,hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ARCHIVE='/scratch/staging/chisel-state-bank-dbd27d3.tar.gz'
SHA='7189aad9b54df27e515044797ecf2cb7944b432b947c657eb6011e859f7ed538'

def configuration(seed):
    if seed not in range(1,10):raise ValueError('authorized new seeds are1..9')
    cfg=json.loads((ROOT/'configs/state_bank1000_seed0.json').read_text())
    cfg['seed']=seed
    cfg['train_data']='/tmp/chisel-multiseed-data/fineweb_train_*.bin'
    cfg['val_data']='/tmp/chisel-multiseed-data/fineweb_val_*.bin'
    cfg['state_bank']['directory']=f'/scratch/checkpoints/learnable-weight-decay/chisel-state-bank-seed{seed}-v1'
    return cfg

def cli(*args):
    return subprocess.run(['strangeloop',*args],check=True,text=True,capture_output=True,timeout=300).stdout

REMOTE=r'''
import hashlib,json,os,pathlib,shutil,subprocess,sys,tarfile,time
p=json.loads(sys.argv[1]);out=pathlib.Path(p['output']);work=pathlib.Path('/tmp')/p['name'];bank=pathlib.Path(p['config']['state_bank']['directory'])
if out.exists() or work.exists() or bank.exists():raise RuntimeError('refuse overwrite/ambiguous relaunch')
if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU not empty')
if p.get('reuse_mounted_volume'):
 prior=[json.loads(f.read_text()) for f in out.parent.glob('chisel-state-bank1000-seed*/launch_manifest.json')]
 assert any(x.get('source_sha256')==p['source_sha256'] and x.get('archive')==p['archive'] for x in prior),'reuse requires previously verified same capture archive on this node'
source=pathlib.Path(p['archive']);assert hashlib.sha256(source.read_bytes()).hexdigest()==p['source_sha256']
work.mkdir();out.mkdir(parents=True)
with tarfile.open(source) as t:t.extractall(work,filter='data')
data=pathlib.Path('/tmp/chisel-multiseed-data');data.mkdir(exist_ok=True)
# Exactly the six original prefix shards; all data are already present on shared storage.
for name in [f'fineweb_train_{i:06d}.bin' for i in range(1,7)]+['fineweb_val_000000.bin']:
    dst=data/name
    src=pathlib.Path('/scratch/datasets/fineweb10B')/name
    cached=pathlib.Path('/tmp/soap46-data/data/fineweb10B')/name
    if not dst.exists():shutil.copy2(cached if cached.exists() else src,dst)
    assert dst.stat().st_size==src.stat().st_size
# baseline.data_manifest independently verifies all pinned data hashes before training.
config=work/'capture_config.json';config.write_text(json.dumps(p['config'],indent=2)+'\n')
shutil.copy2(source,out/'source_archive.tar.gz')
(out/'launch_plan.json').write_text(json.dumps(p,indent=2)+'\n')
env=dict(os.environ,PYTHONHASHSEED='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',TORCHINDUCTOR_COMPILE_THREADS='2',WANDB_MODE='offline')
cmd=['/opt/venv211/bin/torchrun','--standalone','--nproc_per_node=8','-m','baseline.launch','--config',str(config),'--mode','baseline','--output',str(out)]
with (out/'stdout.log').open('xb') as log:
    proc=subprocess.Popen(cmd,cwd=work,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record={**p,'pid':proc.pid,'work':str(work),'started_at':time.time(),'command':cmd}
(out/'launch_manifest.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',required=True);p.add_argument('--seed',required=True,type=int);p.add_argument('--execute',action='store_true');p.add_argument('--reuse-mounted-volume',action='store_true');a=p.parse_args()
    status=json.loads(cli('gpu','status',a.gpu,'--json'))
    assert status['status']=='ready' and status['gpu_count']==8 and status['gpu_type']=='H100'
    name=f'chisel-state-bank1000-seed{a.seed}-v1'
    plan=dict(name=name,gpu=a.gpu,seed=a.seed,archive=ARCHIVE,source_commit='dbd27d3',source_sha256=SHA,
              config=configuration(a.seed),output=status['artifact_dir']+'/'+name,reuse_mounted_volume=a.reuse_mounted_volume)
    directory=ROOT/'runs/multiseed-capture';directory.mkdir(exist_ok=True)
    target=directory/f'seed{a.seed}-launch.json'
    if target.exists():raise FileExistsError('inspect existing launch; never blindly repeat')
    if not a.execute:print(json.dumps(plan,indent=2));return
    if not a.reuse_mounted_volume:cli('gpu','refresh',a.gpu)
    cli('gpu','declare',a.gpu,'--name',name,'--description',f'PR323 seed{a.seed};1000 ordinary GPT iterations; every-step full weights+W/G/M/U/activity; original pure-kernel capture; no controller')
    (directory/f'seed{a.seed}-intent.json').write_text(json.dumps(plan,indent=2))
    result=cli('gpu','exec',a.gpu,'--','/opt/venv211/bin/python','-c',REMOTE,json.dumps(plan))
    target.write_text(result);print(result)
if __name__=='__main__':main()
