"""Fresh, exact corrected-FP32 teacher backfill on a completed child state bank."""
import argparse,json,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.launch_multiseed_capture import cli,ROOT
RECIPE=ROOT/'runs/multiseed-capture/teacher_recipe.json'
REMOTE=r'''
import hashlib,json,os,pathlib,shutil,subprocess,sys,tarfile,time
p=json.loads(sys.argv[1]);out=pathlib.Path(p['output']);work=pathlib.Path('/tmp')/p['name'];cache=pathlib.Path(p['cache']);bank=pathlib.Path(p['bank']);recipe=p['recipe']
if out.exists() or work.exists() or cache.exists():raise RuntimeError('refuse overwrite or ambiguous teacher launch')
if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU not empty')
assert json.loads((bank/'complete.json').read_text())['steps']==1000
assert json.loads((bank/'archive_manifest.json').read_text())['seed']==p['seed']
archive=pathlib.Path(recipe['source_archive']);assert hashlib.sha256(archive.read_bytes()).hexdigest()==recipe['source_archive_sha256']
work.mkdir();out.mkdir(parents=True)
with tarfile.open(archive) as tar:tar.extractall(work,filter='data')
for name,sha in recipe['source_files_sha256'].items():assert hashlib.sha256((work/name).read_bytes()).hexdigest()==sha
shutil.copy2(archive,out/'source_archive.tar.gz')
replacements={'{completed_seed_bank}':str(bank),'{same_pinned_train_data_glob}':'/tmp/chisel-multiseed-data/fineweb_train_*.bin','{new_seed_teacher_cache}':str(cache),'{distinct_seed_cache_wandb_dir}':str(out/'wandb')}
cmd=[replacements.get(x,x) for x in recipe['argv_template']]
env=dict(os.environ,**recipe['environment'],WANDB_MODE='offline')
with (out/'stdout.log').open('xb') as log:
 proc=subprocess.Popen(cmd,cwd=work,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record={**p,'pid':proc.pid,'work':str(work),'started_at':time.time(),'command':cmd}
(out/'launch_manifest.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
 assert a.seed in range(1,10)
 assert hashlib.sha256(RECIPE.read_bytes()).hexdigest()=='28e06feaf6aa8a694ad4a36cd2122fcdd9f116bccfae539280af7263f82a8e7e'
 recipe=json.loads(RECIPE.read_text());status=json.loads(cli('gpu','status',a.gpu,'--json'));assert status['status']=='ready' and status['gpu_count']==8
 name=f'chisel-teacher-cache1000-fp32-eps23-seed{a.seed}-v1'
 plan=dict(name=name,gpu=a.gpu,seed=a.seed,recipe=recipe,output=status['artifact_dir']+'/'+name,bank=f'/scratch/checkpoints/learnable-weight-decay/chisel-state-bank-seed{a.seed}-v1',cache=f'/scratch/checkpoints/learnable-weight-decay/{name}')
 directory=ROOT/'runs/multiseed-capture';target=directory/f'seed{a.seed}-teacher-launch.json'
 if target.exists() or (directory/f'seed{a.seed}-teacher-intent.json').exists():raise FileExistsError('existing teacher launch or ambiguous intent')
 if not a.execute:print(json.dumps(plan,indent=2));return
 # All source files predate capture mount. Do not refresh or restage the 628GB bank.
 cli('gpu','declare',a.gpu,'--name',name,'--description',f'Fresh FP32 RMS epsilon2^-23 teacher targets for seed{a.seed};1000states;128train+64audit;cached finalhead inputs; exact five-state target/cotangent qualification; no GPT backward')
 (directory/f'seed{a.seed}-teacher-intent.json').write_text(json.dumps(plan,indent=2))
 result=cli('gpu','exec',a.gpu,'--','/opt/venv211/bin/python','-c',REMOTE,json.dumps(plan));target.write_text(result);print(result)
if __name__=='__main__':main()
