"""Launch exactly four declared cached-target diagnostics on one owned eight-GPU node.

No allocation, retry, sweep expansion, or outcome-dependent configuration changes.
The existing immutable state bank and complete target cache are staged locally once.
"""
import argparse,hashlib,json,subprocess
from pathlib import Path


def arm_specs():
    return [dict(name=f'chisel-cache25-lr{label}-batch{batch}',gpus=[2*i,2*i+1],lr=lr,batch=batch)
            for i,(label,lr,batch) in enumerate([('2e-5',2e-5,8),('2e-5',2e-5,20),('2e-4',2e-4,8),('2e-4',2e-4,20)])]


def training_command(work,bank,cache,data,output,arm):
    args=['/opt/venv211/bin/torchrun','--standalone','--nproc_per_node=2',str(Path(work)/'scripts/train_multistate_chisel.py')]
    values=dict(state_bank=bank,teacher_cache=cache,train_data=data,output=output,diagnostic_prefix_states=25,
        epochs=30,checkpoint_batch=arm['batch'],sequences_per_checkpoint=32,gpt_microbatch=16,replay_pool_sequences=4096,
        lr=arm['lr'],kl_budget=1e-5,radius_fraction=.01,gradient_clip=1,controller_seed=200001,data_seed=400001,
        split_seed=500001,expected_gpt_seed=0,audit_every_epochs=5,audit_states=8,audit_sequences=32,
        checkpoint_every=10,actor_backend='packed',norm_backend='fused')
    for key,value in values.items():args.extend(['--'+key.replace('_','-'),str(value)])
    args.extend(['--full-training-audit','--gradient-diagnostics','--verify-state-once','--mmap-state','--wandb'])
    return args


WORKER=r'''
import pathlib,json,subprocess,os,hashlib,shutil,tarfile,time
spec=SPEC;root=pathlib.Path(spec['record']);root.mkdir(parents=True,exist_ok=False)
def event(name,**kw):
 with (root/'orchestration.jsonl').open('a') as f:f.write(json.dumps(dict(event=name,time=time.time(),**kw))+'\n')
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
 return h.hexdigest()
source=pathlib.Path(spec['source']);assert sha(source)==spec['source_sha256']
cache_source=pathlib.Path(spec['cache_source']);manifest=json.loads((cache_source/'manifest.json').read_text());assert manifest['complete']
assert len(manifest['panels'])==50 and all(p['qualification_max_logprob_error']==0 for p in manifest['panels'])
if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU work is still active; do not overlap generation')
work=pathlib.Path(spec['work']);work.mkdir(parents=True,exist_ok=False)
with tarfile.open(source) as f:f.extractall(work,filter='data')
shutil.copyfile(source,root/'source.tar.gz');cache=pathlib.Path(spec['cache']);shutil.copytree(cache_source,cache)
for row in manifest['panels']:assert sha(cache/row['file'])==row['sha256']
event('cache_staged',bytes=manifest['total_bytes'],cache_manifest_sha256=sha(cache/'manifest.json'))
processes=[]
for arm in spec['arms']:
 record=root/arm['name'];record.mkdir();(record/'config.json').write_text(json.dumps(arm,indent=2))
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,arm['gpus'])),PYTHONHASHSEED='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',TORCHINDUCTOR_COMPILE_THREADS='4',TORCH_LOGS='recompiles,graph_breaks',TORCHINDUCTOR_CACHE_DIR='/tmp/chisel-multistate-compiler-cache',PYTHONPATH=str(work))
 log=(record/'training.log').open('wb');p=subprocess.Popen(arm['command'],cwd=work,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
 processes.append((arm,p,log));event('training_started',arm=arm['name'],pid=p.pid,command=arm['command'])
while processes:
 for arm,p,log in list(processes):
  code=p.poll()
  if code is not None:
   log.close();event('training_finished' if code==0 else 'training_failed',arm=arm['name'],returncode=code);processes.remove((arm,p,log))
 if processes:time.sleep(10)
event('all_processes_closed')
'''


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',required=True);p.add_argument('--source',required=True);p.add_argument('--source-sha256',required=True);p.add_argument('--cache-source',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    name='chisel-cache25-panel-v1';record=f'/persist/data/runs/{a.gpu}/{name}';work='/workspace/'+name;cache='/tmp/'+name+'-targets';bank='/tmp/chisel-multistate-prefix25-v1-bank';data='/tmp/chisel-multistate-prefix25-v1-data/fineweb_train_*.bin'
    arms=arm_specs()
    for arm in arms:arm['command']=training_command(work,bank,cache,data,record+'/'+arm['name']+'/fit',arm)
    spec=dict(gpu=a.gpu,source=a.source,source_sha256=a.source_sha256,cache_source=a.cache_source,record=record,work=work,cache=cache,arms=arms)
    local=Path('runs')/name;local.mkdir(parents=True,exist_ok=True);(local/'launch_plan.json').write_text(json.dumps(spec,indent=2))
    if not a.execute:print(json.dumps(spec,indent=2));return
    marker=local/'launch_attempt.json'
    with marker.open('x') as f:json.dump(spec,f)
    worker=local/'worker.py';worker.write_text('SPEC='+repr(spec)+'\n'+WORKER)
    subprocess.run(['strangeloop','gpu','push',a.gpu,str(worker),'/tmp/'+name+'.py'],check=True)
    launch='import subprocess,os;f=open('+repr(record+'.orchestrator.log')+',"xb");p=subprocess.Popen(["/opt/venv211/bin/python",'+repr('/tmp/'+name+'.py')+'],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);print(p.pid)'
    subprocess.run(['strangeloop','gpu','exec',a.gpu,'--','/opt/venv211/bin/python','-c',launch],check=True)
    for arm in arms:subprocess.run(['strangeloop','gpu','declare',a.gpu,'--name',arm['name'],'--description','Fixed25-state cached-teacher diagnostic; fresh Chisel,30epochs, no checkpoint selection'],check=True)
if __name__=='__main__':main()
