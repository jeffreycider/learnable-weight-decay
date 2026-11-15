"""Archive one completed 2900-step SOAP run without altering or releasing its GPU."""
import argparse
import json
import subprocess
from pathlib import Path

REMOTE = r'''
import pathlib,hashlib,json,shutil,os,sys,time,subprocess,concurrent.futures
src=pathlib.Path(sys.argv[1]);dst=pathlib.Path(sys.argv[2]);start=time.perf_counter()
assert (src/'finished.json').exists(), 'training not finished'
rows=[json.loads(x) for x in (src/'metrics.jsonl').read_text().splitlines()]
train=[x for x in rows if x.get('event')=='train']
plan=json.loads((src/'launch_plan.json').read_text());inherited=plan.get('freeze_at') or 0
assert len(train)==2900-inherited and train[-1]['step']==2900
assert len(list(src.glob('final_rank*.pt')))==8
if dst.exists():raise RuntimeError('archive exists; inspect rather than overwrite')
dst.mkdir(parents=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8388608),b''):h.update(chunk)
 return h.hexdigest()
def copy(p):
 rel=p.relative_to(src);q=dst/rel;q.parent.mkdir(parents=True,exist_ok=True)
 expected=sha(p);shutil.copy2(p,q)
 assert sha(q)==expected, str(rel)
 return str(rel),{'bytes':q.stat().st_size,'sha256':expected}
files=[p for p in src.rglob('*') if p.is_file() and not p.is_symlink()]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:records=dict(pool.map(copy,files))
checkroot=pathlib.Path('/scratch/checkpoints/learnable-weight-decay')/src.name
checkpoints={}
for folder in sorted(checkroot.glob('step*')):
 assert (folder/'complete.json').exists(), 'incomplete checkpoint '+str(folder)
 shards=list(folder.glob('rank*.pt'))
 if not shards:shards=list(folder.glob('*.pt'))
 assert len(shards)==8,(folder,len(shards))
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  hashes=list(pool.map(sha,shards))
 record={p.name:{'bytes':p.stat().st_size,'sha256':h} for p,h in zip(shards,hashes)}
 (folder/'manifest.json').write_text(json.dumps({'complete':True,'files':record},indent=2))
 checkpoints[str(folder)]={'files':record,'manifest_sha256':sha(folder/'manifest.json')}
manifest={'source':str(src),'files':records,'checkpoints':checkpoints,'complete':True,
 'ordinary_gpt_iterations':2900,'inherited_gpt_iterations':inherited,'new_gpt_iterations':2900-inherited,'training_forward_backward_calls_per_rank':2900,
 'training_forward_backward_calls_global':23200,'new_training_forward_backward_calls_global':8*(2900-inherited),
 'inherited_training_forward_backward_calls_global':8*inherited,'validation_events':sum(x.get('event')=='validation' for x in rows),
 'validation_forward_calls_global':sum(160*(1+(x['step']>=2040)) for x in rows if x.get('event')=='validation'),
 'extra_gpt_training_evaluations':0,'archive_seconds':time.perf_counter()-start,
 'launch_to_finished_seconds':(src/'finished.json').stat().st_mtime-json.loads((src/'launch.json').read_text())['started_at_unix']}
(dst/'manifest.json').write_text(json.dumps(manifest,indent=2))
subprocess.run(['sync','/scratch'],check=True)
print(json.dumps({'target':str(dst),'manifest_sha256':sha(dst/'manifest.json'),'file_count':len(records),
 'bytes':sum(x['bytes'] for x in records.values()),'archive_seconds':manifest['archive_seconds']}))
'''

def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['fixed','learned','low','frozen'],required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
 local=Path(f'runs/soap-full-seed{a.seed}-v1');info=json.loads((local/(a.arm+'-launch.json')).read_text())
 target=f'/scratch/artifacts/learnable-weight-decay/soap-full-seed{a.seed}-v1/{a.arm}'
 result=subprocess.run(['strangeloop','gpu','exec',info['gpu'],'--','/opt/venv211/bin/python','-c',REMOTE,info['output'],target],check=True,text=True,capture_output=True,timeout=900)
 (local/(a.arm+'-archive.json')).write_text(result.stdout);print(result.stdout)
 (local/a.arm).mkdir(exist_ok=True)
 for name in ['manifest.json','config.json','metrics.jsonl','launch_plan.json','data_manifest.json','finished.json','hardware.txt','upstream.log']:
  temporary=local/a.arm/(name+'.download')
  subprocess.run(['strangeloop','scratch','get',target.removeprefix('/scratch/')+'/'+name,str(temporary)],check=True,stdout=subprocess.DEVNULL,timeout=60)
  if name=='metrics.jsonl':
   text=temporary.read_text()
   if not text.endswith('\n'):raise ValueError('completed metrics missing final newline')
   for row in text.splitlines():json.loads(row)
  temporary.replace(local/a.arm/name)

if __name__=='__main__':main()
