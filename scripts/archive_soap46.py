"""Archive one closed record46 trial; never deletes its source or releases a lease."""
import argparse,json,pathlib,subprocess,tempfile,os

def cli(*args):
 result=subprocess.run(['strangeloop',*args],text=True,capture_output=True,timeout=360)
 if result.returncode:raise RuntimeError(result.stderr or result.stdout or f'CLI exit {result.returncode}')
 return result.stdout

REMOTE=r'''
import hashlib,json,pathlib,shutil,subprocess,sys
p=pathlib.Path(sys.argv[1]);d=pathlib.Path(sys.argv[2])
checkpoint_only=bool(int(sys.argv[3]))
complete=json.loads((p/'complete.json').read_text())
if checkpoint_only:
 finished=complete
 assert complete['completed_steps']==int(sys.argv[3])
 assert len(list(p.glob('rank*.pt')))==complete['world_size']
else:
 finished=json.loads((p/'finished.json').read_text())
 assert finished['completed_steps']==complete['completed_steps']
 assert len(list(p.glob('final_rank*.pt')))==finished['world_size']
 rows=[json.loads(x) for x in (p/'metrics.jsonl').read_text().splitlines()]
 assert rows[-1]['event']=='validation' and rows[-1]['step']==finished['completed_steps']
def sha(f):
 h=hashlib.sha256()
 with f.open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
d.mkdir(parents=True,exist_ok=True);files={}
for f in sorted(p.rglob('*')):
 if not f.is_file():continue
 relative=str(f.relative_to(p));target=d/relative;target.parent.mkdir(parents=True,exist_ok=True)
 digest=sha(f)
 if not target.exists():shutil.copy2(f,target)
 assert target.stat().st_size==f.stat().st_size and sha(target)==digest,relative
 files[relative]={'bytes':f.stat().st_size,'sha256':digest}
manifest={'schema':'record46-archive-v1','source':str(p),'completed_steps':finished['completed_steps'],'files':files}
tmp=d/'archive_manifest.json.incomplete';tmp.write_text(json.dumps(manifest,indent=2));tmp.replace(d/'archive_manifest.json')
subprocess.run(['sync','/scratch'],check=True)
print(json.dumps({'archive':str(d),'files':len(files),'bytes':sum(v['bytes'] for v in files.values()),'manifest_sha256':sha(d/'archive_manifest.json')}))
'''

def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',required=True);p.add_argument('--name',required=True);p.add_argument('--checkpoint-step',type=int,default=0);p.add_argument('--execute',action='store_true');a=p.parse_args()
 status=json.loads(cli('gpu','status',a.gpu,'--json'));source=status['artifact_dir']+'/'+a.name
 dest='/scratch/artifacts/learnable-weight-decay/soap46-v1/'+a.name
 if a.checkpoint_step:
  if a.checkpoint_step<0:raise ValueError('checkpoint step must be positive')
  suffix=f'/step{a.checkpoint_step:04d}';source+=suffix;dest+=suffix
 if not a.execute:print(json.dumps(dict(source=source,destination=dest)));return
 result=cli('gpu','exec',a.gpu,'--timeout','300','--','/opt/venv211/bin/python','-c',REMOTE,source,dest,str(a.checkpoint_step))
 local=pathlib.Path('runs/soap46-v1')/a.name
 if a.checkpoint_step:local=local/f'step{a.checkpoint_step:04d}'
 local.mkdir(parents=True,exist_ok=True)
 (local/'archive_result.json').write_text(result)
 for name in (('complete.json',) if a.checkpoint_step else ('metrics.jsonl','config.json','launch_plan.json','finished.json','complete.json','data_manifest.json','stdout.log')):
  temporary=local/(name+'.incomplete');cli('gpu','pull',a.gpu,source+'/'+name,str(temporary));temporary.replace(local/name)
 # Closed records must be completely parseable; no silently dropped final rows.
 if not a.checkpoint_step:
  for line in (local/'metrics.jsonl').read_text().splitlines():json.loads(line)
 print(result)
if __name__=='__main__':main()
