"""Run on the capture node after torchrun exits; preserve all eight resumable shards."""
import argparse,hashlib,json,math,os,pathlib,shutil,subprocess,tarfile,time

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--launch',required=True);a=p.parse_args()
 launch=json.loads(pathlib.Path(a.launch).read_text());out=pathlib.Path(launch['output']);bank=pathlib.Path(launch['config']['state_bank']['directory'])
 rows=[json.loads(x) for x in (out/'metrics.jsonl').read_text().splitlines()]
 finish=[r for r in rows if r.get('event')=='finished'];assert len(finish)==1 and finish[0]['steps']==1000
 train=[r for r in rows if r.get('event')=='train'];assert [r['step'] for r in train]==list(range(1,1001))
 for r in train:
  assert math.isfinite(r['train_loss']) and math.isfinite(r['capture_allones_relative_error'])
  assert r['training_forwards_per_rank']==r['training_backwards_per_rank']==r['captured_states']==r['step']
  assert r['capture_extra_gpt_forwards']==r['capture_extra_gpt_backwards']==0
 assert all(math.isfinite(r['val_loss']) for r in rows if r.get('event')=='validation')
 complete=json.loads((bank/'complete.json').read_text());assert complete['steps']==1000 and digest(bank/'index.jsonl')==complete['index_sha256']
 entries=[json.loads(x) for x in (bank/'index.jsonl').read_text().splitlines()]
 assert [r['completed_steps'] for r in entries]==list(range(1,1001))
 for row in entries:assert (bank/row['file']).stat().st_size==row['file_bytes']
 if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('capture GPU process still active')
 target=bank/'resume1000';target.mkdir(exist_ok=False)
 import torch
 files={}
 for rank in range(8):
  src=out/f'final_rank{rank}.pt';state=torch.load(src,map_location='cpu',weights_only=False,mmap=True)
  assert state['step']==1000 and state['config']['seed']==launch['seed'] and state['config']['world_size']==8
  assert state['data_cursor']['batches']==1000 and state['data_cursor']['rank']==rank
  assert state['rng']['cpu'] is not None and len(state['optimizers'])==2
  del state
  dst=target/f'rank{rank}.pt';shutil.copy2(src,dst)
  sha=digest(src);assert digest(dst)==sha
  files[dst.name]=dict(sha256=sha,bytes=dst.stat().st_size,source=str(src))
 (target/'manifest.json').write_text(json.dumps(dict(step=1000,world_size=8,seed=launch['seed'],files=files,source_sha256=launch['source_sha256']),indent=2)+'\n')
 archive=bank/'run_record.tar.gz'
 with tarfile.open(archive,'w:gz') as tar:
  for path in sorted(out.rglob('*')):
   if path.is_file() and not path.is_symlink() and path.suffix!='.pt':tar.add(path,arcname=str(path.relative_to(out)))
  tar.add(__file__,arcname='archive_multiseed_capture.py')
 summary=dict(seed=launch['seed'],states=1000,state_bytes=sum(r['file_bytes'] for r in entries),index_sha256=complete['index_sha256'],manifest_sha256=digest(bank/'manifest.json'),resume_manifest_sha256=digest(target/'manifest.json'),run_record_sha256=digest(archive),finished=finish[0],source_sha256=launch['source_sha256'],completed_at=time.time())
 (bank/'archive_manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
 subprocess.run(['sync','/scratch'],check=True);subprocess.run(['sync','/persist'],check=True)
 print(json.dumps(summary))
if __name__=='__main__':main()
