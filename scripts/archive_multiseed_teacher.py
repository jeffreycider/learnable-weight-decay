"""Validate and preserve a completed exact FP32 teacher bank without duplicating tensors."""
import argparse,json,pathlib,subprocess,tarfile,time
from archive_multiseed_capture import digest

def main():
 p=argparse.ArgumentParser();p.add_argument('--launch',required=True);a=p.parse_args()
 launch=json.loads(pathlib.Path(a.launch).read_text());out=pathlib.Path(launch['output']);cache=pathlib.Path(launch['cache']);bank=pathlib.Path(launch['bank'])
 if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('teacher processes still active')
 m=json.loads((cache/'manifest.json').read_text());assert m['complete'] and len(m['panels'])==2000
 assert m['execution']['precision']['rms_epsilon']==2**-23 and m['execution']['gpt_precision']=='fp32'
 assert m['source_state_manifest']['provenance']['seed']==launch['seed']
 assert m['source_hashes']==launch['recipe']['source_files_sha256']
 assert {(r['state_id'],r['role']) for r in m['panels']}=={(s,r) for s in range(1,1001) for r in ('train','audit')}
 for r in m['panels']:
  assert len(r['sequence_ids'])=={'train':128,'audit':64}[r['role']]
  assert r['microbatch_sequences']==16
  file=cache/f"state_{r['state_id']:04d}_{r['role']}.pt"
  assert file.stat().st_size==r['bytes']
  sidecar=json.loads(file.with_suffix('.json').read_text())
  assert sidecar['sha256']==r['sha256'] and sidecar['state_sha256']==r['state_sha256']
  if r['state_id'] in (1,2,3,999,1000):assert r['qualification_max_logprob_error']==0
 archive=cache/'run_record.tar.gz'
 with tarfile.open(archive,'w:gz') as tar:
  for path in sorted(out.rglob('*')):
   if path.is_file() and not path.is_symlink():tar.add(path,arcname=str(path.relative_to(out)))
  tar.add(__file__,arcname='archive_multiseed_teacher.py')
 summary=dict(seed=launch['seed'],states=1000,panels=2000,total_bytes=m['total_bytes'],elapsed_seconds=m['elapsed_seconds'],manifest_sha256=digest(cache/'manifest.json'),state_manifest_sha256=digest(bank/'manifest.json'),state_index_sha256=digest(bank/'index.jsonl'),source_archive_sha256=launch['recipe']['source_archive_sha256'],run_record_sha256=digest(archive),completed_at=time.time())
 (cache/'archive_manifest.json').write_text(json.dumps(summary,indent=2)+'\n');subprocess.run(['sync','/scratch'],check=True);subprocess.run(['sync','/persist'],check=True);print(json.dumps(summary))
if __name__=='__main__':main()
