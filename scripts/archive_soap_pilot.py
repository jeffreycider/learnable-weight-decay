"""Preserve a completed SOAP pilot including all optimizer shards, then verify."""
import argparse,json,subprocess
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['fixed','low','high'],required=True);args=p.parse_args()
    local=Path('runs/soap-hyper-pilot-v1');info=json.loads((local/(args.arm+'-launch.json')).read_text())
    target='/scratch/artifacts/learnable-weight-decay/soap-hyper-pilot-v1/'+args.arm
    code=r'''
import pathlib,hashlib,json,shutil,os,sys,time,subprocess
src=pathlib.Path(sys.argv[1]);dst=pathlib.Path(sys.argv[2]);assert (src/'finished.json').exists()
assert len(list(src.glob('final_rank*.pt')))==8
if dst.exists():raise RuntimeError('archive exists; inspect rather than overwrite')
dst.mkdir(parents=True);rows={};start=time.perf_counter()
for p in src.rglob('*'):
 if p.is_symlink():continue
 rel=p.relative_to(src);q=dst/rel
 if p.is_dir():q.mkdir(exist_ok=True);continue
 q.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256()
 with p.open('rb') as f,q.open('wb') as g:
  for chunk in iter(lambda:f.read(8388608),b''):h.update(chunk);g.write(chunk)
 expected=h.hexdigest();verify=hashlib.sha256()
 with q.open('rb') as f:
  for chunk in iter(lambda:f.read(8388608),b''):verify.update(chunk)
 if verify.hexdigest()!=expected:raise RuntimeError('copy hash mismatch '+str(rel))
 rows[str(rel)]={'bytes':q.stat().st_size,'sha256':expected}
manifest={'source':str(src),'files':rows,'complete':True,'ordinary_gpt_iterations':256,
 'training_forward_backward_calls_per_rank':256,'training_forward_backward_calls_global':2048,
 'validation_forwards_global':480,'extra_gpt_training_evaluations':0,'archive_seconds':time.perf_counter()-start}
(dst/'manifest.json').write_text(json.dumps(manifest,indent=2));os.sync();subprocess.run(['sync','/scratch'],check=True)
print(json.dumps({'target':str(dst),'manifest_sha256':hashlib.sha256((dst/'manifest.json').read_bytes()).hexdigest(),'file_count':len(rows),'bytes':sum(x['bytes'] for x in rows.values()),'archive_seconds':manifest['archive_seconds']}))
'''
    r=subprocess.run(['strangeloop','gpu','exec',info['gpu'],'--','/opt/venv211/bin/python','-c',code,info['output'],target],check=True,text=True,capture_output=True,timeout=300)
    (local/(args.arm+'-archive.json')).write_text(r.stdout);print(r.stdout)
    for name in ['manifest.json','config.json','metrics.jsonl','launch_plan.json','data_manifest.json','finished.json']:
        subprocess.run(['strangeloop','scratch','get',target.removeprefix('/scratch/')+'/'+name,str(local/args.arm/name)],check=True,stdout=subprocess.DEVNULL)

if __name__=='__main__':main()
