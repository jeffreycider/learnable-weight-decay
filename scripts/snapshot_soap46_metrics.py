"""Copy a validated immutable snapshot of live metrics; defer only a partial tail."""
import argparse,json,pathlib,subprocess,base64,gzip,hashlib
REMOTE=r'''
import pathlib,json,sys,base64,gzip,hashlib
p=pathlib.Path(sys.argv[1]);data=(p/'metrics.jsonl').read_bytes();deferred=not data.endswith(b'\n')
if deferred:data=data[:data.rfind(b'\n')+1]
rows=[json.loads(line) for line in data.splitlines()]
d=p.parent/'metrics_snapshots';d.mkdir(exist_ok=True);target=d/(p.name+'.jsonl');tmp=target.with_suffix('.incomplete');tmp.write_bytes(data);tmp.replace(target)
print(json.dumps({'path':str(target),'rows':len(rows),'deferred_partial_tail':deferred,'sha256':hashlib.sha256(data).hexdigest(),'payload':base64.b64encode(gzip.compress(data,mtime=0)).decode()}))
'''
def call(*args):
 r=subprocess.run(['strangeloop',*args],capture_output=True,text=True,timeout=90)
 if r.returncode:raise RuntimeError(r.stderr or r.stdout)
 return r.stdout

def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',required=True);p.add_argument('--name',required=True);a=p.parse_args()
 status=json.loads(call('gpu','status',a.gpu,'--json'));record=json.loads(call('gpu','exec',a.gpu,'--','python','-c',REMOTE,status['artifact_dir']+'/'+a.name))
 target=pathlib.Path('runs/soap46-v1')/a.name/'metrics-live.jsonl';target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_suffix('.incomplete')
 data=gzip.decompress(base64.b64decode(record.pop('payload')))
 if hashlib.sha256(data).hexdigest()!=record['sha256']:raise RuntimeError('snapshot transport hash mismatch')
 tmp.write_bytes(data)
 for row in tmp.read_text().splitlines():json.loads(row)
 tmp.replace(target);print(json.dumps(dict(local=str(target),**record)))
if __name__=='__main__':main()
