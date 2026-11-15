"""CPU-only same-record torch.save+rehash versus streaming SHA; no GPT calls."""
import argparse,hashlib,json,os,statistics,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from rewriter.streaming_save import HashingWriter

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for x in iter(lambda:f.read(8<<20),b''):h.update(x)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--state',required=True);p.add_argument('--output',required=True);p.add_argument('--repeats',type=int,default=3);a=p.parse_args()
 root=Path(a.output);root.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
 source=Path(a.state);record=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
 import wandb
 run=wandb.init(project='learnable-weight-decay',name='state-writer-streaming-sha-qualification',mode='offline',dir=str(root),config=vars(a))
 rows=[];expected=None
 for repetition in range(a.repeats+1):
  for mode in (('reference','streaming') if repetition%2==0 else ('streaming','reference')):
   path=root/f'{mode}_{repetition}.pt';start=time.perf_counter()
   with path.open('wb') as f:
    if mode=='reference':torch.save(record,f);f.flush();os.fsync(f.fileno())
    else:
     writer=HashingWriter(f);torch.save(record,writer);writer.flush();os.fsync(f.fileno());sha=writer.hexdigest()
   if mode=='reference':sha=digest(path)
   elapsed=time.perf_counter()-start
   # Qualification reread is outside the measured candidate interval.
   actual=digest(path);assert actual==sha
   if expected is None:expected=sha
   assert sha==expected,'serializer bytes changed between transports'
   row=dict(mode=mode,repetition=repetition,warmup=repetition==0,seconds=elapsed,bytes=path.stat().st_size,sha256=sha);rows.append(row)
   print(json.dumps(row),flush=True);run.log(dict(objective=elapsed,**row));path.unlink()
 medians={m:statistics.median(r['seconds'] for r in rows if r['mode']==m and not r['warmup']) for m in ('reference','streaming')}
 result=dict(source=str(source),source_sha256=digest(source),record_bytes=rows[0]['bytes'],rows=rows,medians=medians,speedup=medians['reference']/medians['streaming'],byte_identity=True,gpt_forwards=0,gpt_backwards=0,torch=str(torch.__version__))
 (root/'result.json').write_text(json.dumps(result,indent=2)+'\n');run.log(dict(speedup=result['speedup']));run.finish();print(json.dumps(result))
if __name__=='__main__':main()
