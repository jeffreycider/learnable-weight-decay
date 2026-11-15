"""Fetch bounded progress and phase timings from a SOAP46 run."""
import argparse
import json
import subprocess
from pathlib import Path


REMOTE = r'''
import json,pathlib,statistics,subprocess,sys
p=pathlib.Path(sys.argv[1]);rows=[]
if (p/'metrics.jsonl').exists():
 text=(p/'metrics.jsonl').read_text();lines=text.splitlines()
 for index,line in enumerate(lines):
  try:rows.append(json.loads(line))
  except json.JSONDecodeError:
   if index==len(lines)-1 and not text.endswith('\n'):break
   raise
train=[r for r in rows if r.get('event')=='train'];val=[r for r in rows if r.get('event')=='validation']
recent=train[-50:];timing={}
for k in ['wall_seconds','gpt_forward_backward_seconds','base_optimizer_seconds','credit_seconds']:
 values=[r[k] for r in recent if k in r]
 if values:timing[k]=statistics.median(values)
result=dict(output=str(p),latest_train=train[-1] if train else None,latest_validation=val[-1] if val else None,median_last50=timing,
 finished=json.loads((p/'finished.json').read_text()) if (p/'finished.json').exists() else None,
 wandb=[str(x) for x in p.glob('wandb/offline-run-*/*.wandb')])
if (p/'launch.json').exists():
 pid=json.loads((p/'launch.json').read_text())['pid'];stat=pathlib.Path(f'/proc/{pid}/stat')
 result['launcher_alive']=stat.exists() and stat.read_text().split(') ',1)[1].split()[0] not in ('Z','X')
if not train and rows:result['last_progress']=rows[-1]
if (p/'stdout.log').exists():result['stdout_tail']=(p/'stdout.log').read_text()[-1600:]
result['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip().splitlines()
print(json.dumps(result))
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',required=True);p.add_argument('--output',required=True)
    p.add_argument('--save',type=Path);a=p.parse_args()
    result=subprocess.run(['strangeloop','gpu','exec',a.gpu,'--','python','-c',REMOTE,a.output],check=True,text=True,capture_output=True,timeout=60)
    record=json.loads(result.stdout)
    if a.save:
        a.save.parent.mkdir(parents=True,exist_ok=True);a.save.write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
