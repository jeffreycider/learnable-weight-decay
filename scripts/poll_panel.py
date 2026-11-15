"""Fetch current panel records and attach each offline log to its declared run."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import subprocess


def cli(*args):
    return subprocess.run(['strangeloop',*args],check=True,text=True,capture_output=True).stdout


def fetch(record, outputroot=Path('runs/panel')):
    remote='''import json, pathlib
p=pathlib.Path(OUTPUT)
files=['metrics.jsonl','config.json','environment.json','data_manifest.json']
r={name:(p/name).read_text() for name in files if (p/name).exists()}
r['wandb']= [str(x.parent) for x in (p/'wandb').glob('offline-run-*/*.wandb')]
print(json.dumps(r))
'''.replace('OUTPUT',repr(record['output']))
    data=json.loads(cli('gpu','exec',record['gpu'],'--','/opt/venv211/bin/python','-c',remote))
    out=outputroot/record['label'];out.mkdir(parents=True,exist_ok=True)
    for name,value in data.items():
        if name!='wandb': (out/name).write_text(value)
    marker=out/'declared.json'
    if data['wandb'] and (not marker.exists() or json.loads(marker.read_text()) != data['wandb']):
        cli('gpu','declare',record['gpu'],'--name',record['name'],'--wandb-path',data['wandb'][0])
        marker.write_text(json.dumps(data['wandb']))
    metrics=[json.loads(line) for line in data.get('metrics.jsonl','').splitlines() if line]
    trains=[x for x in metrics if x.get('event')=='train']
    vals=[x for x in metrics if x.get('event')=='validation']
    latest=trains[-1] if trains else {}
    result=dict(label=record['label'],step=latest.get('step'),train_loss=latest.get('train_loss'),
        kappa=latest.get('kappa'),committed_norm=latest.get('committed_norm'),
        elapsed=latest.get('elapsed_seconds'),validation=vals[-1] if vals else None,
        finished=any(x.get('event')=='finished' for x in metrics))
    print(json.dumps(result),flush=True)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--leases',type=Path,default=Path('runs/panel_leases.json'))
    parser.add_argument('--outputroot',type=Path,default=Path('runs/panel'))
    parser.add_argument('--status',type=Path,help='Default: sibling OUTPUTROOT_status.json')
    args=parser.parse_args()
    records=json.loads(args.leases.read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(lambda record:fetch(record,args.outputroot),records))
    status=args.status or args.outputroot.with_name(args.outputroot.name+'_status.json')
    status.parent.mkdir(parents=True,exist_ok=True)
    status.write_text(json.dumps(results,indent=2)+'\n')
