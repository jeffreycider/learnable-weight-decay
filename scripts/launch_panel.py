"""Launch the predeclared four-arm panel on already allocated Strange Loop leases."""
import concurrent.futures
import json
from pathlib import Path
import shlex
import subprocess

ARMS = [('/tmp/lwd-baseline-gpu.json','baseline','pilot_reference','baseline'),
        ('/tmp/lwd-gpu.json','reference','pilot_reference','learned'),
        ('/tmp/lwd-k2-gpu.json','k2','pilot_k2','learned'),
        ('/tmp/lwd-tolerance-gpu.json','tolerance10x','pilot_tolerance10x','learned')]

def cli(*args):
    return subprocess.run(['strangeloop',*args],check=True,text=True,capture_output=True).stdout

def launch(arm):
    leasefile,label,config,mode=arm
    lease=json.loads(Path(leasefile).read_text()); gpu=lease['id']; output=lease['artifact_dir']+'/pilot-'+label
    status=json.loads(cli('gpu','status',gpu,'--json'))
    if status['gpu_type'] != 'H100' or status['status'] != 'ready':
        raise RuntimeError(f'Lease not a ready H100: {status}')
    cli('gpu','refresh',gpu)
    script='mkdir -p /workspace/lwd /tmp/fineweb10B; tar -xzf /scratch/staging/lwd-source-20261108-panel.tar.gz -C /workspace/lwd; '
    for file in ('fineweb_train_000001.bin','fineweb_train_000002.bin','fineweb_val_000000.bin'):
        script+=f'test -f /tmp/fineweb10B/{file} || cp /scratch/datasets/fineweb10B/{file} /tmp/fineweb10B/{file}; '
    cli('gpu','exec',gpu,'--','bash','-c',script)
    name='lwd-pilot-'+label+'-seed0'
    cli('gpu','declare',gpu,'--name',name,'--description',f'Predeclared 256-step {label} pilot, original PR323 3250-step LR schedule, seed0. Exploratory, not convergence evidence.')
    cmd=f'cd /workspace/lwd; nohup /opt/venv211/bin/torchrun --standalone --nproc_per_node=1 -m baseline.train --config configs/{config}.json --mode {mode} --output {shlex.quote(output)} > {shlex.quote(output+".log")} 2>&1 < /dev/null &'
    cli('gpu','exec',gpu,'--','bash','-c',cmd)
    result=dict(label=label,gpu=gpu,name=name,output=output,config=config,mode=mode)
    print(json.dumps(result),flush=True)
    return result

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records=list(pool.map(launch,ARMS))
    Path('runs').mkdir(exist_ok=True)
    Path('runs/panel_leases.json').write_text(json.dumps(records,indent=2)+'\n')
