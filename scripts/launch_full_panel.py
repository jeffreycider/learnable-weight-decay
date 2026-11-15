"""Launch the predeclared four-arm panel on already allocated Strange Loop leases."""
import concurrent.futures
import json
from pathlib import Path
import shlex
import subprocess

ARMS = [('/tmp/lwd-baseline-gpu.json','baseline','full_reference','baseline'),
        ('/tmp/lwd-gpu.json','reference','full_reference','learned'),
        ('/tmp/lwd-k2-gpu.json','k2','full_k2','learned'),
        ('/tmp/lwd-tolerance-gpu.json','tolerance10x','full_tolerance10x','learned')]

def cli(*args):
    return subprocess.run(['strangeloop',*args],check=True,text=True,capture_output=True).stdout

def launch(arm):
    leasefile,label,config,mode=arm
    lease=json.loads(Path(leasefile).read_text()); gpu=lease['id']; output=lease['artifact_dir']+'/full-'+label
    status=json.loads(cli('gpu','status',gpu,'--json'))
    if status['gpu_type'] != 'H100' or status['status'] != 'ready':
        raise RuntimeError(f'Lease not a ready H100: {status}')
    cli('gpu','extend',gpu,'--hours','12')
    cli('gpu','refresh',gpu)
    script='mkdir -p /workspace/lwd /tmp/fineweb10B; tar -xzf /scratch/staging/lwd-source-20261108-deterministic.tar.gz -C /workspace/lwd; '
    for file in [f'fineweb_train_{i:06d}.bin' for i in range(1,19)]+['fineweb_val_000000.bin']:
        script+=f'test -f /tmp/fineweb10B/{file} || cp /scratch/datasets/fineweb10B/{file} /tmp/fineweb10B/{file}; '
    cli('gpu','exec',gpu,'--','bash','-c',script)
    name='lwd-full-'+label+'-seed0'
    cli('gpu','declare',gpu,'--name',name,'--description',f'Predeclared 3250-step {label} run, original PR323 3250-step LR schedule, seed0. Exploratory, not convergence evidence.')
    cmd=f'cd /workspace/lwd; nohup env PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 /opt/venv211/bin/torchrun --standalone --nproc_per_node=1 -m baseline.train --config configs/{config}.json --mode {mode} --output {shlex.quote(output)} > {shlex.quote(output+".log")} 2>&1 < /dev/null &'
    cli('gpu','exec',gpu,'--','bash','-c',cmd)
    result=dict(label=label,gpu=gpu,name=name,output=output,config=config,mode=mode)
    print(json.dumps(result),flush=True)
    return result

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records=list(pool.map(launch,ARMS))
    Path('runs').mkdir(exist_ok=True)
    Path('runs/panel_leases.json').write_text(json.dumps(records,indent=2)+'\n')
