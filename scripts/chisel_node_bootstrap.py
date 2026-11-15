"""Run bounded Chisel readiness checks and save their status on a new node."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    root=Path(__file__).resolve().parents[1]
    output=root/'runs/lambda-bootstrap'
    output.mkdir(parents=True,exist_ok=True)
    state=dict(status='running',pid=os.getpid(),started_unix=time.time(),scope='Chisel environment and synthetic smoke checks only',stages=[])
    def save():
        path=output/'workflow.json';tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(path)
    save()
    try:
        manifest=json.loads((root/'bootstrap_source_manifest.json').read_text())
        for name,digest in manifest['files'].items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Source checksum mismatch: '+name)
        state['source_commit']=manifest['base_commit']
        commands=[
            ('dependencies',[sys.executable,'-m','pip','check']),
            ('unit-tests',[sys.executable,'-m','pytest','-q','tests/test_grouped_coordinate_chisel.py',
                'tests/test_adaptive_preservation_actor.py','tests/test_cpu_direct_streamed_backend.py',
                'tests/test_forward_precision_diagnostic.py','tests/test_saved_adam_audit.py']),
            ('eight-gpu-smoke',[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
                'scripts/chisel_node_smoke.py',str(output/'smoke')])]
        env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TORCHINDUCTOR_COMPILE_THREADS='2',
                 WANDB_MODE='disabled',PYTHONUNBUFFERED='1')
        for name,command in commands:
            row=dict(name=name,command=command,status='running',started_unix=time.time())
            state['stages'].append(row);save()
            with (output/(name+'.log')).open('w') as log:
                child=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
                row['pid']=child.pid;save()
                try:code=child.wait(timeout=360)
                except subprocess.TimeoutExpired:
                    child.terminate();child.wait(timeout=30);raise
            row.update(exit_code=code,status='completed' if code==0 else 'failed',seconds=time.time()-row['started_unix']);save()
            if code:raise RuntimeError(name+' failed; inspect '+str(output/(name+'.log')))
        (output/'pip-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
        state.update(status='completed',ended_unix=time.time())
    except BaseException as exc:
        state.update(status='failed',error=repr(exc),ended_unix=time.time());save();raise
    save()


if __name__=='__main__':main()
