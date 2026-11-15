"""Run YAML recording recipes with bounded local staging and verified shared copies."""
from pathlib import Path
import os
import shutil
import subprocess
import time
import yaml
from chisel.harness import resolve
from chisel.storage import atomic_json


def staged_recipe(original, local_root, microbatch):
    config = resolve(original)
    name = Path(config['output_dir']).name + '-local-stage'
    config['output_dir'] = str(Path(local_root) / name)
    config['microbatch_size'] = microbatch
    config['pre_forward'].insert(0, dict(name='chisel.performance.start'))
    config['post_optimizer'].insert(0, dict(name='chisel.performance.finish'))
    for hook in config['post_optimizer']:
        if hook['name'] == 'chisel.evaluation.evaluate':
            hook['hyperparams']['microbatch_size'] = microbatch
    return config


def run(spec):
    local_root, shared_root = Path(spec['local_root']), Path(spec['shared_root'])
    local_root.mkdir(parents=True, exist_ok=True)
    while not all(Path(path).exists() for path in spec['wait_for']):
        time.sleep(30)
    queue = Path(spec['queue_dir'])
    queue.mkdir(parents=True, exist_ok=True)
    environment = os.environ | {'OMP_NUM_THREADS': str(spec['threads'])}
    environment.pop('CUDA_VISIBLE_DEVICES', None)
    for original in spec['recipes']:
        config = staged_recipe(original, local_root, spec['microbatch_size'])
        local = Path(config['output_dir'])
        shared = shared_root / local.name
        if (shared / 'queue_complete.json').exists():
            continue
        if local.exists() or shared.exists():
            raise FileExistsError(f'ambiguous existing run; inspect before resume: {local}')
        while shutil.disk_usage(local_root).free < spec['minimum_local_free_bytes']:
            print('waiting for verified staging copies to free local space', flush=True)
            time.sleep(30)
        while True:
            free = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                                           '--format=csv,noheader,nounits'], text=True)
            if min(int(value) for value in free.splitlines()) >= spec['minimum_gpu_free_mib']:
                break
            time.sleep(30)
        local.mkdir()
        recipe = queue / (local.name + '.yaml')
        recipe.write_text(yaml.safe_dump(config, sort_keys=False))
        replication = queue / (local.name + '-replication.yaml')
        replication.write_text(yaml.safe_dump(dict(source=str(local), destination=str(shared),
            workers=spec['copy_workers'], poll_seconds=1, relocate_after_copy=True)))
        with replication.with_suffix('.log').open('w') as log:
            copier = subprocess.Popen([spec['python'], '-u', '-m', 'chisel.replicate', str(replication)],
                cwd=spec['cwd'], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        with recipe.with_suffix('.log').open('w') as log:
            result = subprocess.run([spec['python'], '-m', 'torch.distributed.run', '--standalone',
                f'--nproc_per_node={config["runtime"]["world_size"]}', '-m', 'chisel.harness', str(recipe)],
                cwd=spec['cwd'], env=environment, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'recording failed: {recipe}; exit={result.returncode}')
        atomic_json(local / 'queue_complete.json', dict(recipe=str(recipe), completed=time.time(),
                                                       steps=config['train_steps']))
        if copier.poll() not in (None, 0):
            raise RuntimeError(f'recording completed but replication failed: {replication}')
        print(f'completed recording: {local.name}', flush=True)


if __name__ == '__main__':
    import sys
    run(yaml.safe_load(Path(sys.argv[1]).read_text()))
