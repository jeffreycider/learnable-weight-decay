"""Run records and restart state; all policies are configured as lifecycle hooks."""
from pathlib import Path
import hashlib
import json
import os
import platform
import random
import time
import tarfile
import importlib.metadata
import numpy as np
import torch
import yaml


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_copy(item) for item in value)
    return value


def save_atomic(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def open_run():
    def hook(config, state):
        root = Path(config['output_dir'])
        root.mkdir(parents=True, exist_ok=True)
        if state.extras['rank'] == 0:
            recipe = root / 'recipe.yaml'
            if recipe.exists() and not config.get('resume'):
                raise FileExistsError(f'refuse to overwrite {root}')
            (root / ('resume.yaml' if recipe.exists() else 'recipe.yaml')).write_text(yaml.safe_dump(config, sort_keys=False))
            sources = sorted(Path('chisel').glob('*.py')) + [Path('baseline/model.py'), Path('baseline/data.py')]
            hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
            state.extras['source_hashes'] = hashes
            (root / 'sources.yaml').write_text(yaml.safe_dump(hashes))
            with tarfile.open(root / 'source.tar.gz', 'w:gz') as archive:
                for path in sources + list(Path('recipes').glob('*.yaml')) + list(Path('vendor/cifar_recipe/utils').glob('*.py')):
                    archive.add(path, arcname=str(path))
            (root / 'packages.yaml').write_text(yaml.safe_dump({package.metadata['Name']: package.version for package in importlib.metadata.distributions()}))
            (root / 'environment.yaml').write_text(yaml.safe_dump(dict(python=platform.python_version(), torch=str(torch.__version__), cuda=torch.version.cuda)))
        state.extras['started_at'] = time.monotonic()
    return hook


def progress(every):
    def hook(config, state):
        if state.extras['rank'] or (state.step + 1) % every:
            return
        row = dict(step=state.step + 1, train_loss=state.extras['train_loss'],
                   seconds=time.monotonic() - state.extras['started_at'],
                   training_forwards=state.extras.get('training_forwards', 0))
        row.update(state.extras.get('metrics', {}))
        print(json.dumps(row), flush=True)
        with (Path(config['output_dir']) / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
    return hook


def snapshot(config, state):
    return cpu_copy(dict(source_hashes=state.extras.get('source_hashes', {}), config=config, step=state.step, model=state.model.state_dict(),
                         optimizer=state.optimizer.state_dict(), data=state.train_batches.state_dict(),
                         torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                         python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                         extras={name: obj.state_dict() for name, obj in state.extras.get('checkpointables', {}).items()}))


def checkpoint(every):
    def hook(config, state):
        if (state.step + 1) % every and state.step + 1 != min(config['train_steps'], config.get('stop_step', config['train_steps'])):
            return
        path = Path(config['output_dir']) / f'latest_rank{state.extras["rank"]}.pt'
        version = path.with_name(f'checkpoint_{state.step + 1:06d}_rank{state.extras["rank"]}.pt')
        save_atomic(snapshot(config, state), version)
        temporary = path.with_suffix('.link')
        temporary.unlink(missing_ok=True)
        os.link(version, temporary)
        temporary.replace(path)
    return hook


def restore():
    def hook(config, state):
        if not config.get('resume'):
            return
        path = str(config['resume']).format(rank=state.extras['rank'])
        saved = torch.load(path, map_location='cpu', weights_only=False)
        ignored = {'resume', 'output_dir', 'stop_step'}
        if {k: v for k, v in config.items() if k not in ignored} != {k: v for k, v in saved['config'].items() if k not in ignored}:
            raise ValueError('resume configuration differs from checkpoint')
        if saved.get('source_hashes', {}) != state.extras.get('source_hashes', {}):
            raise ValueError('resume source hashes differ from checkpoint')
        state.model.load_state_dict(saved['model'])
        state.optimizer.load_state_dict(saved['optimizer'])
        state.train_batches.load_state_dict(saved['data'])
        for name, value in saved['extras'].items():
            state.extras['checkpointables'][name].load_state_dict(value)
        state.step = saved['step']
        torch.set_rng_state(saved['torch_rng'])
        if saved['cuda_rng']:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        random.setstate(saved['python_rng'])
        np.random.set_state(saved['numpy_rng'])
    return hook
