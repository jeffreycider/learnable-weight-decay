"""Optional baseline checkpoint branching and fresh-run MuonH radius scaling."""
from pathlib import Path
import hashlib
import random
import numpy as np
import torch
import yaml
from chisel.muonh import MuonH


def optimizer_settings(spec):
    return {key: value for key, value in spec['optimizer'].items() if key not in ('lr', 'compile', 'fused')}


def radius():
    @torch.no_grad()
    def hook(config, state):
        multiplier = config.get('hyperball_radius_multiplier', 1.)
        if not np.isfinite(multiplier) or multiplier <= 0:
            raise ValueError('hyperball radius multiplier must be finite and positive')
        for optimizer in state.optimizer.groups.values():
            if isinstance(optimizer, MuonH):
                for group in optimizer.param_groups:
                    for parameter in group['params']:
                        parameter.mul_(multiplier)
    return hook


def baseline_checkpoint():
    """Load an ordinary harness checkpoint before constructing R=W and C=0.

    Model, data, seeds, optimizer algorithms and gradient units must match. LR may
    change; initialization radii cannot change when branching existing weights.
    """
    def hook(config, state):
        if not config.get('baseline_checkpoint'):
            return
        if config.get('resume'):
            raise ValueError('choose either baseline_checkpoint or resume')
        path = Path(str(config['baseline_checkpoint']).format(rank=state.extras['rank']))
        saved = torch.load(path, map_location='cpu', weights_only=False)
        previous = saved['config']
        if saved['extras'] or saved['optimizer'].keys() != state.optimizer.groups.keys():
            raise ValueError('branch source must be an ordinary baseline checkpoint')
        for key in ('model', 'data', 'seeds', 'train_steps', 'gradient_units', 'initialize_model'):
            if previous.get(key) != config.get(key):
                raise ValueError(f'baseline branch mismatch: {key}')
        for key in ('world_size', 'autocast', 'matmul_precision'):
            if previous['runtime'][key] != config['runtime'][key]:
                raise ValueError(f'baseline runtime mismatch: {key}')
        if config.get('hyperball_radius_multiplier', 1.) != previous.get('hyperball_radius_multiplier', 1.):
            raise ValueError('branching inherits checkpoint radii; tune radius on fresh runs')
        for name, spec in config['optimizers'].items():
            old = previous['optimizers'][name]
            if old['parameters'] != spec['parameters'] or optimizer_settings(old) != optimizer_settings(spec):
                raise ValueError(f'baseline optimizer mismatch: {name}')
        for source in ('baseline/model.py', 'chisel/muonh.py', 'chisel/training.py'):
            if saved['source_hashes'].get(source) != hashlib.sha256(Path(source).read_bytes()).hexdigest():
                raise ValueError(f'baseline implementation mismatch: {source}')
        state.model.load_state_dict(saved['model'])
        state.optimizer.load_state_dict(saved['optimizer'])
        state.train_batches.load_state_dict(saved['data'])
        state.step = saved['step']
        for name, optimizer in state.optimizer.groups.items():
            for group in optimizer.param_groups:
                group['initial_lr'] = config['optimizers'][name]['optimizer']['lr']
        torch.set_rng_state(saved['torch_rng'])
        if saved['cuda_rng']:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        random.setstate(saved['python_rng'])
        np.random.set_state(saved['numpy_rng'])
        with path.open('rb') as stream:
            checkpoint_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
        origin = dict(path=str(path.resolve()), sha256=checkpoint_sha256, step=saved['step'],
                      source_hashes=saved['source_hashes'], config=previous)
        record = Path(config['output_dir']) / f'baseline_origin_rank{state.extras["rank"]}.yaml'
        record.write_text(yaml.safe_dump(origin, sort_keys=False))
    return hook
