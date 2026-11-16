"""Explicit sampler transition from a committed offline generation, retaining learned state."""
import random
from pathlib import Path
import numpy as np
import torch
import yaml
from chisel.harness import build
from chisel.storage import sha256 as digest


def comparable(config):
    ignored = {'output_dir', 'setup', 'resume', 'resume_compatibility',
               'compile_preservation_loss', 'cache_normalized_sensors'}
    result = {key: value for key, value in config.items() if key not in ignored}
    for key in ('data', 'training_audit_data'):
        if result.get(key):
            result[key] = {k: v for k, v in result[key].items() if k not in ('_target_', 'owner_count')}
    result['private_probes'] = {k: v for k, v in result['private_probes'].items() if k != 'cache_bytes'}
    return result


def initialize(generation, sha256):
    """Keep the update clock and all optimizer state; begin the declared new sampler order."""
    def hook(config, state):
        if config.get('resume'):
            return
        path = Path(generation)
        if digest(path) != sha256:
            raise ValueError('parent generation SHA256 mismatch')
        receipt = yaml.safe_load(path.read_text())
        world = state.extras['world_size']
        if len(receipt['ranks']) != world:
            raise ValueError('this transition preserves world size')
        first, local = None, None
        private, duals = {}, {}
        for rank, row in enumerate(receipt['ranks']):
            source = path.parent / row['path']
            if digest(source) != row['sha256']:
                raise ValueError('parent rank checkpoint SHA256 mismatch')
            saved = torch.load(source, map_location='cpu', weights_only=False)
            if saved['step'] + 1 != receipt['step'] or comparable(saved['config']) != comparable(config):
                raise ValueError('repartition changed training settings or parent update clock')
            if first is None:
                first = saved
            elif any(not torch.equal(value, first['model'][name]) for name, value in saved['model'].items()):
                raise ValueError('parent replicas disagree on shared Chisel weights')
            if rank == state.extras['rank']:
                local = saved
            sampler = build(saved['config']['data'])
            def owner(identity):
                return sampler.owner(identity, world) if hasattr(sampler, 'owner') else int(identity, 16) % world
            private.update({identity: value for identity, value in saved['extras']['private_probes']['latest'].items()
                            if owner(identity) == rank})
            duals.update({identity: value for identity, value in saved['extras'].get('checkpoint_duals', {}).items()
                          if owner(identity) == rank})
        state.model.load_state_dict(local['model'])
        state.optimizer.load_state_dict(local['optimizer'])
        state.extras['private_probes'].load_state_dict(dict(latest=private))
        state.extras['dual'].load_state_dict(local['extras']['dual'])
        if 'checkpoint_duals' in state.extras:
            state.extras['checkpoint_duals'].load_state_dict(duals)
        state.step = local['step']
        torch.set_rng_state(local['torch_rng'])
        if local['cuda_rng']:
            torch.cuda.set_rng_state_all(local['cuda_rng'])
        random.setstate(local['python_rng'])
        np.random.set_state(local['numpy_rng'])
        state.extras['repartition_parent'] = dict(path=str(path), sha256=sha256, step=receipt['step'])
    return hook
