"""Atomic, same-run checkpointing; all research-affecting config is immutable."""
import os
import random
from pathlib import Path
import numpy as np
import torch

# Different machine/local staging paths and observability do not change learning.
_RESUME_MUTABLE = {'train_data', 'val_data', 'checkpoint_every', 'wandb'}


def validate_resume_config(saved, current):
    differences = {key: (saved.get(key), current.get(key)) for key in saved.keys() | current.keys()
                   if key not in _RESUME_MUTABLE and saved.get(key) != current.get(key)}
    if differences:
        raise ValueError(f'resume would alter research configuration: {differences}')


def rng_state():
    return dict(cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                python=random.getstate(), numpy=np.random.get_state())


def restore_rng(state):
    torch.set_rng_state(state['cpu'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'].cpu())
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])


def build_checkpoint(model, optimizers, adapter, loader, config, step, counters, elapsed_seconds):
    return dict(format_version=1, model=model.state_dict(),
                optimizers=[o.state_dict() for o in optimizers],
                controller=adapter.state_dict() if adapter else None,
                data_cursor=loader.state_dict(), config=config, step=step,
                counters=counters, elapsed_seconds=elapsed_seconds, rng=rng_state())


def restore_checkpoint(state, model, optimizers, adapter, loader, config):
    if state.get('format_version') != 1:
        raise ValueError('unsupported or legacy checkpoint format')
    validate_resume_config(state['config'], config)
    if len(state['optimizers']) != len(optimizers):
        raise ValueError('optimizer count mismatch')
    if (state['controller'] is None) != (adapter is None):
        raise ValueError('controller mode mismatch')
    if state['data_cursor']['batches'] != state['step']:
        raise ValueError('checkpoint data cursor and training clock disagree')
    if not 0 <= state['step'] <= config['max_steps']:
        raise ValueError('checkpoint outside configured run')
    # load_state_dict replaces group settings: reject silent beta/decay/LR-base changes.
    for optimizer, saved in zip(optimizers, state['optimizers']):
        if len(optimizer.param_groups) != len(saved['param_groups']):
            raise ValueError('optimizer group count mismatch')
        for actual, prior in zip(optimizer.param_groups, saved['param_groups']):
            for key in actual.keys() | prior.keys():
                if key not in {'params', 'lr'} and actual.get(key) != prior.get(key):
                    raise ValueError(f'optimizer hyperparameter changed on resume: {key}')
            if 'initial_lr' not in actual and actual['lr'] != prior['lr']:
                raise ValueError('optimizer learning rate changed on resume')
    model.load_state_dict(state['model'])
    for optimizer, saved in zip(optimizers, state['optimizers']):
        optimizer.load_state_dict(saved)
    loader.load_state_dict(state['data_cursor'])
    if adapter:
        adapter.load_state_dict(state['controller'])
    # Pending-graph reconstruction happens first; it cannot perturb resumed RNG.
    restore_rng(state['rng'])
    return state['step'], state['counters'], state['elapsed_seconds']


def save_atomic(state, filename):
    filename = Path(filename)
    temporary = filename.with_name(filename.name+'.tmp')
    torch.save(state, temporary)
    with temporary.open('rb') as f:
        os.fsync(f.fileno())
    os.replace(temporary, filename)
