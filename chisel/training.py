"""Injected supervised task: model, data, loss, precision and optimizer groups."""
from contextlib import nullcontext
import os
import re
import torch
import torch.distributed as dist
from chisel.harness import build, locate
from chisel.muonh import Optimizers


def initialize():
    def hook(config, state):
        runtime = config['runtime']
        device = torch.device(runtime['device'])
        if device.type == 'cuda':
            device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', device.index or 0)))
            torch.cuda.set_device(device)
        torch.set_num_threads(runtime['threads'])
        torch._dynamo.config.recompile_limit = runtime['recompile_limit']
        torch.set_float32_matmul_precision(runtime['matmul_precision'])
        torch.manual_seed(config['seeds']['initialization'])
        world = int(os.environ.get('WORLD_SIZE', 1))
        if world != runtime['world_size']:
            raise ValueError('world size differs from YAML')
        if world > 1 and not dist.is_initialized():
            dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
        state.extras.update(device=device, world_size=world, rank=dist.get_rank() if world > 1 else 0)
    return hook


def model():
    def hook(config, state):
        state.model = build(config['model']).to(state.extras['device'])
        if config.get('initialize_model'):
            locate(config['initialize_model'])(state.model)
        for parameter in state.model.parameters():
            if state.extras['world_size'] > 1:
                dist.broadcast(parameter.detach(), 0)
        state.extras['forward'] = torch.compile(state.model, **config['compile']['kwargs']) if config['compile']['enabled'] else state.model
    return hook


def optimizers():
    def hook(config, state):
        named = dict(state.model.named_parameters())
        remaining = dict(named)
        groups = {}
        for name, spec in config['optimizers'].items():
            selected = {key: value for key, value in named.items() if re.fullmatch(spec['parameters'], key)}
            if not selected or any(key not in remaining for key in selected):
                raise ValueError(f'empty or overlapping optimizer selection: {name}')
            groups[name] = build(spec['optimizer'], params=list(selected.values()))
            for key in selected:
                remaining.pop(key)
            for group in groups[name].param_groups:
                group['initial_lr'] = group['lr']
                if spec['tensor_lr']:
                    group['lr'] = torch.tensor(group['lr'], device=state.extras['device'])
        if remaining:
            raise ValueError(f'parameters without an optimizer: {list(remaining)}')
        state.optimizer = Optimizers(groups)
    return hook


def data():
    def hook(config, state):
        state.train_batches = build(config['data'], device=state.extras['device'],
                                    rank=state.extras['rank'], world_size=state.extras['world_size'])
        state.backward = backward
    return hook


def autocast(config, state):
    dtype = config['runtime']['autocast']
    return nullcontext() if dtype is None else torch.autocast(state.extras['device'].type, dtype=getattr(torch, dtype))


def backward(config, state):
    batch = next(state.train_batches)
    if isinstance(batch, tuple):
        batch = dict(inputs=batch[0], targets=batch[1])
    state.extras['batch'] = batch
    inputs, targets = batch['inputs'], batch['targets']
    microbatch = config['microbatch_size']
    local_count = targets.shape[0] if targets.is_floating_point() else targets.numel()
    denominator = local_count if config['gradient_units'] == 'mean' else 1
    if config['gradient_units'] not in ('mean', 'sum'):
        raise ValueError('gradient_units must be mean or sum')
    loss_fn = locate(config['loss']['name'])
    total = torch.zeros((), device=inputs.device)
    for start in range(0, len(inputs), microbatch):
        with autocast(config, state):
            logits = state.extras['forward'](inputs[start:start + microbatch])
            target = targets[start:start + microbatch]
            target = target.reshape(-1, logits.shape[-1]) if target.is_floating_point() else target.reshape(-1)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), target, reduction='sum', **config['loss']['kwargs'])
        (loss / denominator).backward()
        total += loss.detach()
    for parameter in state.model.parameters():
        if parameter.grad is None:
            raise RuntimeError('missing task gradient')
        if state.extras['world_size'] > 1:
            dist.all_reduce(parameter.grad)
            if config['gradient_units'] == 'mean':
                parameter.grad.div_(state.extras['world_size'])
    if not torch.isfinite(total):
        raise FloatingPointError('non-finite training loss')
    state.extras['train_loss'] = (total / local_count).item()
    state.extras['training_forwards'] = state.extras.get('training_forwards', 0) + (len(inputs) + microbatch - 1) // microbatch


def schedule():
    def hook(config, state):
        for name, spec in config['optimizers'].items():
            rule = spec['schedule']
            factor = locate(rule['name'])(state.step, config['train_steps'], **rule['kwargs'])
            for group in state.optimizer.groups[name].param_groups:
                value = group['initial_lr'] * factor
                if torch.is_tensor(group['lr']):
                    group['lr'].fill_(value)
                else:
                    group['lr'] = value
    return hook
