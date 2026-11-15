"""Merge PR323 rank-sharded momentum for frozen-controller observation only.

Example: python scripts/merge_baseline_checkpoint.py --input-dir retained/step1750 \
    --output frozen_step1750.pt

The model and AdamW are replicated; Muon owns group index modulo world_size.
Inputs must be trusted, completed checkpoints from the SAME baseline run. Each
rank is memory-mapped in turn. The resulting artifact retains the source world's
configuration, clock, and rank-zero cursor/RNG, but deliberately has a different
format_version so the training resume API rejects it. No controller is copied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline.model import GPT

FORMAT = 'frozen_observation_merge_v1'


def _equal(left, right, label):
    if isinstance(left, torch.Tensor):
        same = (isinstance(right, torch.Tensor) and left.shape == right.shape
                and left.dtype == right.dtype and torch.equal(left, right))
    elif isinstance(left, dict):
        same = isinstance(right, dict) and left.keys() == right.keys()
        if same:
            for key in left:
                _equal(left[key], right[key], f'{label}.{key}')
    elif isinstance(left, (list, tuple)):
        same = type(left) is type(right) and len(left) == len(right)
        if same:
            for i, (a, b) in enumerate(zip(left, right)):
                _equal(a, b, f'{label}[{i}]')
    else:
        same = type(left) is type(right) and left == right
    if not same:
        raise ValueError(f'checkpoint mismatch: {label}')


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _layout(state):
    # Meta construction validates names/order/shapes without allocating GPT
    # weights, touching CUDA, or consuming the CPU initialization RNG stream.
    with torch.device('meta'):
        model = GPT(**state['config']['model'])
    expected = model.state_dict()
    if state['model'].keys() != expected.keys():
        raise ValueError('model keys differ from the baseline GPT')
    for name, reference in expected.items():
        value = state['model'][name]
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise ValueError(f'model shape/dtype mismatch: {name}')
        if not torch.isfinite(value).all():
            raise ValueError(f'nonfinite model tensor: {name}')
    named = list(model.named_parameters())
    adam_names = [['embed.weight'], ['proj.weight'], [n for n, p in named if p.ndim < 2]]
    muon_names = [n for n, _ in sorted(
        [(n, p) for n, p in named if n.startswith('blocks.') and p.ndim >= 2],
        key=lambda item: item[1].size(), reverse=True)]
    return adam_names, [muon_names]


def _optimizer_ids(optimizer, name_groups, label):
    if set(optimizer) != {'state', 'param_groups'} or len(optimizer['param_groups']) != len(name_groups):
        raise ValueError(f'{label}: unsupported optimizer layout')
    mapping = {}
    for group, names in zip(optimizer['param_groups'], name_groups):
        ids = group['params']
        expected_ids = list(range(len(mapping), len(mapping) + len(names)))
        if ids != expected_ids:
            raise ValueError(f'{label}: parameter IDs/order differ from make_optimizers')
        mapping.update(zip(ids, names))
    return mapping


def _validate_state(state):
    if state.get('format_version') != 1:
        raise ValueError('input must be an unmerged baseline checkpoint format 1')
    if state.get('controller') is not None or state['config'].get('mode') != 'baseline':
        raise ValueError('learned/controller checkpoints cannot be merged')
    config, cursor, step = state['config'], state['data_cursor'], state['step']
    world, rank = config['world_size'], cursor['rank']
    if world not in (1, 2, 4, 8) or not 0 <= rank < world or cursor['world_size'] != world:
        raise ValueError('invalid source rank/world size')
    if not 0 < step <= config['max_steps'] or cursor['batches'] != step:
        raise ValueError('checkpoint training/data clock mismatch (positive step required)')
    if cursor['batch_size'] != config['batch_tokens'] or cursor['seq_len'] != config['seq_len']:
        raise ValueError('data cursor batch/sequence configuration mismatch')
    divisor = config['seq_len'] * world * config['microbatch_sequences']
    if config['batch_tokens'] % divisor:
        raise ValueError('invalid microbatch accumulation configuration')
    expected = step * (config['batch_tokens'] // divisor)
    _equal(state['counters'], dict(forward=expected, backward=expected), 'compute counters')
    if len(state['optimizers']) != 2:
        raise ValueError('expected AdamW followed by Muon')
    return rank, world


def _validate_moments(optimizer, mapping, model, *, muon, expected_ids, step):
    if set(optimizer['state']) != set(expected_ids):
        raise ValueError('Muon owner state missing/extra/conflicting' if muon else 'Adam state missing/extra')
    for param_id in expected_ids:
        item = optimizer['state'][param_id]
        keys = {'momentum'} if muon else {'step', 'exp_avg', 'exp_avg_sq'}
        if set(item) != keys:
            raise ValueError(f'unsupported optimizer state keys for parameter {param_id}')
        weight = model[mapping[param_id]]
        for key in keys - {'step'}:
            value = item[key]
            if value.shape != weight.shape or value.dtype != weight.dtype or not torch.isfinite(value).all():
                raise ValueError(f'optimizer {key} shape/dtype/value mismatch: {mapping[param_id]}')
        if not muon and (not isinstance(item['step'], torch.Tensor)
                         or item['step'].numel() != 1 or item['step'].item() != step):
            raise ValueError('Adam update clock mismatch')


def merge_checkpoints(paths, output):
    """Validate all source ranks, atomically write one diagnostic artifact.

    Rank ordering comes from the saved cursor, never the filename. Paths must
    reference immutable completed checkpoints (prefer retained/stepN/rankN.pt).
    The return value is small JSON-compatible provenance, not the tensor state.
    """
    paths = [Path(p).resolve() for p in paths]
    output = Path(output).resolve()
    if not paths or len(set(paths)) != len(paths):
        raise ValueError('provide distinct checkpoint paths for every source rank')
    if output.exists() or output in paths:
        raise FileExistsError(output)
    sources, seen, merged_moments = [], set(), {}
    reference = rank_zero = None
    for path in paths:
        before = path.stat()
        state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        rank, world = _validate_state(state)
        if rank in seen:
            raise ValueError(f'duplicate source rank {rank}')
        seen.add(rank)
        if reference is None:
            reference = state
            adam_names, muon_names = _layout(state)
            adam_map = _optimizer_ids(state['optimizers'][0], adam_names, 'Adam')
            muon_map = _optimizer_ids(state['optimizers'][1], muon_names, 'Muon')
            _validate_moments(state['optimizers'][0], adam_map, state['model'], muon=False,
                              expected_ids=adam_map, step=state['step'])
        else:
            for key in ('config', 'step', 'counters', 'model'):
                _equal(reference[key], state[key], key)
            _equal({k: v for k, v in reference['data_cursor'].items() if k != 'rank'},
                   {k: v for k, v in state['data_cursor'].items() if k != 'rank'}, 'data_cursor')
            _equal(reference['optimizers'][0], state['optimizers'][0], 'replicated AdamW')
            _equal(reference['optimizers'][1]['param_groups'], state['optimizers'][1]['param_groups'],
                   'Muon parameter groups')
        owned = [pid for group in state['optimizers'][1]['param_groups']
                 for i, pid in enumerate(group['params']) if i % world == rank]
        _validate_moments(state['optimizers'][1], muon_map, state['model'], muon=True,
                          expected_ids=owned, step=state['step'])
        merged_moments.update(state['optimizers'][1]['state'])
        if rank == 0:
            rank_zero = state
        sources.append(dict(rank=rank, path=str(path), sha256=_sha256(path), bytes=before.st_size))
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError(f'source changed while reading: {path}; use retained checkpoints')
    world = reference['config']['world_size']
    if seen != set(range(world)) or set(merged_moments) != set(muon_map):
        raise ValueError(f'missing source ranks/states: expected ranks {list(range(world))}, got {sorted(seen)}')
    metadata = dict(format_version=1, purpose='frozen_controller_observation_only', resumable=False,
                    source_checkpoint_format=1, source_world_size=world, step=reference['step'],
                    sources=sorted(sources, key=lambda x: x['rank']),
                    muon_parameter_names=muon_map, muon_states=len(merged_moments),
                    replicated_fields_verified=['model', 'AdamW', 'config', 'step', 'counters', 'data_cursor_except_rank'],
                    rank_zero_fields_retained=['rng', 'elapsed_seconds', 'data_cursor.rank'])
    merged = dict(rank_zero, format_version=FORMAT, optimizer_merge=metadata,
                  optimizers=[rank_zero['optimizers'][0], dict(rank_zero['optimizers'][1], state=merged_moments)])
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=output.name + '.', suffix='.tmp', dir=output.parent)
    os.close(fd)
    try:
        torch.save(merged, temporary)
        with open(temporary, 'rb') as stream:
            os.fsync(stream.fileno())
        # Hard-link publishes atomically without replacing an existing artifact.
        os.link(temporary, output)
    finally:
        os.unlink(temporary)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input-dir', type=Path, help='retained step directory containing rankN.pt')
    source.add_argument('--inputs', nargs='+', type=Path, help='explicit completed checkpoint files, one per rank')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    paths = sorted(args.input_dir.glob('rank[0-9]*.pt')) if args.input_dir else args.inputs
    print(json.dumps(merge_checkpoints(paths, args.output), indent=2))


if __name__ == '__main__':
    main()
