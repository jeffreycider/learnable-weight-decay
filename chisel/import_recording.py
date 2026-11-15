"""Lossless migration of sharded PR267 text recordings into the shared replay contract."""
from pathlib import Path
import copy
import hashlib
import json
import tempfile
import torch
import yaml
from chisel.capture import TrajectoryWriter
from chisel.harness import RunState, build, resolve
from chisel.storage import sha256
from chisel.training import optimizers


def read_shard(path):
    import zstandard
    path = Path(path)
    receipt = json.loads(path.with_suffix('.ready.json').read_text())
    if path.stat().st_size != receipt['bytes'] or sha256(path) != receipt['sha256']:
        raise ValueError(f'corrupt original shard: {path}')
    with path.open('rb') as source, tempfile.TemporaryFile() as destination:
        zstandard.ZstdDecompressor().copy_stream(source, destination)
        destination.seek(0)
        return torch.load(destination, map_location='cpu', weights_only=False), receipt


def merge(shards, template):
    first = shards[0]
    if len(shards) != 8 or [s['rank'] for s in shards] != list(range(8)):
        raise ValueError('import requires all eight original ranks')
    for shard in shards:
        if shard['schema'] != 'chisel-muonh-gpt-transition-v1':
            raise ValueError('only ordinary training transitions have task gradients')
        for key in ('seed', 'iteration', 'world_size', 'loss_units', 'consolidated_weights'):
            if shard[key] != first[key]:
                raise ValueError(f'inconsistent original {key}')
    if first['loss_units'] != 'global_token_sum' or first['consolidated_weights'] != 'implicit zero':
        raise ValueError('unsupported gradient units or nonzero consolidation')
    config = copy.deepcopy(template)
    config['seeds']['initialization'] = first['seed']
    model = build(config['model'])
    named = dict(model.named_parameters())
    banks = {}
    for bank in ('weights', 'gradient', 'optimizer_state'):
        merged = {}
        for shard in shards:
            if merged.keys() & shard[bank].keys():
                raise ValueError(f'duplicate parameter in {bank}')
            merged.update(shard[bank])
        if merged.keys() != named.keys():
            raise ValueError(f'missing or unexpected parameter in {bank}')
        banks[bank] = merged
    weights = banks['weights'] | first['buffers']
    model.load_state_dict(weights, strict=True)
    state = RunState(model=model, extras=dict(device=torch.device('cpu')))
    optimizers()(config, state)
    names = {id(value): name for name, value in named.items()}
    original_groups = {name: group for optimizer in first['optimizer_groups'] for group in optimizer['groups'] for name in group['params']}
    for optimizer in state.optimizer.groups.values():
        for group in optimizer.param_groups:
            source = original_groups[names[id(group['params'][0])]]
            for parameter in group['params']:
                recorded = original_groups[names[id(parameter)]]
                for key in ('lr', 'mu', 'betas', 'eps', 'weight_decay'):
                    if key in group and key in recorded and recorded[key] != source[key]:
                        raise ValueError(f'cannot merge heterogeneous {key}')
                optimizer.state[parameter] = banks['optimizer_state'][names[id(parameter)]]
            for key in ('lr', 'mu', 'betas', 'eps', 'weight_decay', 'initial_lr'):
                if key in source:
                    group[key] = source[key]
    activity = {}
    keys = set(first['observations'])
    if any(set(shard['observations']) != keys for shard in shards):
        raise ValueError('inconsistent observation banks')
    for key in keys:
        if key.startswith('input_rms/'):
            name = key.removeprefix('input_rms/') + '.weight'
            activity[name] = dict(input=torch.stack([s['observations'][key].square() for s in shards]).mean(0).sqrt())
    current = dict(first['data_cursor'])
    manifest = config['data'].get('identity_manifest')
    current['manifest_sha256'] = hashlib.sha256(Path(manifest).read_bytes()).hexdigest() if manifest else None
    next_data = current | dict(pos=current['pos'] + current['batch_size'], batches=current['batches'] + 1)
    sources = ['baseline/model.py', 'baseline/data.py', 'chisel/models.py', 'chisel/muonh.py', 'chisel/import_recording.py']
    return dict(schema='chisel-muonh-transition-v2', config=config, step=first['iteration'],
                phase='after_backward_before_optimizer', activation_shards=0, gradient_units='sum',
                source_hashes={name: sha256(Path(name)) for name in sources},
                weights=weights, gradients=banks['gradient'], optimizer=state.optimizer.state_dict(), activity=activity,
                current=dict(data=current, torch_rng=first['torch_rng'], cuda_rng=[s['cuda_rng'] for s in shards]),
                next_data=next_data, has_next_training_step=first['iteration'] + 1 < config['train_steps'],
                raw_radii={name: p.norm() for name, p in banks['weights'].items() if name.startswith('blocks.') and p.ndim == 2},
                legacy_samples=[dict(rank=s['rank'], positions=s['observed_global_token_positions'], observations=s['observations']) for s in shards],
                import_provenance=dict(original_schema=first['schema'], original_baseline='PR267@99604285f4da6afc43b5b49f55a4548ebf5318ef',
                                       source_hashes_mean='qualified replay implementation, not original producer',
                                       unavailable=['output-adjoint RMS', 'raw linear input/adjoint samples', 'Python/NumPy RNG']))


def run(spec):
    config = resolve(spec['template'])
    config['data']['filename_pattern'] = spec['data_pattern']
    config['output_dir'] = spec['output_dir']
    root = Path(spec['output_dir'])
    root.mkdir(parents=True, exist_ok=True)
    stream = build(config['data'], device='cpu', rank=0, world_size=1)
    writer = TrajectoryWriter(root / 'trajectory')
    try:
        for step in spec['steps']:
            target = root / 'trajectory' / f'{step:06d}' / 'state.pt'
            if target.with_suffix('.ready.json').exists():
                continue
            shards, receipts = [], []
            for rank in range(8):
                path = Path(spec['source']) / f'rank{rank}' / f'step_{step:06d}.pt.zst'
                shard, receipt = read_shard(path)
                shards.append(shard)
                receipts.append(dict(path=str(path), sha256=receipt['sha256']))
            record = merge(shards, config)
            cursor = record['current']['data'] | dict(rank=0, world_size=1)
            cached = stream.tokens if stream.file_index == cursor['file_index'] else None
            stream.load_state_dict(cursor)
            stream.tokens = cached
            inputs, targets = next(stream)
            if not torch.equal(inputs, torch.cat([s['inputs'] for s in shards])) or not torch.equal(targets, torch.cat([s['targets'] for s in shards])):
                raise ValueError('imported cursor does not reproduce recorded token batch')
            record['import_provenance']['shards'] = receipts
            (root / 'hyperparameters.yaml').write_text(yaml.safe_dump(record['config'], sort_keys=False))
            (root / 'import.yaml').write_text(yaml.safe_dump(spec, sort_keys=False))
            writer.write(step, record)
            print(json.dumps(dict(imported_step=step, seed=shards[0]['seed'])), flush=True)
    finally:
        writer.close()


if __name__ == '__main__':
    import sys
    torch.set_num_threads(2)
    run(yaml.safe_load(Path(sys.argv[1]).read_text()))
