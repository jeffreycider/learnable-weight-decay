"""Balanced checkpoint sampling and replay of the recorded ordinary base proposal."""
from pathlib import Path
import hashlib
import torch
import yaml
from chisel.harness import RunState, build
from chisel.training import optimizers
from chisel.transfer import observe
from chisel.checkpoint_io import CheckpointReader, map_tensors


class Catalog:
    def __init__(self, manifest, batch_counts, seed):
        path = Path(manifest).resolve()
        self.identity = hashlib.sha256(path.read_bytes()).hexdigest()
        entries = yaml.safe_load(path.read_text())['states']
        self.domains = {}
        for entry in entries:
            item = dict(entry, path=str((path.parent / entry['path']).resolve()))
            self.domains.setdefault(entry['domain'], []).append(item)
        if set(batch_counts) != set(self.domains) or any(value <= 0 for value in batch_counts.values()):
            raise ValueError('batch_counts must specify each manifest domain positively')
        if len({entry['sha256'] for entry in entries}) != len(entries):
            raise ValueError('duplicate checkpoint identity in manifest')
        self.batch_counts = batch_counts
        self.rng = torch.Generator().manual_seed(seed)
        self.orders, self.cursors = {}, {}

    def __next__(self):
        batch = []
        for domain, count in self.batch_counts.items():
            for _ in range(count):
                if domain not in self.orders or self.cursors[domain] == len(self.orders[domain]):
                    self.orders[domain] = torch.randperm(len(self.domains[domain]), generator=self.rng).tolist()
                    self.cursors[domain] = 0
                batch.append(self.domains[domain][self.orders[domain][self.cursors[domain]]])
                self.cursors[domain] += 1
        return batch

    def state_dict(self):
        return dict(identity=self.identity, rng=self.rng.get_state(), orders={key: list(value) for key, value in self.orders.items()}, cursors=dict(self.cursors))

    def load_state_dict(self, saved):
        if saved['identity'] != self.identity:
            raise ValueError('offline manifest changed')
        self.rng.set_state(saved['rng'])
        self.orders = {key: list(value) for key, value in saved['orders'].items()}
        self.cursors = dict(saved['cursors'])


class Replay:
    def __init__(self, device, reader=None):
        self.device, self.models, self.streams = device, {}, {}
        self.reader = reader or CheckpointReader()

    def load(self, entry, record=None):
        record = self.reader.read(entry) if record is None else record
        record = map_tensors(record, lambda value: value.to(self.device, non_blocking=True))
        if record['schema'] not in ('chisel-muonh-transition-v1', 'chisel-muonh-transition-v2'):
            raise ValueError('unsupported transition schema')
        for source, digest in record['source_hashes'].items():
            if hashlib.sha256(Path(source).read_bytes()).hexdigest() != digest:
                raise ValueError(f'captured base implementation differs: {source}')
        config = record['config']
        key = yaml.safe_dump(config['model'])
        if key not in self.models:
            with torch.random.fork_rng(devices=[]):
                self.models[key] = build(config['model']).to(self.device)
        model = self.models[key]
        model.load_state_dict(record['weights'])
        for name, parameter in model.named_parameters():
            parameter.grad = record['gradients'][name].to(parameter)
        state = RunState(model=model, extras=dict(device=self.device))
        optimizers()(config, state)
        state.optimizer.load_state_dict(record['optimizer'])
        hidden = state.optimizer.groups['hidden']
        hidden.prepare()
        selected = {id(p) for group in hidden.param_groups for p in group['params']}
        parameters = {name: p for name, p in model.named_parameters() if id(p) in selected}
        raw = {name: p.detach().clone() for name, p in parameters.items()}
        consolidated = {name: torch.zeros_like(p) for name, p in parameters.items()}
        snapshot, directions, rates = observe(parameters, raw, consolidated, hidden, record['activity'])
        state.optimizer.step()
        base = {name: p.detach().clone() for name, p in model.named_parameters()}
        stream_key = yaml.safe_dump(config['data'])
        if stream_key not in self.streams:
            self.streams[stream_key] = build(config['data'], device=self.device, rank=0, world_size=1)
        stream = self.streams[stream_key]
        cursor = dict(record['next_data'])
        if 'world_size' in cursor:
            cursor.update(rank=0, world_size=1)  # Replay the entire global text batch.
        if hasattr(stream, 'replay'):
            batch = stream.replay(cursor['step'])
        else:
            cached = getattr(stream, 'tokens', None) if getattr(stream, 'file_index', None) == cursor.get('file_index') else None
            stream.load_state_dict(cursor)
            if hasattr(stream, 'tokens'):
                stream.tokens = cached
            batch = next(stream)
        if isinstance(batch, tuple):
            batch = dict(inputs=batch[0], targets=batch[1])
        return model, base, snapshot, directions, rates, batch, config
