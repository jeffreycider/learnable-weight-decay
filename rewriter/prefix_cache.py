"""Lossless, ordered observations from a fixed-seed identity GPT prefix.

The cache contains actor observations and the four *training* preservation
probes, never controller weights, controller Adam state, audit probes, data
batches, or raw GPT activations (aligned projected trajectories are included).
Fresh controllers may replay ordinary auxiliary
updates on these summaries; replay cannot provide task credit for unexecuted
actions and performs no GPT evaluation.

Matrix feature constants are stored once; anchor feature columns reuse the
separately stored anchors only after byte equality is checked. Original weight
probes remain explicit: multiplying normalized probes by their scales would not
be bit-exact. Vector constants are folded similarly. Unrecognized layouts use
a lossless dense fallback. Optional zlib compression is standard-library only.

For PR323/r8, ideal layout compaction gives approximately 88.2 MiB per step
before zlib, versus approximately 119 MiB expanded. The first 1000 records thus
need roughly 86 GiB before compression. These are layout estimates, not measured
compression ratios; writer statistics report actual sizes and times. Memory is
bounded by max_pending CPU records plus one serialization/compression buffer.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import time
import zlib

import torch

from .algebra import Anchors, Measurements
from .controller import ControllerSnapshot, MatrixObservation, VectorObservation


SCHEMA = 1
MAGIC = b'LWDPC01\x00'
HEADER = struct.Struct('<8sQ')


def _copy(tensor):
    return tensor.detach().to(device='cpu', copy=True).contiguous()


def _same_bits(left, right):
    return (left.shape == right.shape and left.dtype == right.dtype
            and torch.equal(left.contiguous().reshape(-1).view(torch.uint8),
                            right.contiguous().reshape(-1).view(torch.uint8)))


def _restore_layout(tensor, stride, device):
    """Preserve logical values and strides, including QR column-major anchors.

    Storage offsets are reset to zero; backing-storage padding is zero-filled.
    Preserving layout avoids gratuitous changes to GEMM transpose/stride paths.
    It does not guarantee identical kernels across different hardware stacks.
    """
    stride = tuple(stride)
    shape = tuple(tensor.shape)
    if len(stride) != len(shape) or any(s < 0 for s in stride):
        raise ValueError('invalid cached tensor layout')
    size = 0 if any(d == 0 for d in shape) else 1 + sum((d-1)*s for d, s in zip(shape, stride))
    storage = torch.zeros(size, dtype=tensor.dtype, device=device)
    target = storage.as_strided(shape, stride)
    # Broadcast layouts can be restored by writing their unique slice once.
    source, destination = tensor, target
    for axis, (dimension, step) in enumerate(zip(shape, stride)):
        if dimension > 1 and step == 0:
            source = source.narrow(axis, 0, 1)
            destination = destination.narrow(axis, 0, 1)
    destination.copy_(source)
    return target


def _pack_matrix_features(features, anchor):
    rank = anchor.shape[1]
    if (features.ndim == 2 and features.shape == (anchor.shape[0], 5 * rank + 15)
            and _same_bits(features[:, 4*rank:5*rank], anchor)
            and _same_bits(features[:, 5*rank+1:], features[:1, 5*rank+1:].expand(features.shape[0], -1))):
        return dict(encoding='matrix_v1', banks=_copy(features[:, :4*rank]),
                    activity=_copy(features[:, 5*rank:5*rank+1]),
                    constants=_copy(features[0, 5*rank+1:]))
    return dict(encoding='dense', tensor=_copy(features))


def _unpack_matrix_features(packed, anchor):
    if packed['encoding'] == 'dense':
        return packed['tensor']
    if packed['encoding'] != 'matrix_v1':
        raise ValueError('unknown matrix feature encoding')
    return torch.cat((packed['banks'], anchor, packed['activity'],
                      packed['constants'][None].expand(anchor.shape[0], -1)), dim=1)


def _pack_vector_features(features):
    dynamic = [0, 2, 4, 6]
    constants = [1, 3, 5, 7] + list(range(8, 18))
    if (features.ndim == 2 and features.shape[1] == 18
            and _same_bits(features[:, constants], features[:1, constants].expand(features.shape[0], -1))):
        return dict(encoding='vector_v1', values=_copy(features[:, dynamic]),
                    constants=_copy(features[0, constants]))
    return dict(encoding='dense', tensor=_copy(features))


def _unpack_vector_features(packed):
    if packed['encoding'] == 'dense':
        return packed['tensor']
    if packed['encoding'] != 'vector_v1':
        raise ValueError('unknown vector feature encoding')
    values, constants = packed['values'], packed['constants']
    result = values.new_empty(values.shape[0], 18)
    result[:, [0, 2, 4, 6]] = values
    result[:, [1, 3, 5, 7] + list(range(8, 18))] = constants[None]
    return result


def _tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(v) for v in value)
    return 0


def pack_record(step, snapshot, train, *, train_probe_count=4):
    """Freeze an independent CPU record; caller may then release live buffers."""
    if train_probe_count not in (0,4):
        raise ValueError('supported cached training-probe counts are zero or four')
    if not isinstance(snapshot, ControllerSnapshot) or not isinstance(train, Measurements):
        raise TypeError('prefix records require a ControllerSnapshot and training Measurements')
    if set(snapshot.matrices) != set(train.matrices) or set(snapshot.vectors) != set(train.vectors):
        raise ValueError('training measurements must cover every actor parameter exactly once')
    matrices, vectors = {}, {}
    expanded = snapshot.trajectories.numel() * snapshot.trajectories.element_size()
    expanded += _tensor_bytes([snapshot.weight_norm, snapshot.radius])
    for name, obs in snapshot.matrices.items():
        a, b = obs.anchors.A, obs.anchors.B
        gb, gta = train.matrices[name]
        if tuple(gb.shape) != (train_probe_count, *a.shape) or tuple(gta.shape) != (train_probe_count, *b.shape):
            raise ValueError(f'cache requires exactly {train_probe_count} training sign probes')
        matrices[name] = dict(anchors=(_copy(a), _copy(b)),
            rows=_pack_matrix_features(obs.rows, a), columns=_pack_matrix_features(obs.columns, b),
            weight_probes=tuple(_copy(x) for x in obs.weight_probes),
            train=(_copy(gb), _copy(gta)),
            strides=dict(anchors=(a.stride(), b.stride()), rows=obs.rows.stride(), columns=obs.columns.stride(),
                         weight_probes=tuple(t.stride() for t in obs.weight_probes), train=(gb.stride(), gta.stride())))
        expanded += _tensor_bytes([a, b, obs.rows, obs.columns, obs.weight_probes, gb, gta])
    for name, obs in snapshot.vectors.items():
        measurement = train.vectors[name]
        if tuple(measurement.shape) != (train_probe_count, *obs.weight.shape):
            raise ValueError(f'cache requires exactly {train_probe_count} training vector probes')
        vectors[name] = dict(coordinates=_pack_vector_features(obs.coordinates),
                             weight=_copy(obs.weight), train=_copy(measurement),
                             strides=dict(coordinates=obs.coordinates.stride(), weight=obs.weight.stride(),
                                          train=measurement.stride()))
        expanded += _tensor_bytes([obs.coordinates, obs.weight, measurement])
    record = dict(schema=SCHEMA, step=int(step), identity_action=True, train_probe_count=train_probe_count,
                  loss_units='sequence mean CE in nats/token', matrices=matrices, vectors=vectors,
                  trajectories=_copy(snapshot.trajectories), weight_norm=_copy(snapshot.weight_norm),
                  radius=_copy(snapshot.radius), expanded_tensor_bytes=expanded,
                  strides=dict(trajectories=snapshot.trajectories.stride(),
                               weight_norm=snapshot.weight_norm.stride(), radius=snapshot.radius.stride()))
    if getattr(snapshot,'trajectory_mask',None) is not None:
        record['trajectory_mask']=_copy(snapshot.trajectory_mask)
        record['strides']['trajectory_mask']=snapshot.trajectory_mask.stride()
        record['expanded_tensor_bytes']+=_tensor_bytes(snapshot.trajectory_mask)
    record['packed_tensor_bytes'] = _tensor_bytes(record)
    return record


@dataclass(frozen=True)
class PrefixRecord:
    step: int
    snapshot: ControllerSnapshot
    train: Measurements


def unpack_record(record, device='cpu'):
    if record['schema'] != SCHEMA or record.get('identity_action') is not True:
        raise ValueError('unsupported or nonidentity prefix record')
    if record.get('loss_units') != 'sequence mean CE in nats/token':
        raise ValueError('unsupported prefix measurement units')
    def move(t, stride):
        return _restore_layout(t, stride, device)
    matrices, vectors, train_m, train_v = {}, {}, {}, {}
    for name, obs in record['matrices'].items():
        a, b = obs['anchors']
        strides = obs['strides']
        matrices[name] = MatrixObservation(Anchors(move(a, strides['anchors'][0]), move(b, strides['anchors'][1])),
            move(_unpack_matrix_features(obs['rows'], a), strides['rows']),
            move(_unpack_matrix_features(obs['columns'], b), strides['columns']),
            tuple(move(t, s) for t, s in zip(obs['weight_probes'], strides['weight_probes'])))
        train_m[name] = tuple(move(t, s) for t, s in zip(obs['train'], strides['train']))
    for name, obs in record['vectors'].items():
        strides = obs['strides']
        vectors[name] = VectorObservation(move(_unpack_vector_features(obs['coordinates']), strides['coordinates']),
                                         move(obs['weight'], strides['weight']))
        train_v[name] = move(obs['train'], strides['train'])
    snapshot = ControllerSnapshot(matrices, vectors, move(record['trajectories'], record['strides']['trajectories']),
                                  move(record['weight_norm'], record['strides']['weight_norm']),
                                  move(record['radius'], record['strides']['radius']),
                                  trajectory_mask=move(record['trajectory_mask'],record['strides']['trajectory_mask']) if 'trajectory_mask' in record else None)
    return PrefixRecord(record['step'], snapshot, Measurements(train_m, train_v))


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read_index(directory):
    path = directory / 'index.jsonl'
    if not path.exists():
        return []
    with path.open() as stream:
        entries = [json.loads(line) for line in stream if line.strip()]
    for step, entry in enumerate(entries):
        if entry['step'] != step or entry['file'] != f'step_{step:04d}.lwdpc':
            raise ValueError('prefix cache index is not a contiguous ordered prefix')
    return entries


class PrefixCacheWriter:
    """Bounded asynchronous compression, atomic files, and durable ordered index.

    CPU freezing occurs synchronously in append, before the caller can reuse
    sensor storage. Compression/file I/O run on one worker thread. A full queue
    blocks instead of accumulating unbounded host memory. Files are written
    only beneath the caller's requested directory.

    metadata should pin the GPT seed, model/data order, optimizer/schedule,
    sensor seeds/rank, and code revision. Reuse is valid only for that same
    unchanged GPT prefix; metadata is compared exactly when resuming writes.
    Set metadata['editable_scope']='hidden' for Chisel: scope determines cached
    norm/radius values even though the underlying identity GPT prefix is shared.
    Missing scope means historical 'all', and cross-scope replay is rejected.
    """
    def __init__(self, directory, metadata, expected_steps=1000, *, resume=False,
                 max_pending=1, compression_level=1):
        if expected_steps < 1 or max_pending < 1 or not 0 <= compression_level <= 9:
            raise ValueError('invalid prefix length, queue limit, or compression level')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest = dict(schema=SCHEMA, identity_only=True, expected_steps=int(expected_steps),
                             compression='zlib', compression_level=compression_level, metadata=metadata)
        # JSON metadata cannot smuggle tensor/controller state into the cache.
        self.manifest = json.loads(json.dumps(self.manifest, sort_keys=True, allow_nan=False))
        manifest_path = self.directory / 'manifest.json'
        if manifest_path.exists():
            if not resume or json.loads(manifest_path.read_text()) != self.manifest:
                raise ValueError('existing prefix cache requires resume=True and identical metadata')
        else:
            if any(self.directory.iterdir()):
                raise ValueError('refusing to initialize a cache in a nonempty unrelated directory')
            _atomic_json(manifest_path, self.manifest)
        self.entries = _read_index(self.directory)
        if len(self.entries) > expected_steps:
            raise ValueError('cache index exceeds declared prefix length')
        self.next_step = len(self.entries)
        self.max_pending, self.level = max_pending, compression_level
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='prefix-cache')
        self.pending, self.closed = [], False

    def append(self, step, snapshot, train, *, identity_action):
        if self.closed:
            raise RuntimeError('prefix cache writer is closed')
        if identity_action is not True:
            raise ValueError('only observations from actually executed identity actions may be cached')
        if step != self.next_step or step >= self.manifest['expected_steps']:
            raise ValueError('prefix records must be appended once, in consecutive order from step zero')
        while len(self.pending) >= self.max_pending:
            self.pending.pop(0).result()
        started = time.perf_counter()
        record = pack_record(step, snapshot, train)
        freeze_seconds = time.perf_counter() - started
        self.pending.append(self.executor.submit(self._write, record, freeze_seconds))
        self.next_step += 1

    def _write(self, record, freeze_seconds):
        if record['step'] != len(self.entries):
            raise RuntimeError('cannot write beyond an uncommitted or failed prefix record')
        started = time.perf_counter()
        raw = io.BytesIO()
        torch.save(record, raw)
        raw_view = raw.getbuffer()
        compressed = zlib.compress(raw_view, level=self.level)
        blob = HEADER.pack(MAGIC, len(raw_view)) + compressed
        path = self.directory / f"step_{record['step']:04d}.lwdpc"
        temporary = path.with_suffix(path.suffix + '.tmp')
        with temporary.open('wb') as stream:
            stream.write(blob)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        entry = dict(step=record['step'], file=path.name, sha256=hashlib.sha256(blob).hexdigest(),
                     file_bytes=len(blob), serialized_bytes=len(raw_view),
                     expanded_tensor_bytes=record['expanded_tensor_bytes'],
                     packed_tensor_bytes=record['packed_tensor_bytes'],
                     freeze_seconds=freeze_seconds, write_seconds=time.perf_counter()-started)
        with (self.directory / 'index.jsonl').open('a') as stream:
            stream.write(json.dumps(entry, sort_keys=True) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        self.entries.append(entry)
        return entry

    def flush(self):
        while self.pending:
            self.pending.pop(0).result()
        return list(self.entries)

    def close(self):
        if self.closed:
            return list(self.entries)
        try:
            self.flush()
            if len(self.entries) == self.manifest['expected_steps']:
                index_hash = hashlib.sha256((self.directory / 'index.jsonl').read_bytes()).hexdigest()
                _atomic_json(self.directory / 'complete.json', dict(steps=len(self.entries), index_sha256=index_hash))
        finally:
            self.executor.shutdown(wait=True)
            self.closed = True
        return list(self.entries)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class PrefixCacheReader:
    def __init__(self, directory, *, require_complete=True, expected_metadata=None):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / 'manifest.json').read_text())
        if self.manifest['schema'] != SCHEMA or self.manifest.get('identity_only') is not True:
            raise ValueError('unsupported or nonidentity prefix cache')
        if self.manifest.get('compression') != 'zlib':
            raise ValueError('unsupported prefix compression')
        if expected_metadata is not None and self.manifest['metadata'] != json.loads(json.dumps(expected_metadata)):
            raise ValueError('prefix metadata differs from the requested GPT/sensor trajectory')
        self.entries = _read_index(self.directory)
        if require_complete:
            complete_path = self.directory / 'complete.json'
            if not complete_path.exists() or len(self.entries) != self.manifest['expected_steps']:
                raise ValueError('prefix cache is incomplete')
            complete = json.loads(complete_path.read_text())
            digest = hashlib.sha256((self.directory / 'index.jsonl').read_bytes()).hexdigest()
            if complete != dict(steps=len(self.entries), index_sha256=digest):
                raise ValueError('prefix completion/index integrity check failed')

    def __len__(self):
        return len(self.entries)

    def read(self, step, device='cpu'):
        if not 0 <= step < len(self.entries):
            raise IndexError('prefix step is outside the committed record range')
        entry = self.entries[step]
        blob = (self.directory / entry['file']).read_bytes()
        if hashlib.sha256(blob).hexdigest() != entry['sha256']:
            raise ValueError('prefix record SHA256 mismatch')
        magic, raw_length = HEADER.unpack(blob[:HEADER.size])
        if magic != MAGIC:
            raise ValueError('invalid prefix file header')
        raw = zlib.decompress(blob[HEADER.size:])
        if len(raw) != raw_length or len(raw) != entry['serialized_bytes']:
            raise ValueError('prefix record length mismatch')
        # Format contains primitives/tensors only, not arbitrary Python classes.
        packed = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        if packed['step'] != step:
            raise ValueError('prefix file timestep disagrees with its index')
        return unpack_record(packed, device=device)

    def iter_records(self, device='cpu', prefetch=1):
        """Stream in order with at most one or two prefetched *CPU* records."""
        if prefetch not in (0, 1, 2):
            raise ValueError('prefix prefetch must be 0, 1, or 2 records')
        if not prefetch:
            for step in range(len(self)):
                yield self.read(step, device=device)
            return
        def move(record):
            if torch.device(device).type == 'cpu':
                return record
            def transfer(t):
                return _restore_layout(t, t.stride(), device)
            matrices = {name: MatrixObservation(
                Anchors(transfer(obs.anchors.A), transfer(obs.anchors.B)), transfer(obs.rows),
                transfer(obs.columns), tuple(transfer(t) for t in obs.weight_probes))
                for name, obs in record.snapshot.matrices.items()}
            vectors = {name: VectorObservation(transfer(obs.coordinates), transfer(obs.weight))
                       for name, obs in record.snapshot.vectors.items()}
            snapshot = ControllerSnapshot(matrices, vectors, transfer(record.snapshot.trajectories),
                                          transfer(record.snapshot.weight_norm), transfer(record.snapshot.radius),
                                          trajectory_mask=transfer(record.snapshot.trajectory_mask) if record.snapshot.trajectory_mask is not None else None)
            train = Measurements({n: tuple(transfer(t) for t in values) for n, values in record.train.matrices.items()},
                                 {n: transfer(t) for n, t in record.train.vectors.items()})
            return PrefixRecord(record.step, snapshot, train)
        # GPU transfer remains on the caller's thread/current stream. Background
        # I/O cannot race controller kernels or retain GPU observation buffers.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='prefix-read') as executor:
            futures = []
            next_step = 0
            while next_step < min(prefetch, len(self)):
                futures.append(executor.submit(self.read, next_step, 'cpu'))
                next_step += 1
            while futures:
                record = futures.pop(0).result()
                if next_step < len(self):
                    futures.append(executor.submit(self.read, next_step, 'cpu'))
                    next_step += 1
                yield move(record)


def replay_identity_prefix(online, reader, *, max_steps, device='cpu', progress=None):
    """Replay cached auxiliary learning once into a fresh controller/Adam state.

    The caller must randomly initialize the controller for this run. Nothing
    about a completed run's learned controller is transferred by this helper.
    """
    if (online.pending is not None or online.updates or online.aux_evaluations
            or online.optimizer.state or getattr(online, '_prefix_replayed', False)):
        raise ValueError('prefix replay requires a fresh controller optimizer and no pending credit')
    if len(reader) != reader.manifest['expected_steps']:
        raise ValueError('controller replay requires the complete declared identity prefix')
    if not len(reader) < max_steps:
        raise ValueError('prefix must end before the declared terminal GPT iteration')
    if online.config.get('warmup_steps', 64) < len(reader):
        raise ValueError('all cached prefix decisions must lie within declared identity warmup')
    if reader.manifest['metadata'].get('editable_scope','all')!=online.config.get('editable_scope','all'):
        raise ValueError('cached norm/radius editable scope differs from controller replay')
    online._prefix_replayed = True
    for record in reader.iter_records(device=device):
        # shadow=True guarantees identity; audit=None is deliberately unavailable.
        result = online.propose(record.snapshot, record.train, None, record.step, max_steps, shadow=True)
        if result is not None or online.pending is not None:
            raise RuntimeError('identity prefix unexpectedly created an executable action')
        if progress is not None:
            progress(record.step, dict(online.metrics))
    return dict(cached_gpt_iterations=len(reader), extra_gpt_forwards=0, extra_gpt_backwards=0,
                controller_updates=online.updates, auxiliary_evaluations=online.aux_evaluations)
