"""Past-only frozen-checkpoint probes; extra GPT use is separately accounted.

Sampling is uniform over consumed (zero-based GPT step, global sequence) IDs.
Audit IDs are disjoint, not necessarily token-disjoint: adjacent sequences share
one target/input token and underlying source text may repeat. No all-input claim.
"""
from dataclasses import dataclass
from pathlib import Path
import glob
import re
import struct
import numpy as np
import torch


@dataclass
class SequenceBatch:
    inputs: torch.Tensor
    targets: torch.Tensor
    provenance: list[dict]


@dataclass
class PastReplay:
    train: SequenceBatch
    audit: SequenceBatch
    metadata: dict


def next_training_sequences(files, completed_steps, *, batch_tokens=524288, seq_len=1024, count=32, seed=600001):
    """Audit-only sample from the ordinary batch immediately after a checkpoint.

    These sequences were not consumed by that checkpoint. Callers must never use
    the result for Chisel fitting or its safety gate.
    """
    paths=sorted(Path(p) for p in files)
    if not paths or completed_steps<0 or batch_tokens%seq_len or not 0<count<=batch_tokens//seq_len:
        raise ValueError('valid prescribed batch and sample count required')
    if any(re.search(r'(?:^|[_\-.])(val(?:idation)?|test|eval)(?:[_\-.]|$)',p.name,re.I) for p in paths):
        raise ValueError('validation/test shards prohibited')
    sizes={}
    def size(i):
        if i>=len(paths):raise ValueError('not enough training shards for next-batch audit')
        if i not in sizes:
            with paths[i].open('rb') as stream:header=stream.read(1024)
            magic,version,n=struct.unpack('<iii',header[:12])
            if (magic,version)!=(20240520,1) or paths[i].stat().st_size!=1024+2*n:raise ValueError('invalid shard')
            sizes[i]=n
        return sizes[i]
    index=pos=0
    for step in range(completed_steps+1):
        if pos+batch_tokens+1>=size(index):index,pos=index+1,0
        if pos+batch_tokens+1>size(index):raise ValueError('shard too small')
        if step<completed_steps:pos+=batch_tokens
    generator=torch.Generator().manual_seed(seed)
    sequences=torch.randperm(batch_tokens//seq_len,generator=generator)[:count].tolist()
    mapped=np.memmap(paths[index],mode='r',dtype='<u2',offset=1024,shape=(size(index),))
    spans=[torch.from_numpy(np.array(mapped[pos+s*seq_len:pos+s*seq_len+seq_len+1],dtype=np.int64,copy=True)) for s in sequences]
    tokens=torch.stack(spans)
    return SequenceBatch(tokens[:,:-1].int().contiguous(),tokens[:,1:].contiguous(),
        [dict(id=completed_steps*(batch_tokens//seq_len)+s,step=completed_steps,sequence=s,role='fresh_next_training_audit') for s in sequences])


def sample_past_sequences(files, completed_steps, *, batch_tokens=524288, seq_len=1024,
                          seed=400001, count=4096, audit_fraction=.25, data_manifest=None):
    """Read headers and selected seq_len+1 uint16 spans, never whole data shards.

    ``files`` is a glob or file list, sorted exactly like FinewebStream. Optional
    caller-supplied manifest is recorded without claiming hashes were verified.
    ``count`` includes train and audit; insufficient past data is an error.
    """
    paths = sorted(Path(p) for p in (glob.glob(str(files)) if isinstance(files, (str, Path)) else files))
    if not paths or len(set(paths)) != len(paths):
        raise ValueError('provide a nonempty unique training shard list')
    if any(re.search(r'(?:^|[_\-.])(val(?:idation)?|test|eval)(?:[_\-.]|$)', p.name, re.I) for p in paths):
        raise ValueError('validation/test/evaluation shards are prohibited')
    if completed_steps < 1 or batch_tokens < 1 or seq_len < 1 or batch_tokens % seq_len:
        raise ValueError('positive completed steps and whole-sequence batches required')
    total = completed_steps * (batch_tokens // seq_len)
    if not 1 < count <= total or not 0 < audit_fraction < 1:
        raise ValueError('need 2..available past sequences and a nonempty train/audit split')
    headers, schedule = {}, []
    def size(index):
        if index >= len(paths):
            raise ValueError('not enough training shards for the completed prefix')
        if index not in headers:
            with paths[index].open('rb') as f:
                header = f.read(1024)
            if len(header) != 1024:
                raise ValueError('truncated FineWeb header')
            magic, version, tokens = struct.unpack('<iii', header[:12])
            if (magic, version) != (20240520, 1) or tokens < 0 or paths[index].stat().st_size != 1024+2*tokens:
                raise ValueError('invalid FineWeb header or size')
            headers[index] = tokens
        return headers[index]
    index = pos = 0
    for _ in range(completed_steps):
        if pos + batch_tokens + 1 >= size(index):  # exactly one advance, like FinewebStream
            index, pos = index+1, 0
        if pos + batch_tokens + 1 > size(index):
            raise ValueError('shard too small for prescribed batch')
        schedule.append((index, pos))
        pos += batch_tokens
    generator = torch.Generator(device='cpu').manual_seed(seed)
    ids = torch.randperm(total, generator=generator)[:count].tolist()
    inputs = torch.empty(count, seq_len, dtype=torch.int32)
    targets = torch.empty(count, seq_len, dtype=torch.int64)
    provenance, maps = [], {}
    for row, sample_id in enumerate(ids):
        step, sequence = divmod(sample_id, batch_tokens // seq_len)
        shard, batch_offset = schedule[step]
        offset = batch_offset + sequence*seq_len
        if shard not in maps:
            maps[shard] = np.memmap(paths[shard], mode='r', dtype='<u2', offset=1024, shape=(headers[shard],))
        span = np.array(maps[shard][offset:offset+seq_len+1], dtype=np.int64, copy=True)
        inputs[row] = torch.from_numpy(span[:-1]).int()
        targets[row] = torch.from_numpy(span[1:])
        provenance.append(dict(id=sample_id, step=step, sequence=sequence,
                               shard=str(paths[shard]), shard_index=shard, offset=offset))
    audit_count = min(count-1, max(1, int(count*audit_fraction)))
    cut = count-audit_count
    return PastReplay(SequenceBatch(inputs[:cut], targets[:cut], provenance[:cut]),
                      SequenceBatch(inputs[cut:], targets[cut:], provenance[cut:]),
                      dict(kind='previously_consumed_training_sequences', completed_steps=completed_steps,
                           batch_tokens=batch_tokens, seq_len=seq_len, seed=seed,
                           sample_count=count, audit_fraction=audit_fraction,
                           sampling='uniform_without_replacement_by_global_sequence_id',
                           data_manifest=data_manifest, manifest_verified_by_sampler=False))


def synthetic_sequences(kind, count, seq_len, vocab_size, *, seed=500001, past=None):
    """Independent synthetic probes, not real-data replay or all-input coverage.

    Empirical-unigram probabilities use only the training split of PastReplay,
    never the withheld audit split, future tokens, or a validation corpus.
    """
    if count < 1 or seq_len < 1 or vocab_size < 1:
        raise ValueError('positive count, sequence length and vocabulary required')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    shape = (count, seq_len+1)
    if kind == 'uniform':
        tokens = torch.randint(vocab_size, shape, generator=generator)
    elif kind == 'repeated':
        tokens = torch.randint(vocab_size, (count,1), generator=generator).expand(shape).clone()
    elif kind == 'alternating':
        if vocab_size < 2:
            raise ValueError('alternating requires at least two distinct tokens')
        first = torch.randint(vocab_size, (count,1), generator=generator)
        second = (first+torch.randint(1,vocab_size,(count,1),generator=generator)) % vocab_size
        tokens = torch.where(torch.arange(seq_len+1)[None,:] % 2 == 0, first, second)
    elif kind == 'empirical_unigram':
        if not isinstance(past, PastReplay) or not past.train.inputs.numel():
            raise ValueError('empirical unigram requires a nonempty past-only replay buffer')
        source = past.train.inputs.detach().cpu().long().flatten()
        if source.min() < 0 or source.max() >= vocab_size:
            raise ValueError('past token outside requested vocabulary')
        counts = torch.bincount(source, minlength=vocab_size).double()
        tokens = torch.multinomial(counts, count*(seq_len+1), replacement=True, generator=generator).view(shape)
    else:
        raise ValueError(f'unknown synthetic probe kind: {kind}')
    provenance = [dict(kind='synthetic', distribution=kind, seed=seed, sequence=i,
                       all_input_coverage=False) for i in range(count)]
    return SequenceBatch(tokens[:,:-1].to(torch.int32).contiguous(), tokens[:,1:].long().contiguous(), provenance)
