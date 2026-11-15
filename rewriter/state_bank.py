"""Actor-free baseline-state bank for finite preservation across training states.

A record is the live model after completed update s, before pass s (zero based),
plus sensors from ordinary pass s-1 and its completed optimizer proposal. No
future batch, validation feedback, controller state, or dense optimizer state
belongs here. Trainer hooks and distributed momentum extraction are separate.
Only rank zero writes replicated weights; all sensor reductions must finish first.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
import subprocess
import torch
from .prefix_cache import pack_record, unpack_record

SCHEMA = 1
REQUIRED_PROVENANCE = {'baseline_commit', 'model_config', 'dataset', 'seed',
                       'schedule_steps', 'capture_world_size', 'sensor_config'}


def _json(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def _digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):value.update(chunk)
    return value.hexdigest()


def state_split(steps, *, audit_every=5):
    """Predeclared state holdout; never select states by performance."""
    steps=list(steps)
    if len(steps)!=len(set(steps)) or not steps or audit_every<2:
        raise ValueError('unique nonempty state IDs and audit_every >= 2 required')
    return {'train':[s for i,s in enumerate(steps) if (i+1)%audit_every],
            'audit':[s for i,s in enumerate(steps) if not (i+1)%audit_every]}


class StateBankWriter:
    def __init__(self,directory,provenance,*,expected_steps=1000,audit_every=5,sync_every=25):
        if expected_steps<1 or not REQUIRED_PROVENANCE.issubset(provenance):
            raise ValueError('positive state count and complete pinned provenance required')
        if provenance['schedule_steps']<expected_steps:
            raise ValueError('capture must not exceed the original optimizer schedule')
        if sync_every<1:raise ValueError('positive durable sync cadence required')
        self.sync_every=sync_every
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        if any(self.directory.iterdir()):raise FileExistsError('state bank must be a new empty directory')
        self.manifest=json.loads(json.dumps(dict(schema=SCHEMA,expected_steps=expected_steps,
            provenance=provenance,split=state_split(range(1,expected_steps+1),audit_every=audit_every),
            state_clock='completed_steps=s; next ordinary pass index=s; sensor pass index=s-1',
            baseline_identity=True,dense_optimizer_state=False,controller_state=False),allow_nan=False))
        _json(self.directory/'manifest.json',self.manifest)
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='state-bank')
        self.pending=None;self.next_step=1;self.entries=[];self.closed=False

    def append(self,completed_steps,model_state,snapshot,train,*,data_cursor_batches,identity_action):
        if self.closed:raise RuntimeError('state bank closed')
        if identity_action is not True:raise ValueError('only unedited baseline states may enter bank')
        if completed_steps!=self.next_step or completed_steps>self.manifest['expected_steps']:
            raise ValueError('states must be contiguous completed updates starting at one')
        if data_cursor_batches!=completed_steps:raise ValueError('model and training-data clocks differ')
        if self.pending is not None:self.pending.result()  # bounded one queued record
        started=time.perf_counter()
        names=set(snapshot.matrices)|set(snapshot.vectors)
        if not names.issubset(model_state):raise ValueError('observations refer to missing model parameters')
        if not all(isinstance(v,torch.Tensor) for v in model_state.values()):
            raise TypeError('model_state contains non-tensors; pass model.state_dict(), not a checkpoint')
        weights={n:v.detach().to('cpu',copy=True) for n,v in model_state.items()}
        packed=pack_record(completed_steps-1,snapshot,train,
            train_probe_count=self.manifest['provenance']['sensor_config'].get('train_sequence_probes',4))
        record=dict(schema=SCHEMA,completed_steps=completed_steps,data_cursor_batches=data_cursor_batches,
                    model=weights,observation=packed)
        self.pending=self.executor.submit(self._write,record,time.perf_counter()-started)
        self.next_step+=1

    def _write(self,record,freeze_seconds):
        step=record['completed_steps'];path=self.directory/f'state_{step:04d}.pt'
        temporary=path.with_suffix('.pt.tmp');start=time.perf_counter()
        with temporary.open('wb') as stream:
            torch.save(record,stream);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
        entry=dict(completed_steps=step,file=path.name,sha256=_digest(path),file_bytes=path.stat().st_size,
            weight_tensor_bytes=sum(v.numel()*v.element_size() for v in record['model'].values()),
            packed_observation_tensor_bytes=record['observation']['packed_tensor_bytes'],
            freeze_seconds=freeze_seconds,write_and_hash_seconds=time.perf_counter()-start)
        with (self.directory/'index.jsonl').open('a') as stream:
            stream.write(json.dumps(entry)+'\n');stream.flush();os.fsync(stream.fileno())
        self.entries.append(entry)
        if step%self.sync_every==0 or step==self.manifest['expected_steps']:
            _json(self.directory/'available.json',dict(completed_steps=step,complete=(step==self.manifest['expected_steps'])))
            subprocess.run(['sync',str(self.directory)],check=True)

    def close(self):
        if self.closed:return self.entries
        try:
            if self.pending is not None:self.pending.result()
            if len(self.entries)==self.manifest['expected_steps']:
                _json(self.directory/'complete.json',dict(steps=len(self.entries),
                    index_sha256=_digest(self.directory/'index.jsonl')))
                subprocess.run(['sync',str(self.directory)],check=True)
        finally:self.executor.shutdown(wait=True);self.closed=True
        return self.entries

    def __enter__(self):return self
    def __exit__(self,*args):self.close()


@dataclass(frozen=True)
class StateRecord:
    completed_steps: int
    model: dict
    snapshot: object
    train: object
    data_cursor_batches: int
    verified_source: dict | None = None
    # Optional composite-bank namespace. Local GPT/sensor/data clocks above
    # remain untouched; optimization/cache routing uses state_id instead.
    training_state_id: int | None = None
    bank_seed: int | None = None

    @property
    def state_id(self):
        return self.completed_steps if self.training_state_id is None else self.training_state_id

    def assert_verified(self):
        if self.verified_source is None:raise ValueError('state record lacks verified source provenance')
        source=self.verified_source;stat=Path(source['path']).stat()
        if (stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)!=source['file_stamp']:
            raise ValueError('verified state file changed; reload through reader')
        signature={n:(id(v),v._version) for n,v in self.model.items()}
        if signature!=source['tensor_signature']:raise ValueError('verified state tensors were mutated')
        return source['sha256']

    def load_model(self,model):
        current=model.state_dict()
        if set(current)!=set(self.model):raise ValueError('model state names differ')
        for name,tensor in current.items():
            saved=self.model[name]
            if tensor.shape!=saved.shape or tensor.dtype!=saved.dtype:
                raise ValueError(f'model state shape/dtype differs: {name}')
        model.load_state_dict(self.model,strict=True)


class StateBankReader:
    """Read immutable records, with optional repeated-epoch I/O optimizations.

    verify_once checks SHA256 on first access and whenever inode, size, mtime or
    ctime changes. Default False verifies every read. mmap_model maps CPU tensor
    storage privately instead of copying the dense weights; observation arrays
    are still restored independently. Neither option caches whole model states.
    """
    def __init__(self,directory,*,expected_provenance=None,require_complete=True,verify_once=False,mmap_model=False):
        self.directory=Path(directory)
        self.verify_once=bool(verify_once);self.mmap_model=bool(mmap_model);self._verified={}
        self.manifest=json.loads((self.directory/'manifest.json').read_text())
        if self.manifest['schema']!=SCHEMA or not self.manifest['baseline_identity']:
            raise ValueError('unsupported state bank')
        if expected_provenance is not None and self.manifest['provenance']!=expected_provenance:
            raise ValueError('state-bank provenance mismatch')
        index=(self.directory/'index.jsonl').read_text()
        if not require_complete:index=index[:index.rfind('\n')+1]
        self.entries=[json.loads(line) for line in index.splitlines()]
        if [e['completed_steps'] for e in self.entries]!=list(range(1,len(self.entries)+1)):
            raise ValueError('state bank is not a contiguous prefix')
        if require_complete:
            complete=json.loads((self.directory/'complete.json').read_text())
            if (complete['steps']!=len(self.entries) or len(self.entries)!=self.manifest['expected_steps']
                or complete['index_sha256']!=_digest(self.directory/'index.jsonl')):
                raise ValueError('incomplete or corrupted state-bank index')
        split=self.manifest['split'];a,b=set(split['train']),set(split['audit'])
        if a&b or a|b!=set(range(1,self.manifest['expected_steps']+1)):
            raise ValueError('state split overlaps or omits states')

    def read(self,completed_steps,*,device='cpu',model_device=None):
        if not 1<=completed_steps<=len(self.entries):raise IndexError('state outside committed bank')
        entry=self.entries[completed_steps-1];path=self.directory/entry['file']
        before=path.stat();stamp=(before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)
        if not self.verify_once or self._verified.get(completed_steps)!=stamp:
            if _digest(path)!=entry['sha256']:raise ValueError('state-bank record SHA256 mismatch')
            after=path.stat()
            if stamp!=(after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
                raise ValueError('state-bank record changed during verification')
            self._verified[completed_steps]=stamp
        # MAP_PRIVATE is PyTorch's default: consumer writes cannot alter the bank.
        # Same-device .to below preserves mapped CPUweights when requested.
        data=torch.load(path,map_location='cpu',weights_only=False,mmap=self.mmap_model)
        if data['completed_steps']!=completed_steps or data['data_cursor_batches']!=completed_steps:
            raise ValueError('state record clocks mismatch')
        obs=unpack_record(data['observation'],device=device)
        if obs.step!=completed_steps-1:raise ValueError('sensor and model clocks mismatch')
        weights={n:v.to(device if model_device is None else model_device) for n,v in data['model'].items()}
        source=dict(path=str(path.resolve()),sha256=entry['sha256'],file_stamp=stamp,
                    tensor_signature={n:(id(v),v._version) for n,v in weights.items()})
        return StateRecord(completed_steps,weights,obs.snapshot,obs.train,data['data_cursor_batches'],source)

    def iter_split(self,split='train',*,device='cpu'):
        if split not in ('train','audit'):raise ValueError('unknown state split')
        for step in self.manifest['split'][split]:yield self.read(step,device=device)
