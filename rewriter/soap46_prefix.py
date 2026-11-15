"""Explicit same-source, same-variant forks of an immutable zero-X step500.

Only future residual LR/decay metadata may change. The frozen role instead
extends the zero-LR warmup to the existing endpoint. Tensor storage is never
edited; fork_payload aliases immutable input banks until serialization.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import torch


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


def _stamp(s):return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)


def load_immutable(path):
    path=Path(path)
    if path.is_symlink():raise ValueError('prefix shards must be immutable regular files')
    with path.open('rb') as f:
        before=_stamp(os.fstat(f.fileno()));h=hashlib.sha256()
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
        f.seek(0);state=torch.load(f,map_location='cpu',weights_only=False)
        if before!=_stamp(os.fstat(f.fileno())) or before!=_stamp(path.stat()):
            raise ValueError('prefix changed while hashing/loading')
    return state,h.hexdigest()


def fork_payload(state,*,role,variant,seed,parent_sha256,implementation_sha256,
                 meta_lr=None,residual_decay=None):
    expected=dict(schema='record46-basis-v1',implementation_sha256=implementation_sha256,
                  completed_steps=500,data_cursor_batches=500,seed=seed,variant=variant,
                  warmup_steps=500,warmup_meta='moments',meta_ramp_steps=500,stop_step=2900)
    for key,value in expected.items():
        if state.get(key)!=value:raise ValueError('prefix mismatch: '+key)
    if role not in ('learned','frozen') or variant not in ('inner-outer','outer-only'):
        raise ValueError('unknown fork role/variant')
    if len(parent_sha256)!=64 or any(c not in '0123456789abcdef' for c in parent_sha256):
        raise ValueError('parent SHA256 is required')
    if state.get('residual_frame','basis')!='basis' or state.get('reset_meta_at_activation',False):
        raise ValueError('prefix requires original basis coordinates and no moment reset')
    hyper=state['hyper']
    for key in ('variant','warmup_steps','warmup_meta','meta_ramp_steps','residual_decay'):
        if hyper.get(key)!=state[key]:raise ValueError('inconsistent hyper metadata: '+key)
    if hyper['meta_steps']!=498 or hyper['warmup_meta_steps']!=498 or hyper['residual_updates']!=0:
        raise ValueError('prefix must contain exactly498 moment-only updates and no residual updates')
    entries=hyper['entries'];optimizer=hyper['optimizer']
    if not entries or optimizer is None:raise ValueError('missing warmup moments or pending credit')
    for name,row in entries.items():
        if name not in state['model']:raise ValueError('unknown controlled parameter')
        if any(torch.count_nonzero(row[k]).item()!=0 for k in ('xl','xr')):
            raise ValueError('prefix X must be exactly zero')
        record=row['record']
        if record is None or row['action_step']!=499 or int(record.step)!=499 or record.variant!=variant:
            raise ValueError('missing or stale action499 pending credit')
        if not math.isfinite(float(row['coefficient'])):raise ValueError('invalid wrapper cotangent')
    if len(optimizer['state'])!=2*len(entries):raise ValueError('incomplete residual Adam state')
    for saved in optimizer['state'].values():
        if float(saved['step'])!=498 or not all(torch.isfinite(saved[k]).all() for k in ('exp_avg','exp_avg_sq')):
            raise ValueError('invalid residual Adam moments/counter')
    if any(float(group['lr'])!=0 for group in optimizer['param_groups']):raise ValueError('prefix used nonzero residual LR')
    if state['wrapper_clock']['it']!=500 or state['wrapper_clock']['lookahead_status']:
        raise ValueError('prefix is not at the committed optimizer boundary')
    if state['muon_step_count']!=500:raise ValueError('wrong Muon clock')
    lr=state['meta_lr'] if meta_lr is None else meta_lr
    decay=state['residual_decay'] if residual_decay is None else residual_decay
    if not math.isfinite(lr) or lr<=0 or not math.isfinite(decay) or not 0<=decay<=1:
        raise ValueError('nominal LR must stay positive; direct decay must be in[0,1]')
    if role=='frozen' and (lr!=state['meta_lr'] or decay!=state['residual_decay']):
        raise ValueError('frozen role retains nominal LR and decay; only future warmup changes')
    result=dict(state);result['hyper']=dict(hyper);changes={}
    for key,value in (('meta_lr',lr),('residual_decay',decay)):
        if value!=state[key]:changes[key]={'before':state[key],'after':value}
        result[key]=value
    result['hyper']['residual_decay']=decay
    if role=='frozen':
        result['warmup_steps']=result['hyper']['warmup_steps']=state['stop_step']
        changes['warmup_steps']={'before':500,'after':state['stop_step']}
    result['prefix_fork']=dict(schema='soap46-prefix-fork-v1',role=role,parent_rank_sha256=parent_sha256,
        parent_implementation_sha256=implementation_sha256,completed_steps=500,seed=seed,variant=variant,
        declared_future_changes=changes,moments_reset=False,numerical_tensors_modified=False,
        converter_sha256=_sha(__file__))
    return result


def _model_hash(model):
    h=hashlib.sha256()
    for name,t in sorted(model.items()):
        h.update(repr((name,str(t.dtype),tuple(t.shape),tuple(t.stride()))).encode())
        h.update(memoryview(t.detach().contiguous().reshape(-1).view(torch.uint8).numpy()))
    return h.hexdigest()


def fork_directory(parent,output,*,role,variant,seed,implementation_sha256,meta_lr=None,
                   residual_decay=None,expected_world_size=8,expected_matrices=72):
    parent,output=Path(parent),Path(output)
    if output.exists():raise FileExistsError(output)
    header_bytes=(parent/'complete.json').read_bytes();header=json.loads(header_bytes)
    if header.get('completed_steps')!=500 or header.get('world_size')!=expected_world_size:
        raise ValueError('complete step500 distributed prefix required')
    output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.'+output.name+'.incomplete-',dir=output.parent))
    files={};ownership=set();model_hash=None;common=None;fork_header=None
    try:
        for rank in range(expected_world_size):
            filename=f'rank{rank}.pt';state,parent_hash=load_immutable(parent/filename)
            if state['world_size']!=expected_world_size:raise ValueError('wrong shard world size')
            metadata={k:state[k] for k in header if k in state}
            if any(metadata[k]!=header[k] for k in metadata):raise ValueError('shard/header metadata mismatch')
            if common is not None and metadata!=common:raise ValueError('rank metadata differs')
            common=metadata
            names=set(state['hyper']['entries'])
            if names&ownership:raise ValueError('duplicated controller ownership across shards')
            ownership|=names
            fingerprint=_model_hash(state['model'])
            if model_hash is not None and fingerprint!=model_hash:raise ValueError('replicated GPT weights differ')
            model_hash=fingerprint
            child=fork_payload(state,role=role,variant=variant,seed=seed,parent_sha256=parent_hash,
                implementation_sha256=implementation_sha256,meta_lr=meta_lr,residual_decay=residual_decay)
            torch.save(child,stage/filename)
            files[filename]=dict(parent_sha256=parent_hash,sha256=_sha(stage/filename),bytes=(stage/filename).stat().st_size)
            fork_header={**header,**{k:child[k] for k in ('meta_lr','residual_decay','warmup_steps')}}
            del state,child
        if len(ownership)!=expected_matrices:raise ValueError('incomplete hidden-matrix ownership')
        if (parent/'complete.json').read_bytes()!=header_bytes:raise ValueError('parent completion metadata changed')
        manifest=dict(schema='soap46-prefix-directory-fork-v1',role=role,seed=seed,variant=variant,
            parent=str(parent.resolve()),parent_complete_sha256=hashlib.sha256(header_bytes).hexdigest(),
            parent_implementation_sha256=implementation_sha256,model_sha256=model_hash,files=files,
            completed_steps=500,continuation_end_step=2900,training_launched=False,
            declared_future_changes={k:{'before':header[k],'after':fork_header[k]} for k in ('meta_lr','residual_decay','warmup_steps') if header[k]!=fork_header[k]},
            converter_sha256=_sha(__file__))
        (stage/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        fork_header['prefix_fork_manifest_sha256']=_sha(stage/'manifest.json')
        (stage/'complete.json').write_text(json.dumps(fork_header,indent=2)+'\n')
        stage.rename(output)
        return manifest
    except Exception as error:
        (stage/'FAILED.json').write_text(json.dumps(dict(error=repr(error)))+'\n')
        raise
