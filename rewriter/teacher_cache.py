"""Lossless final-head-feature teacher cache, not a dense-logit cache.

Generation evaluates the frozen transformer once per declared past-data panel.
Each use still evaluates the original frozen head GEMM, rational softcap, and
FP64 log-softmax. No transformer replay, quantization, or vocabulary truncation
occurs at cache load. The binding caller must have loaded the verified state-bank
record; the engine UUID prevents reuse after any later checkpoint load.
"""
from dataclasses import dataclass
import hashlib
import json
import os
import time
from pathlib import Path
import torch
from torch import nn
from torch.func import functional_call
import torch.nn.functional as F
from .function_preservation import TeacherBatch

FORMAT = 'exact_head_input_v1'
SOFTCAP = '15*x*(x.square()+225).rsqrt(); x=raw_head.float()'


def tensor_hash(value):
    value=value.detach().cpu().contiguous()
    digest=hashlib.sha256(str((tuple(value.shape),str(value.dtype))).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def file_hash(path):
    digest=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):digest.update(chunk)
    return digest.hexdigest()


def _head_signature(engine):
    return tuple((name,id(engine.parameters[name]),engine.parameters[name]._version)
                 for name in ('proj.weight','proj.bias'))


def bind_engine(engine,state_sha256):
    """Assert caller has loaded this SHA-verified complete checkpoint record."""
    if len(state_sha256)!=64:raise ValueError('checkpoint SHA256 required')
    engine._teacher_cache_binding=(engine.teacher_id,state_sha256)
    engine._teacher_cache_head_signature=None
    engine._teacher_cache_checked_head=None


def validate_precision(metadata,engine):
    from .gpt_precision import precision_metadata
    if metadata.get('gpt_precision','baseline')!=engine.gpt_precision:
        raise ValueError('teacher cache and student GPT precision differ')
    if engine.gpt_precision=='fp32' and metadata.get('precision')!=precision_metadata('fp32'):
        raise ValueError('FP32 teacher arithmetic or normalization epsilon differs')


class _HeadInputView(nn.Module):
    def __init__(self,model,gpt_precision='baseline'):
        super().__init__();self.gpt=model;self.gpt_precision=gpt_precision
    def forward(self,inputs):
        if self.gpt_precision=='fp32':
            from .gpt_precision import fp32_head_inputs
            return fp32_head_inputs(self.gpt,inputs)
        x=self.gpt.norm1(self.gpt.embed(inputs))
        for block in self.gpt.blocks:x=block(x)
        return self.gpt.norm2(x)


@torch.no_grad()
def head_inputs(engine,inputs):
    state={'gpt.'+n:v for n,v in engine.parameters.items()}
    buffers={'gpt.'+n:v for n,v in engine.buffers.items()}
    return functional_call(_HeadInputView(engine.view.gpt,engine.gpt_precision),(state,buffers),
        (inputs.to(engine.device),),strict=True,tie_weights=True)


@torch.no_grad()
def head_logits(engine,features):
    features=features.to(engine.device)
    raw=F.linear(features,engine.parameters['proj.weight'].to(features.dtype),
        engine.parameters['proj.bias'].to(features.dtype))
    x=raw.float()
    return 15*x*(x.square()+225).rsqrt()


@dataclass(frozen=True)
class TeacherPanel:
    inputs: torch.Tensor
    features: torch.Tensor
    sequence_ids: tuple
    metadata: dict

    def select_blocks(self, blocks):
        """Select complete generation microbatches, preserving row arithmetic.

        Inputs, sequence identities and exact head features move together. The
        parent panel's verified hashes remain provenance, rather than pretending
        the selected tensor has the parent's full-panel hash.
        """
        size=self.metadata['microbatch_sequences']
        blocks=tuple(int(b) for b in blocks)
        if not blocks or len(set(blocks))!=len(blocks) or any(b<0 or (b+1)*size>len(self.inputs) for b in blocks):
            raise ValueError('unique complete in-range microbatch blocks required')
        indices=torch.tensor([i for b in blocks for i in range(b*size,(b+1)*size)],dtype=torch.long)
        metadata=dict(self.metadata,selected_parent_blocks=list(blocks))
        return TeacherPanel(self.inputs.index_select(0,indices),self.features.index_select(0,indices),
                            tuple(self.sequence_ids[i] for i in indices.tolist()),metadata)

    @torch.no_grad()
    def teacher_batch(self,engine,start,stop):
        validate_precision(self.metadata,engine)
        if not 0<=start<stop<=len(self.inputs):raise ValueError('invalid panel slice')
        if stop-start>engine.microbatch_sequences:raise ValueError('slice exceeds GPT microbatch')
        if getattr(engine,'_teacher_cache_binding',None)!=(engine.teacher_id,self.metadata['state_sha256']):
            raise ValueError('cache must be bound to currently loaded verified checkpoint')
        if stop-start!=self.metadata['microbatch_sequences']:
            raise ValueError('teacher head reconstruction must preserve generation microbatch shape')
        if engine.kl_dtype!=torch.float64:raise ValueError('persisted targets require FP64 KL arithmetic')
        signature=getattr(engine,'_teacher_cache_head_signature',None)
        if signature is not None and signature!=_head_signature(engine):
            raise ValueError('bound frozen head tensor identity/version changed')
        checked=getattr(engine,'_teacher_cache_checked_head',None)
        if checked!=(engine.teacher_id,self.metadata['state_sha256']):
            for name,key in [('proj.weight','head_weight_sha256'),('proj.bias','head_bias_sha256')]:
                if tensor_hash(engine.parameters[name])!=self.metadata[key]:
                    raise ValueError('frozen head differs from cached checkpoint')
            engine._teacher_cache_checked_head=(engine.teacher_id,self.metadata['state_sha256'])
            engine._teacher_cache_head_signature=_head_signature(engine)
        logits=head_logits(engine,self.features[start:stop])
        log_probs=torch.empty(logits.shape,dtype=torch.float64,device=engine.teacher_cache_device)
        flat=logits.flatten(0,1);out=log_probs.flatten(0,1)
        for token in range(0,len(flat),engine.chunk_tokens):
            out[token:token+engine.chunk_tokens].copy_(F.log_softmax(flat[token:token+engine.chunk_tokens].double(),dim=-1))
        engine.cached_teacher_head_forwards=getattr(engine,'cached_teacher_head_forwards',0)+1
        engine.cached_teacher_head_tokens=getattr(engine,'cached_teacher_head_tokens',0)+self.inputs[start:stop].numel()
        return TeacherBatch(self.inputs[start:stop].clone(),log_probs,engine.teacher_id,
            'past_training',self.sequence_ids[start:stop])


@torch.no_grad()
def write_panel(directory,state_id,role,engine,inputs,sequence_ids,*,state_sha256,execution):
    if role not in ('train','audit'):raise ValueError('separate train/audit panel role required')
    if inputs.ndim!=2 or inputs.dtype not in (torch.int32,torch.int64):raise ValueError('integer sequence inputs required')
    if len(inputs)!=len(sequence_ids) or len(set(sequence_ids))!=len(sequence_ids):raise ValueError('unique matching sequence IDs required')
    batch=engine.microbatch_sequences
    if len(inputs)%batch:raise ValueError('fixed complete microbatches required')
    bind_engine(engine,state_sha256)
    generation_started=time.perf_counter()
    features=[]
    for start in range(0,len(inputs),batch):
        features.append(head_inputs(engine,inputs[start:start+batch]).cpu())
    features=torch.cat(features).contiguous()
    generation_seconds=time.perf_counter()-generation_started
    write_started=time.perf_counter()
    from .gpt_precision import precision_metadata
    metadata=dict(format=FORMAT,state_id=state_id,role=role,state_sha256=state_sha256,gpt_precision=engine.gpt_precision,precision=precision_metadata(engine.gpt_precision),
        input_sha256=tensor_hash(inputs),feature_sha256=tensor_hash(features),
        head_weight_sha256=tensor_hash(engine.parameters['proj.weight']),
        head_bias_sha256=tensor_hash(engine.parameters['proj.bias']),
        sequence_ids=list(map(int,sequence_ids)),microbatch_sequences=batch,
        feature_dtype=str(features.dtype),softcap=SOFTCAP,execution=execution,
        generation_transformer_forwards=len(inputs)//batch,generation_tokens=inputs.numel(),
        generation_transformer_and_cpu_copy_seconds=generation_seconds)
    path=Path(directory)/f'state_{state_id:04d}_{role}.pt';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    temporary=path.with_suffix('.tmp')
    torch.save(dict(inputs=inputs.detach().cpu().clone(),features=features,metadata=metadata),temporary)
    os.replace(temporary,path)
    info=dict(file=path.name,sha256=file_hash(path),bytes=path.stat().st_size,**metadata)
    info['write_hash_seconds']=time.perf_counter()-write_started
    path.with_suffix('.json').write_text(json.dumps(info,indent=2)+'\n')
    return info


class TeacherCacheReader:
    def __init__(self,directory,*,require_complete=True,verify_once=False,mmap=False):
        self.verify_once=bool(verify_once);self.mmap=bool(mmap);self._verified={}
        self.directory=Path(directory)
        manifest=self.directory/'manifest.json'
        self.manifest=json.loads(manifest.read_text()) if manifest.exists() else None
        if require_complete and (self.manifest is None or not self.manifest.get('complete')):
            raise ValueError('teacher cache generation is incomplete')
        self.expected={}
        if self.manifest is not None:
            for row in self.manifest['panels']:
                key=(row['state_id'],row['role'])
                if key in self.expected:raise ValueError('duplicate panel in cache manifest')
                self.expected[key]=row
            count=self.manifest['args']['states']
            if set(self.expected)!={(state,role) for state in range(1,count+1) for role in ('train','audit')}:
                raise ValueError('cache manifest omits requested panels')
    bind_engine=staticmethod(bind_engine)

    def load_engine(self,engine,record):
        """Own the verified record→frozen engine copy; no caller-asserted SHA."""
        sha=record.assert_verified()
        rows=[self.expected.get((record.completed_steps,role)) for role in ('train','audit')]
        if any(row is None or row['state_sha256']!=sha for row in rows):
            raise ValueError('verified state differs from cached teacher checkpoint')
        for row in rows:validate_precision(row,engine)
        if set(record.model)!=set(engine.parameters)|set(engine.buffers):
            raise ValueError('verified checkpoint model architecture differs')
        engine.load_teacher_state({n:record.model[n] for n in engine.parameters},
                                  {n:record.model[n] for n in engine.buffers})
        bind_engine(engine,sha)
        engine._teacher_cache_checked_head=(engine.teacher_id,sha)
        engine._teacher_cache_head_signature=_head_signature(engine)

    def read(self,state_id,role):
        if role not in ('train','audit'):raise ValueError('invalid panel role')
        path=self.directory/f'state_{state_id:04d}_{role}.pt'
        info=json.loads(path.with_suffix('.json').read_text())
        if self.manifest is not None:
            expected=self.expected.get((state_id,role))
            if expected is None or any(info[key]!=expected[key] for key in ('sha256','state_sha256','input_sha256','feature_sha256','sequence_ids')):
                raise ValueError('panel provenance differs from complete cache manifest')
        stat=path.stat();stamp=(stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
        checked=self.verify_once and self._verified.get((state_id,role))==(stamp,info['sha256'])
        if not checked and file_hash(path)!=info['sha256']:raise ValueError('teacher panel file hash mismatch')
        value=torch.load(path,map_location='cpu',weights_only=True,mmap=self.mmap);metadata=value['metadata']
        if metadata['format']!=FORMAT or metadata['softcap']!=SOFTCAP:raise ValueError('unsupported target execution')
        if metadata['state_id']!=state_id or metadata['role']!=role:raise ValueError('panel identity differs')
        for key,name in [('inputs','input_sha256'),('features','feature_sha256')]:
            if not checked and tensor_hash(value[key])!=metadata[name]:raise ValueError('panel tensor hash mismatch')
        after=path.stat()
        if stamp!=(after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
            raise ValueError('teacher panel changed during verification')
        self._verified[(state_id,role)]=(stamp,info['sha256'])
        return TeacherPanel(value['inputs'],value['features'],tuple(metadata['sequence_ids']),metadata)
