"""Opt-in CPU FP32-logit disk cache prototype; default cache remains FP64.

One accounted frozen-teacher forward per sequence, no engine modification or
monkeypatch. Reconstruct the exact chunked FP64 log_softmax at access time.
Runtime/thread identity is deliberately strict; immutable trusted local files
are required, as for DiskTeacherCache. The local diagnostic selects this only with explicit fp32_logits format.
"""
import json
import platform
from pathlib import Path
import time
import torch
from rewriter.function_preservation import TeacherBatch
from scripts.diagnose_cpu_norm_preservation import tensor_sha,_verify_teacher_file,_teacher_stat,_hash_created_teacher_file

REPRESENTATION='cpu_softcapped_logits_fp32_reconstruct_logsoftmax_fp64_v1'


def runtime_identity(engine):
    return dict(torch_version=str(torch.__version__),torch_git_version=torch.version.git_version,
        machine=platform.machine(),cpu_capability=torch.backends.cpu.get_cpu_capability(),
        threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
        chunk_tokens=engine.chunk_tokens,gpt_precision=engine.gpt_precision,kl_dtype=str(engine.kl_dtype),
        float32_matmul_precision=torch.get_float32_matmul_precision(),deterministic_algorithms=torch.are_deterministic_algorithms_enabled())


def reconstruct_log_probs(logits,chunk_tokens):
    if logits.dtype!=torch.float32 or logits.device.type!='cpu' or logits.ndim!=3:raise ValueError('CPU FP32 three-dimensional logits required')
    if type(chunk_tokens) is not int or chunk_tokens<1:raise ValueError('positive integer token chunk required')
    result=torch.empty(logits.shape,dtype=torch.float64,device='cpu')
    flat=logits.flatten(0,1);out=result.flatten(0,1)
    for start in range(0,len(flat),chunk_tokens):
        out[start:start+chunk_tokens].copy_(torch.nn.functional.log_softmax(flat[start:start+chunk_tokens].to(dtype=torch.float64),dim=-1))
    return result


def _validate(engine,inputs,ids,state_sha):
    if getattr(engine,'_local_cpu_binding',None)!=(state_sha,engine.teacher_id):raise ValueError('verified teacher state binding required')
    if engine.device.type!='cpu' or engine.microbatch_sequences!=1 or engine.gpt_precision!='fp32' or engine.kl_dtype!=torch.float64:raise ValueError('CPU FP32/FP64 single-sequence teacher required')
    if inputs.ndim!=2 or min(inputs.shape)<1 or inputs.dtype not in (torch.int32,torch.int64) or len(ids)!=len(inputs):raise ValueError('nonempty integer inputs and exact sequence IDs required')


class RawLogitTeacherCache:
    @staticmethod
    @torch.no_grad()
    def create(engine,inputs,ids,state_sha,path):
        _validate(engine,inputs,ids,state_sha)
        path=Path(path);path.mkdir(parents=True,exist_ok=False)
        before=engine.compute.as_dict();files=[];runtime=runtime_identity(engine)
        for i in range(len(inputs)):
            selected=inputs[i:i+1].detach().to(device=engine.device,copy=True)
            logits=engine._forward(engine.parameters,selected)
            engine.compute.teacher_forwards+=1;engine.compute.teacher_tokens+=selected.numel()
            if logits.dtype!=torch.float32:raise ValueError('teacher did not emit FP32 logits')
            file=path/f'{i:04d}.pt'
            torch.save(dict(inputs=selected.cpu(),logits=logits.detach().cpu(),sequence_ids=ids[i:i+1]),file)
            files.append(dict(file=file.name,sha256=_hash_created_teacher_file(file),logits_shape=list(logits.shape)))
            del logits,selected
        if runtime_identity(engine)!=runtime:raise ValueError('teacher runtime changed during cache creation')
        meta=dict(representation=REPRESENTATION,runtime=runtime,state_sha256=state_sha,input_sha256=tensor_sha(inputs),sequence_ids=ids,
            files=files,teacher_compute={k:v-before[k] for k,v in engine.compute.as_dict().items()})
        (path/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
        return meta

    def __init__(self,path,engine,inputs,ids,state_sha):
        _validate(engine,inputs,ids,state_sha)
        self.path=Path(path);self.meta=json.loads((self.path/'manifest.json').read_text())
        if self.meta.get('representation')!=REPRESENTATION:raise ValueError('teacher representation mismatch')
        if self.meta['runtime']!=runtime_identity(engine):raise ValueError('teacher runtime/chunk/thread identity mismatch')
        if self.meta['state_sha256']!=state_sha or self.meta['input_sha256']!=tensor_sha(inputs) or self.meta['sequence_ids']!=ids or len(self.meta['files'])!=len(inputs):raise ValueError('teacher state/input identity mismatch')
        self.engine=engine;self.load_id=engine.teacher_id;self.inputs=inputs;self.ids=ids
        self.checksum_stats=dict(hits=0,files_hashed=0,bytes_hashed=0,hash_seconds=0.)
        self.files=[_verify_teacher_file(self.path/r['file'],r['sha256'],self.checksum_stats) for r in self.meta['files']]
        self.reconstruction_seconds=0.;self.reconstruction_calls=0

    @torch.no_grad()
    def __call__(self,start,stop):
        if self.engine.teacher_id!=self.load_id or getattr(self.engine,'_local_cpu_binding',None)!=(self.meta['state_sha256'],self.load_id):raise ValueError('stale teacher provider')
        if runtime_identity(self.engine)!=self.meta['runtime']:raise ValueError('teacher runtime/chunk/thread identity changed')
        if stop!=start+1 or not 0<=start<len(self.inputs):raise ValueError('single-sequence access required')
        path,stat=self.files[start]
        if _teacher_stat(path)!=stat:raise ValueError('teacher logit file changed')
        data=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        if data['logits'].dtype!=torch.float32 or list(data['logits'].shape)!=self.meta['files'][start]['logits_shape'] or not torch.equal(data['inputs'],self.inputs[start:stop]) or list(data['sequence_ids'])!=self.ids[start:stop]:raise ValueError('teacher cached batch mismatch')
        before=time.perf_counter();probs=reconstruct_log_probs(data['logits'],self.engine.chunk_tokens)
        self.reconstruction_seconds+=time.perf_counter()-before;self.reconstruction_calls+=1
        if _teacher_stat(path)!=stat:raise ValueError('teacher logit file changed during reconstruction')
        # Do not keep the entire raw-logit mmap alive through its small input view.
        return TeacherBatch(data['inputs'].clone(),probs,self.load_id,'past_training',tuple(data['sequence_ids']))
