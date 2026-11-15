"""Bounded no-gradient PR323 throughput measurement, outside training benchmarks.

Uses real already-consumed sequence IDs 0..127 from the first training shard.
Raw logits include the exact rational 15-softcap. Pipeline adds the actual FP64
teacher log-probability cache and chunked teacher||student KL. No batch cap from
FunctionalPreservation is applied to this separate diagnostic.
"""
from __future__ import annotations
import argparse,gc,hashlib,json,math,statistics,struct,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baseline.model import GPT
from baseline.execution import enable_deterministic_execution,execution_environment
from baseline.train import source_record
from rewriter.function_preservation import gpt_logits,chunked_teacher_kl

BATCHES=(1,2,4,8,16,32,64,128)


def first_past_sequences(path,count,seq_len):
    path=Path(path)
    if 'train' not in path.name:raise ValueError('training shard required')
    with path.open('rb') as f:
        header=f.read(1024)
        if len(header)!=1024:raise ValueError('truncated dataset header')
        magic,version,tokens=struct.unpack('<iii',header[:12])
        if (magic,version)!=(20240520,1) or tokens<count*seq_len+1:
            raise ValueError('invalid or insufficient training shard')
        raw=f.read(2*count*seq_len)
    if len(raw)!=2*count*seq_len:raise ValueError('truncated token data')
    values=np.frombuffer(raw,dtype='<u2').astype(np.int32).reshape(count,seq_len).copy()
    return torch.from_numpy(values)


def teacher_log_probs(logits,chunk_tokens=128):
    if chunk_tokens<1:raise ValueError('positive chunk size required')
    result=torch.empty_like(logits,dtype=torch.float64)
    source=logits.flatten(0,1);target=result.flatten(0,1)
    for i in range(0,len(source),chunk_tokens):
        target[i:i+chunk_tokens].copy_(F.log_softmax(source[i:i+chunk_tokens].double(),dim=-1))
    return result


@torch.no_grad()
def preservation_pipeline(teacher,student,inputs,chunk_tokens=128,counter=None):
    if counter is not None:counter["teacher_forward_attempts"]+=1
    logits=gpt_logits(teacher,inputs)
    logq=teacher_log_probs(logits,chunk_tokens)
    del logits
    if counter is not None:counter["student_forward_attempts"]+=1
    edited=gpt_logits(student,inputs)
    return chunked_teacher_kl(edited,logq,chunk_tokens)


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def benchmark_point(call,*,batch,seq_len,warmup,repeats,model_forwards):
    """Synchronized host latency; catches CUDA OOM only, with no batch fallback."""
    gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
    times=[];completed=0;result=None
    try:
        for index in range(warmup+repeats):
            torch.cuda.synchronize();start=time.perf_counter()
            result=call();torch.cuda.synchronize();elapsed=time.perf_counter()-start
            completed+=1
            if index>=warmup:times.append(elapsed)
            del result;result=None
        median=statistics.median(times)
        return dict(status='ok',batch_sequences=batch,seq_len=seq_len,latency_seconds=times,
            median_seconds=median,tokens_per_second=batch*seq_len/median,
            model_token_evaluations_per_second=model_forwards*batch*seq_len/median,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),baseline_allocated_bytes=baseline,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),warmup=warmup,repeats=repeats,
            completed_calls=completed,model_forward_attempts=model_forwards*completed)
    except torch.cuda.OutOfMemoryError as exc:
        row=dict(status='oom',batch_sequences=batch,seq_len=seq_len,error=str(exc),
            completed_calls=completed,model_forward_attempts_upper_bound=model_forwards*(completed+1),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),baseline_allocated_bytes=baseline,
            completed_latency_seconds=times)
        result=None
        return row
    finally:
        gc.collect();torch.cuda.empty_cache()


def phase_timings(teacher,student,inputs,chunk_tokens=128):
    """One no-grad phase profile; synchronization costs belong to this diagnostic."""
    rows={}
    def timed(name,fn):
        torch.cuda.synchronize();start=time.perf_counter();value=fn();torch.cuda.synchronize()
        rows[name]=time.perf_counter()-start;return value
    with torch.no_grad():
        logits=timed('teacher_gpt_forward_seconds',lambda:gpt_logits(teacher,inputs))
        logq=timed('teacher_fp64_cache_seconds',lambda:teacher_log_probs(logits,chunk_tokens));del logits
        edited=timed('edited_student_gpt_forward_seconds',lambda:gpt_logits(student,inputs))
        loss=timed('fp64_kl_seconds',lambda:chunked_teacher_kl(edited,logq,chunk_tokens))
        rows['mean_kl']=float(loss);rows['batch_sequences']=len(inputs)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True,type=Path);p.add_argument('--edited-checkpoint',required=True,type=Path)
    p.add_argument('--train-shard',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--warmup',type=int,default=2);p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--chunk-tokens',type=int,default=128)
    a=p.parse_args()
    if a.warmup<0 or a.repeats<1 or a.chunk_tokens<1:p.error('invalid repetition/chunk count')
    a.output.mkdir(exist_ok=False,parents=True)
    torch.set_num_interop_threads(20);enable_deterministic_execution();torch.cuda.set_device(0)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    edited=torch.load(a.edited_checkpoint,map_location='cpu',weights_only=False,mmap=True)
    cfg=state['config']
    if state['step']!=1750 or edited['step']!=1750 or cfg['model']!=edited['config']['model']:
        raise ValueError('requires matching source/fork checkpoint1750')
    if cfg['seq_len']!=1024 or cfg['model']!=dict(vocab_size=50304,num_layers=12,model_dim=768):
        raise ValueError('requires full prescribed PR323 model')
    teacher=GPT(**cfg['model']).cuda().eval();teacher.load_state_dict(state['model']);teacher.requires_grad_(False)
    student=GPT(**cfg['model']).cuda().eval();student.load_state_dict(edited['model']);student.requires_grad_(False)
    del state,edited
    inputs=first_past_sequences(a.train_shard,128,1024).cuda()
    metadata=dict(experiment='actual PR323 no-grad forward batch throughput',benchmark_compliant=False,
        extra_training_updates=0,gpt_backwards=0,checkpoint_sha256=sha(a.checkpoint),
        edited_checkpoint_sha256=sha(a.edited_checkpoint),source_checkpoint_step=1750,
        model=cfg['model'],seq_len=1024,batches=list(BATCHES),warmup=a.warmup,repeats=a.repeats,
        source_sequence_ids=list(range(128)),source_training_step=0,
        inputs_sha256=hashlib.sha256(inputs.cpu().numpy().tobytes()).hexdigest(),train_shard=str(a.train_shard),
        execution=execution_environment(),torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),resident_gpt_weight_copies=2,scope='eager GPT no-grad FP32 full-vocabulary softcapped logits; BF16 internal activations',
        pipeline='teacher forward + GPU FP64 logprob cache + edited student forward + chunked FP64 KL; outputs not retained across repetitions',
        pipeline_tokens_per_second_definition='distinct input tokens/sec; two model-token evaluations per input token',
        batch128_policy='attempt only when twice batch64 transient peak plus baseline fits total VRAM; report skipped otherwise',
        oom_policy='record failure and clear CUDA cache; never silently lower batch')
    (a.output/'config.json').write_text(json.dumps(metadata,indent=2)+'\n');source_record(metadata,a.output)
    log=(a.output/'metrics.jsonl').open('w');results=[];started=time.perf_counter()
    counter=dict(teacher_forward_attempts=0,student_forward_attempts=0)
    def raw_forward(model,selected):
        counter['teacher_forward_attempts' if model is teacher else 'student_forward_attempts']+=1
        return gpt_logits(model,selected)
    def emit(row):
        row=dict(elapsed_seconds=time.perf_counter()-started,**row)
        line=json.dumps(row);print(line,flush=True);log.write(line+'\n');log.flush();return row
    emit(dict(event='start',**metadata))
    with torch.no_grad():
        for name,model in [('raw_teacher',teacher),('raw_edited',student),('teacher_student_fp64_kl',None)]:
            previous=None
            for batch in BATCHES:
                if batch==128:
                    total=torch.cuda.get_device_properties(0).total_memory
                    predicted=(previous['baseline_allocated_bytes']+2*(previous['peak_allocated_bytes']-previous['baseline_allocated_bytes'])) if previous and previous['status']=='ok' else None
                    if predicted is None or predicted>total*.95:
                        results.append(emit(dict(event='measurement',mode=name,batch_sequences=batch,status='skipped_memory',predicted_peak_bytes=predicted,total_bytes=total,reason='batch64 failed or extrapolated transient peak exceeds95percent VRAM')));continue
                selected=inputs[:batch]
                call=(lambda:raw_forward(model,selected)) if model is not None else (lambda:preservation_pipeline(teacher,student,selected,a.chunk_tokens,counter))
                row=benchmark_point(call,batch=batch,seq_len=1024,warmup=a.warmup,repeats=a.repeats,model_forwards=1 if model is not None else 2)
                previous=row;results.append(emit(dict(event='measurement',mode=name,**row)))
    profiles=[]
    for batch in (1,2):
        profiles.append(emit(dict(event='pipeline_phase_profile',**phase_timings(teacher,student,inputs[:batch],a.chunk_tokens))))
    best={mode:max((r for r in results if r['mode']==mode and r['status']=='ok'),key=lambda:r['tokens_per_second']) for mode in ('raw_teacher','raw_edited','teacher_student_fp64_kl')}
    summary=dict(complete=True,results=results,phase_profiles=profiles,best_measured=best,elapsed_seconds=time.perf_counter()-started,
        training_updates=0,gpt_backwards=0,not_global_optimum_claim=True,
        forward_attempts=counter,phase_profile_additional_teacher_forwards=2,phase_profile_additional_student_forwards=2)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');emit(dict(event='finished',**summary));log.close()


if __name__=='__main__':main()
