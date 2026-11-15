"""Bounded integrity/I/O/head/probability timing for exact cached teacher targets."""
import argparse,json,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baseline.model import GPT
from baseline.execution import enable_deterministic_execution
from rewriter.state_bank import StateBankReader
from rewriter.teacher_cache import TeacherCacheReader
from rewriter.function_preservation import FunctionalPreservation


def main():
    p=argparse.ArgumentParser();p.add_argument('--state-bank',required=True);p.add_argument('--teacher-cache',required=True)
    p.add_argument('--state',type=int,default=1);p.add_argument('--repeats',type=int,default=3);p.add_argument('--output',required=True)
    args=p.parse_args()
    if not 1<=args.repeats<=10:p.error('bounded1..10 repeats required')
    enable_deterministic_execution();torch.cuda.set_device(0);torch.set_num_threads(1)
    device=torch.device('cuda');timings={}
    def timed(name,fn):
        torch.cuda.synchronize();start=time.perf_counter();value=fn();torch.cuda.synchronize()
        timings.setdefault(name,[]).append(time.perf_counter()-start);return value
    states=StateBankReader(args.state_bank,require_complete=False,verify_once=True,mmap_model=True)
    cache=TeacherCacheReader(args.teacher_cache,verify_once=True,mmap=True)
    record=timed('state_read_verify_cold',lambda:states.read(args.state))
    model=GPT(**states.manifest['provenance']['model_config']).to(device)
    microbatch=cache.manifest['args']['microbatch']
    engine=FunctionalPreservation(model,microbatch_sequences=microbatch,teacher_cache_device=device,kl_dtype=torch.float64)
    panel=timed('panel_read_verify_cold',lambda:cache.read(args.state,'train'))
    timed('owned_engine_load_bind',lambda:cache.load_engine(engine,record))
    maximum=0.
    for _ in range(args.repeats):
        panel=timed('panel_read_verified_warm',lambda:cache.read(args.state,'train'))
        target=timed('factorized_target_total',lambda:panel.teacher_batch(engine,0,microbatch))
        reference=timed('full_teacher_target_total',lambda:engine.cache_teacher(panel.inputs[:microbatch],data_role='past_training')[0])
        maximum=max(maximum,(target.log_probs-reference.log_probs).abs().max().item())
        features=timed('feature_host_to_device',lambda:panel.features[:microbatch].to(device))
        weight,bias=timed('head_parameter_cast',lambda:(engine.parameters['proj.weight'].to(features.dtype),engine.parameters['proj.bias'].to(features.dtype)))
        raw=timed('head_gemm',lambda:F.linear(features,weight,bias))
        def cap():
            x=raw.float();return 15*x*(x.square()+225).rsqrt()
        logits=timed('softcap',cap)
        def probabilities():
            result=torch.empty_like(logits,dtype=torch.float64);flat=logits.flatten(0,1);out=result.flatten(0,1)
            for start in range(0,len(flat),engine.chunk_tokens):out[start:start+engine.chunk_tokens].copy_(F.log_softmax(flat[start:start+engine.chunk_tokens].double(),dim=-1))
            return result
        measured=timed('fp64_probability_construction',probabilities)
        assert torch.equal(measured,target.log_probs)
        del target,reference,features,weight,bias,raw,logits,measured
    if maximum!=0:raise AssertionError('cached teacher changed exact target')
    result=dict(args=vars(args),seconds=timings,maximum_logprob_error=maximum,
        extra_full_teacher_forwards=args.repeats,extra_head_only_forwards=2*args.repeats,
        extra_gpt_backwards=0,torch=str(torch.__version__),peak_gpu_memory_bytes=torch.cuda.max_memory_allocated())
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
