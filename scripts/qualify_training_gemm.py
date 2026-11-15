"""Isolated ordinary-GPT dense-gradient GEMM qualification on a retained state.

Uses the first16 sequences of rank0's NEXT ordinary training batch. No optimizer
step or active source mutation occurs here. Eager and actual compiled reference
loss/gradients are both retained; this is NOT an optimizer/trajectory certificate.
"""
from pathlib import Path
import argparse,hashlib,json,statistics,sys,time
from contextlib import nullcontext
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baseline.execution import enable_deterministic_execution
from baseline.fp32_execution import enable_fp32_execution
from baseline.data import FinewebStream
from baseline.model import GPT
from rewriter.experimental_training_gemm import experimental_training_gemms
from scripts.qualify_graph_offline import record_qualification_source


def comparison(actual,reference):
    totals=torch.zeros(5,dtype=torch.float64)
    by_name={}
    for name,ref in reference.items():
        got=actual[name];a=ref.double();b=got.double();error=b-a
        if got.shape!=ref.shape:raise ValueError('comparison tensor shape differs: '+name)
        if ref.numel()==0:
            by_name[name]=dict(reference_norm=0.,absolute_l2_error=0.,relative_error=0.,max_absolute_error=0.,empty_tensor=True)
            continue
        values=torch.tensor([a.square().sum(),b.square().sum(),error.square().sum(),(a*b).sum(),error.abs().max()],dtype=torch.float64)
        totals[:4]+=values[:4];totals[4]=torch.maximum(totals[4],values[4])
        by_name[name]=dict(reference_norm=float(values[0].sqrt()),absolute_l2_error=float(values[2].sqrt()),
            relative_error=float((values[2]/values[0].clamp_min(1e-60)).sqrt()),max_absolute_error=float(values[4]))
    rr,cc,ee,dot,maximum=totals.tolist()
    return dict(relative_error=(ee/max(rr,1e-60))**.5,absolute_l2_error=ee**.5,reference_norm=rr**.5,
        cosine=dot/max((rr*cc)**.5,1e-60),max_absolute_error=maximum,per_parameter=by_name)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--expected-sha256',required=True)
    p.add_argument('--train-data',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dx-backend',choices=('ieee','tf32x3','cublas3_large'),default='cublas3_large')
    p.add_argument('--dw-backend',choices=('ieee','tf32x3','tf32x3_splitk'),default='tf32x3')
    p.add_argument('--repeats',type=int,default=4)
    p.add_argument('--profile',action='store_true',help='Separate post-timing operator profiles of compiled arms')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    record_qualification_source(args.output,vars(args))
    digest=hashlib.sha256()
    with args.checkpoint.open('rb') as f:
        for chunk in iter(lambda:f.read(16<<20),b''):digest.update(chunk)
    if digest.hexdigest()!=args.expected_sha256:raise ValueError('retained checkpoint SHA differs')
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    config=state['config']
    if config.get('gpt_precision')!='fp32':raise ValueError('retained FP32 checkpoint required')
    torch.set_num_threads(1);enable_deterministic_execution();device=torch.device('cuda:0')
    model=GPT(**config['model']).float().to(device);model.load_state_dict(state['model'])
    enable_fp32_execution(model,compile_loss=bool(config.get('compile_loss',False)))
    loader=FinewebStream(args.train_data,config['batch_tokens'],config['seq_len'],rank=0,world_size=config['world_size'],device=device)
    by_name={f.name:f for f in loader.files};loader.files=[by_name[n] for n in state['data_cursor']['file_names']]
    loader.load_state_dict(state['data_cursor']);all_inputs,all_targets=next(loader)
    inputs,targets=all_inputs[:16].clone(),all_targets[:16].clone();del all_inputs,all_targets,loader
    compiled=torch.compile(model,fullgraph=True,dynamic=False)
    import wandb
    run=wandb.init(project='chisel',mode='offline',dir=str(args.output),config=dict(kind='ordinary_fp32_fullgradient_gemm',optimizer_updates=0,production_changed=False,**{k:str(v) for k,v in vars(args).items()}))
    stream=(args.output/'metrics.jsonl').open('w')
    def log(row):
        stream.write(json.dumps(row)+'\n');stream.flush();print(json.dumps(row),flush=True)
        run.log({k:v for k,v in row.items() if isinstance(v,(int,float,str,bool))})
    def execute(kind):
        model.zero_grad(set_to_none=True)
        context=(experimental_training_gemms(model,dx_backend=args.dx_backend,dw_backend=args.dw_backend)
                 if kind.startswith('experimental') else nullcontext())
        with context:
            loss=(compiled if kind in ('compiled_reference','experimental_compiled') else model)(inputs,targets)
            loss.backward()
        return loss.detach()
    kinds=['eager_reference','compiled_reference','experimental_eager']
    if args.dx_backend!='cublas3_large':kinds.append('experimental_compiled')
    cold={};losses={};grads={}
    for kind in kinds:
        torch.cuda.synchronize();tick=time.perf_counter();loss=execute(kind);torch.cuda.synchronize()
        cold[kind]=time.perf_counter()-tick;losses[kind]=float(loss)
        grads[kind]={n:v.grad.detach().cpu().clone() for n,v in model.named_parameters()}
        if any(not torch.isfinite(v).all() for v in grads[kind].values()):raise FloatingPointError('nonfinite ordinary gradient')
        log(dict(event='cold',backend=kind,seconds=cold[kind],mean_ce=float(loss)/targets.numel()))
    errors={kind:comparison(grads[kind],grads['eager_reference']) for kind in kinds[1:]}
    errors['experimental_vs_compiled']=comparison(grads['experimental_eager'],grads['compiled_reference'])
    for key,error in errors.items():log(dict(event='gradient',backend=key,**{k:v for k,v in error.items() if k!='per_parameter'}))
    # Pilot threshold only; passing does not qualify Muon or a scientific fit.
    if any(errors[k]['relative_error']>.003 or errors[k]['cosine']<.99999 for k in kinds if k.startswith('experimental')):
        (args.output/'failed_gradients.json').write_text(json.dumps(errors,indent=2)+'\n')
        raise RuntimeError('full-gradient pilot gate failed')
    rows=[]
    for kind in kinds:execute(kind)
    counters=dict(torch._dynamo.utils.counters['stats'])
    for repeat in range(args.repeats):
        for kind in (kinds if repeat%2==0 else list(reversed(kinds))):
            torch.cuda.synchronize();tick=time.perf_counter();execute(kind);torch.cuda.synchronize()
            row=dict(event='warm',backend=kind,repeat=repeat,seconds=time.perf_counter()-tick);rows.append(row);log(row)
    if dict(torch._dynamo.utils.counters['stats'])!=counters:raise RuntimeError('warm timing recompiled')
    if args.profile:
        for kind in ('compiled_reference','experimental_compiled'):
            if kind not in kinds:continue
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as profile:
                execute(kind);torch.cuda.synchronize()
            (args.output/(kind+'-profile.txt')).write_text(profile.key_averages(group_by_input_shape=True).table(sort_by='cuda_time_total',row_limit=80))
            profile.export_chrome_trace(str(args.output/(kind+'-trace.json')))
    summary=dict(scope='Retained actual1500 GPT, first16 rank0 next-batch1501 sequences; full dense gradients; no optimizer, sensors, distributed reduction, state/cache IO or trajectory',
        checkpoint_sha256=digest.hexdigest(),completed_steps=state['step'],microbatch=16,tokens=inputs.numel(),
        next_training_cursor=state['data_cursor'],dx_backend=args.dx_backend,dw_backend=args.dw_backend,
        mean_ce={k:v/targets.numel() for k,v in losses.items()},gradient_errors=errors,cold_seconds=cold,
        median_seconds={k:statistics.median(r['seconds'] for r in rows if r['backend']==k) for k in kinds},rows=rows,
        compilation_counters=counters,peak_allocated_bytes=torch.cuda.max_memory_allocated(),optimizer_updates=0,
        production_changed=False,source_config_compile=config.get('compile'),source_config_compile_loss=config.get('compile_loss'))
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');log(dict(event='finished',**summary['median_seconds']))
    run.finish();stream.close()


if __name__=='__main__':main()
