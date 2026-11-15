"""Real retained-GPT attention cotangents and complete ordinary F+B pilot."""
from pathlib import Path
import argparse,hashlib,json,statistics,sys,time
from contextlib import nullcontext
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from baseline.model import GPT
from baseline.data import FinewebStream
from baseline.execution import enable_deterministic_execution
from baseline.fp32_execution import enable_fp32_execution
from rewriter.experimental_fp32_attention import fp32_attention,experimental_attention_context
from scripts.qualify_training_gemm import comparison
from scripts.qualify_graph_offline import record_qualification_source


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--expected-sha256',required=True)
    p.add_argument('--train-data',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=4)
    p.add_argument('--tune',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);record_qualification_source(a.output,vars(a))
    sha=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    if sha!=a.expected_sha256:raise ValueError('retained checkpoint SHA differs')
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False,mmap=True);cfg=state['config']
    torch.set_num_threads(1);enable_deterministic_execution();device=torch.device('cuda:0')
    model=GPT(**cfg['model']).float().to(device);model.load_state_dict(state['model']);enable_fp32_execution(model)
    loader=FinewebStream(a.train_data,cfg['batch_tokens'],cfg['seq_len'],rank=0,world_size=cfg['world_size'],device=device)
    names={f.name:f for f in loader.files};loader.files=[names[n] for n in state['data_cursor']['file_names']]
    loader.load_state_dict(state['data_cursor']);inputs,targets=next(loader);inputs,targets=inputs[:16].clone(),targets[:16].clone();del loader
    import wandb
    run=wandb.init(project='chisel',mode='offline',dir=str(a.output),config=dict(kind='fp32_compensated_attention',production_changed=False))
    captured={};index=0
    def capture(q,k,v,*,scale):
        nonlocal index
        current=index;index+=1
        out=F.scaled_dot_product_attention(q,k,v,is_causal=True,scale=scale)
        if current in (0,len(model.blocks)-1):
            captured[current]=dict(q=q.detach().cpu(),k=k.detach().cpu(),v=v.detach().cpu(),scale=scale)
            def hook(g):captured[current]['g']=g.detach().cpu()
            out.register_hook(hook)
        return out
    with experimental_attention_context(capture):model(inputs,targets).backward()
    model.zero_grad(set_to_none=True)
    results=[];tuning=[];selected_tiles=(32,32,4)
    for layer,record in captured.items():
        q,k,v=[record[key].to(device).requires_grad_() for key in ('q','k','v')];g=record['g'].to(device)
        def execute(backend):
            if backend=='math':
                with sdpa_kernel(SDPBackend.MATH):output=F.scaled_dot_product_attention(q,k,v,is_causal=True,scale=record['scale'])
            else:output=fp32_attention(q,k,v,scale=record['scale'])
            gradients=torch.autograd.grad(output,(q,k,v),g)
            return {'output':output.detach().cpu(),**{n:x.detach().cpu() for n,x in zip(('dq','dk','dv'),gradients)}}
        reference=execute('math');candidate=execute('compensated');errors=comparison(candidate,reference)
        # Timing excludes CPU diagnostic copies, includes layout copies in kernel.
        def timed(backend):
            if backend=='math':
                with sdpa_kernel(SDPBackend.MATH):output=F.scaled_dot_product_attention(q,k,v,is_causal=True,scale=record['scale'])
            else:output=fp32_attention(q,k,v,scale=record['scale'])
            torch.autograd.grad(output,(q,k,v),g)
        if a.tune and layer==0:
            from rewriter.kernels.fp32_attention_triton import configure_tiles
            for tile in ((32,32,4),(64,32,4),(32,64,8),(64,64,8),(64,64,4),(128,32,8)):
                configure_tiles(*tile);timed('compensated');samples=[]
                for _ in range(4):
                    torch.cuda.synchronize();tick=time.perf_counter();timed('compensated');torch.cuda.synchronize();samples.append(time.perf_counter()-tick)
                tuning.append(dict(tile=tile,seconds=samples,median_seconds=statistics.median(samples)))
                print(json.dumps(tuning[-1]),flush=True)
            selected_tiles=tuple(min(tuning,key=lambda x:x['median_seconds'])['tile']);configure_tiles(*selected_tiles)
            candidate=execute('compensated');errors=comparison(candidate,reference)
        rows=[]
        for repeat in range(a.repeats):
            for backend in ('math','compensated') if repeat%2==0 else ('compensated','math'):
                torch.cuda.synchronize();tick=time.perf_counter();timed(backend);torch.cuda.synchronize()
                row=dict(layer=layer,backend=backend,repeat=repeat,seconds=time.perf_counter()-tick);rows.append(row);run.log(dict(objective=row['seconds'],**row))
        result=dict(layer=layer,errors=errors,rows=rows,median_seconds={b:statistics.median(r['seconds'] for r in rows if r['backend']==b) for b in ('math','compensated')})
        results.append(result);print(json.dumps({k:v for k,v in result.items() if k!='errors'}),flush=True)
        (a.output/'components.json').write_text(json.dumps(results,indent=2)+'\n')
    del captured,q,k,v,g,reference,candidate
    compiled=torch.compile(model,fullgraph=True,dynamic=False)
    gradients={};losses={};cold={}
    def whole(backend):
        model.zero_grad(set_to_none=True)
        with experimental_attention_context() if backend=='compensated' else nullcontext():
            loss=compiled(inputs,targets);loss.backward()
        return loss.detach()
    for backend in ('math','compensated'):
        torch.cuda.synchronize();tick=time.perf_counter();loss=whole(backend);torch.cuda.synchronize();cold[backend]=time.perf_counter()-tick
        gradients[backend]={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters()};losses[backend]=float(loss)/targets.numel()
    error=comparison(gradients['compensated'],gradients['math']);rows=[]
    for backend in ('math','compensated'):whole(backend)
    before=dict(torch._dynamo.utils.counters['stats'])
    for repeat in range(a.repeats):
        for backend in ('math','compensated') if repeat%2==0 else ('compensated','math'):
            torch.cuda.synchronize();tick=time.perf_counter();whole(backend);torch.cuda.synchronize()
            row=dict(backend=backend,repeat=repeat,seconds=time.perf_counter()-tick);rows.append(row);run.log(dict(objective=row['seconds'],**row));print(json.dumps(row),flush=True)
    summary=dict(scope=__doc__,checkpoint_sha256=sha,components=results,whole_gradient_error=error,mean_ce=losses,cold_seconds=cold,rows=rows,
        median_seconds={b:statistics.median(r['seconds'] for r in rows if r['backend']==b) for b in ('math','compensated')},
        compilation_stable=before==dict(torch._dynamo.utils.counters['stats']),optimizer_updates=0,production_changed=False,
        arithmetic='FP32 inputs/probabilities/accumulators/outputs; TF32x3 dot products, online softmax; not IEEE math SDPA',selected_tiles=selected_tiles,tuning=tuning,
        limits='One actual checkpoint/micro16; no sensors, distributed SUM or saved-Muon response in this pilot')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');run.finish()


if __name__=='__main__':main()
