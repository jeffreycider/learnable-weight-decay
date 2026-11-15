"""Isolated GPU optimizer-map qualification: zero GPT calls, no model training."""
import argparse
import ast
import dataclasses
import json
from pathlib import Path
import statistics
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from rewriter.soap46_map import committed_weight
from rewriter.soap46_conditioning import parameter_frame_weight
from test_soap46_map import make_record,reference_proposal


def relative(a,b):
    return float((a.double()-b.double()).norm()/a.double().norm().clamp_min(1e-30))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--compile',action='store_true')
    p.add_argument('--preserve-precision',action='store_true')
    p.add_argument('--residual-frame',choices=['basis','parameter'],default='basis')
    p.add_argument('--variants',nargs='+',choices=['inner-outer','outer-only'],default=['inner-outer','outer-only'])
    p.add_argument('--repeats',type=int,default=5);p.add_argument('--shapes',nargs='+',default=['768x768','3072x768'])
    a=p.parse_args();device=torch.device(a.device)
    if a.residual_frame=='parameter' and a.variants!=['outer-only']:raise ValueError('parameter frame requires --variants outer-only')
    map_function=parameter_frame_weight if a.residual_frame=='parameter' else committed_weight
    if device.type!='cuda':raise ValueError('this qualification requires CUDA')
    torch.set_num_threads(4);rows=[]
    a.output.mkdir(parents=True,exist_ok=True)
    for shape in a.shapes:
        n,m=map(int,shape.split('x'))
        for variant in a.variants:
            cpu=dataclasses.replace(make_record((n,m)),variant=variant)
            r=dataclasses.replace(cpu,**{f.name:getattr(cpu,f.name).to(device) for f in dataclasses.fields(cpu) if isinstance(getattr(cpu,f.name),torch.Tensor)})
            r,expected=reference_proposal(r)
            xl=torch.zeros_like(r.q_left,requires_grad=True);xr=torch.zeros_like(r.q_right,requires_grad=True)
            actual=map_function(r,xl,xr)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            row=dict(shape=shape,variant=variant,residual_frame=a.residual_frame,zero_x_upstream_bit_exact=True)
            # Same nonzero residual/tensor inputs for eager and compiled paths.
            with torch.no_grad():xl.normal_(std=.001);xr.normal_(std=.001)
            rt=dataclasses.replace(r,lr=r.weight.new_tensor(r.lr),step=r.weight.new_tensor(float(r.step)),cwd=r.weight.new_tensor(r.cwd))
            cot=torch.randn_like(r.weight)
            def run(fn,record):
                torch.cuda.synchronize();start=time.perf_counter()
                val=fn(record,xl,xr);grads=torch.autograd.grad((val*cot).sum(),(xl,xr))
                torch.cuda.synchronize()
                return val.detach(),tuple(g.detach() for g in grads),time.perf_counter()-start
            baseline,bgrad,_=run(map_function,rt)
            if not torch.isfinite(baseline).all() or not all(torch.isfinite(g).all() for g in bgrad):raise FloatingPointError('eager map or gradients nonfinite')
            row['eager_gradient_norms']=[float(g.double().norm()) for g in bgrad]
            durations=[run(map_function,rt)[2] for _ in range(a.repeats)]
            row['eager_median_seconds']=statistics.median(durations)
            if a.compile:
                from torch._dynamo.utils import counters
                compiled=torch.compile(map_function,fullgraph=True,dynamic=False,options={'emulate_precision_casts':True} if a.preserve_precision else None)
                torch.cuda.synchronize();cold_nograd_start=time.perf_counter()
                with torch.no_grad():compiled(rt,xl,xr)
                torch.cuda.synchronize();row['compile_cold_no_grad_seconds']=time.perf_counter()-cold_nograd_start
                value,grad,cold=run(compiled,rt)
                row.update(compile_cold_seconds=cold,compiled_value_relative_l2=relative(baseline,value),
                           compiled_gradient_relative_l2=[relative(x,y) for x,y in zip(bgrad,grad)])
                row['compiled_gradient_cosine']=[float(torch.nn.functional.cosine_similarity(x.flatten().double(),y.flatten().double(),dim=0)) for x,y in zip(bgrad,grad)]
                row['compiled_value_max_abs']=float((value-baseline).abs().max())
                (a.output/'numerical_check.json').write_text(json.dumps(row,indent=2))
                # Fail on material mismatch; retain raw measurements for review.
                torch.testing.assert_close(value,baseline,rtol=2e-5,atol=2e-5)
                if not all(torch.isfinite(g).all() for g in grad):raise FloatingPointError('compiled gradients nonfinite')
                before=counters['stats']['unique_graphs'];warm=[]
                for i in range(a.repeats):
                    state=dataclasses.replace(rt,lr=r.weight.new_tensor(.012+i*.0001),step=r.weight.new_tensor(1500.+i*100),direction=r.direction+i*.0001)
                    with torch.no_grad():compiled(state,xl,xr)
                    warm.append(run(compiled,state)[2])
                row.update(compiled_median_seconds=statistics.median(warm),graphs_after_first_gradient=before,
                           graphs_final=counters['stats']['unique_graphs'],graph_breaks=sum(counters['graph_break'].values()))
                if row['graphs_final']!=before:raise RuntimeError('schedule or observation change recompiled the map')
            row['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            rows.append(row);print(json.dumps(row),flush=True)
            (a.output/'qualification.json').write_text(json.dumps(dict(complete=False,rows=rows,gpt_forwards=0,gpt_backwards=0),indent=2))
            del r,rt,xl,xr,actual,expected,baseline,bgrad
            torch.cuda.empty_cache()
    (a.output/'qualification.json').write_text(json.dumps(dict(complete=True,rows=rows,gpt_forwards=0,gpt_backwards=0,
        torch_version=torch.__version__,gpu=torch.cuda.get_device_name(),allow_tf32=torch.backends.cuda.matmul.allow_tf32),indent=2))

if __name__=='__main__':main()
