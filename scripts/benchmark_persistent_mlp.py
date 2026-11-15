"""Qualification of a literal persistent cooperative MLP training kernel.

The complete Chisel/GPT update is not implemented by this initial stage.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rewriter.persistent_mlp import PersistentMLPTrainer, CooperativeTrainingKernel


def discrepancy(reference, actual):
    return dict(relative_l2=float((reference-actual).double().norm()/reference.double().norm().clamp_min(1e-30)),
                max_absolute=float((reference-actual).abs().max()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows',type=int,default=8192)
    p.add_argument('--input-width',type=int,default=55)
    p.add_argument('--hidden-width',type=int,default=64)
    p.add_argument('--output-width',type=int,default=64)
    p.add_argument('--iterations',type=int,default=20)
    p.add_argument('--blocks',type=int,default=0,help='resident cooperative CTAs; zero uses one per SM')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    import wandb
    run=wandb.init(project='chisel-persistent-kernel',mode='offline',dir=str(args.output))
    torch.manual_seed(719);torch.backends.cuda.matmul.allow_tf32=False
    x=torch.randn(args.rows,args.input_width,device='cuda')
    dy=torch.randn(args.rows,args.output_width,device='cuda')/args.rows
    weights=[torch.randn(args.hidden_width,args.input_width,device='cuda')*.03,
             torch.randn(args.hidden_width,device='cuda')*.01,
             torch.randn(args.output_width,args.hidden_width,device='cuda')*.03,
             torch.randn(args.output_width,device='cuda')*.01]
    initial=[w.clone() for w in weights]
    start=time.perf_counter();kernel=PersistentMLPTrainer(x,dy,*weights,
        kernel=CooperativeTrainingKernel(blocks=args.blocks or None));torch.cuda.synchronize()
    compile_seconds=time.perf_counter()-start
    reference=[w.clone().requires_grad_() for w in initial]
    rx=x.clone().requires_grad_();optimizer=torch.optim.Adam(reference,lr=2e-4,betas=(.9,.99),eps=1e-8)
    checks=[]
    for iteration in range(3):
        optimizer.zero_grad(set_to_none=True);rx.grad=None
        output=F.linear(F.gelu(F.linear(rx,reference[0],reference[1])),reference[2],reference[3])
        output.backward(dy);optimizer.step()
        kernel.step();torch.cuda.synchronize()
        check=dict(iteration=iteration,output=discrepancy(output.detach(),kernel.output),
            input_gradient=discrepancy(rx.grad,kernel.input_cotangent),
            gradients=[discrepancy(p.grad,g) for p,g in zip(reference,kernel.gradients)],
            updated_parameters=[discrepancy(p.detach(),w) for p,w in zip(reference,weights)])
        checks.append(check);print(json.dumps(check),flush=True)
        assert check['output']['relative_l2']<3e-5
        assert check['input_gradient']['relative_l2']<5e-5
        assert max(c['relative_l2'] for c in check['gradients'])<5e-5
        assert max(c['relative_l2'] for c in check['updated_parameters'])<5e-5
    def eager():
        optimizer.zero_grad(set_to_none=True);rx.grad=None
        F.linear(F.gelu(F.linear(rx,reference[0],reference[1])),reference[2],reference[3]).backward(dy)
        optimizer.step()
    def measure(function):
        for _ in range(3):function()
        torch.cuda.synchronize();samples=[]
        for _ in range(args.iterations):
            start=time.perf_counter();function();torch.cuda.synchronize()
            samples.append((time.perf_counter()-start)*1000)
        return dict(median_ms=statistics.median(samples),samples_ms=samples)
    result=dict(coverage='one exact-GELU shared MLP, supplied output cotangent, forward/backward/Adam; NOT whole Chisel/GPT',
        implementation='one cooperative CUDA launch; device work queue, WMMA TF32x3, grid.sync stage barriers',
        shape=dict(rows=args.rows,input=args.input_width,hidden=args.hidden_width,output=args.output_width),
        compile_seconds=compile_seconds,checks=checks,stages=len(kernel.stages),blocks=kernel.kernel.blocks,
        maximum_resident_blocks=kernel.kernel.maximum_resident_blocks,registers=kernel.kernel.registers,
        shared_bytes=kernel.kernel.shared_bytes,eager=measure(eager),persistent=measure(kernel.step))
    result['speedup']=result['eager']['median_ms']/result['persistent']['median_ms']
    result['stage_clock_cycles']=(kernel.clocks[1:]-kernel.clocks[:-1]).cpu().tolist()
    if args.profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            kernel.step();torch.cuda.synchronize()
        trace=args.output/'persistent-trace.json';prof.export_chrome_trace(str(trace))
        events=json.loads(trace.read_text())['traceEvents']
        result['profiled_kernel_names']=[e['name'] for e in events if e.get('cat')=='kernel']
        result['profiled_kernel_count']=len(result['profiled_kernel_names'])
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('checks',)}),flush=True)
    run.log(dict(objective=result['speedup'],speedup=result['speedup'],persistent_ms=result['persistent']['median_ms']))
    run.finish()


if __name__=='__main__':main()
