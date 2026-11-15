"""Qualify the one-launch FP32 attention/VJP primitive, not whole offline training."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rewriter.persistent_attention import PersistentAttentionVJP
from scripts.benchmark_persistent_mlp import discrepancy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queries',type=int,default=64);p.add_argument('--keys',type=int,default=459)
    p.add_argument('--heads',type=int,default=4);p.add_argument('--width',type=int,default=32)
    p.add_argument('--causal',action='store_true');p.add_argument('--iterations',type=int,default=20)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    import wandb
    run=wandb.init(project='chisel-persistent-kernel',mode='offline',dir=str(args.output))
    torch.manual_seed(819);torch.backends.cuda.matmul.allow_tf32=False
    q=torch.randn(args.heads,args.queries,args.width,device='cuda',requires_grad=True)
    k=torch.randn(args.heads,args.keys,args.width,device='cuda',requires_grad=True)
    v=torch.randn_like(k,requires_grad=True);dy=torch.randn_like(q)
    kernel=PersistentAttentionVJP(q.detach(),k.detach(),v.detach(),dy,causal=args.causal)
    def eager():
        q.grad=k.grad=v.grad=None
        scores=(q@k.transpose(-1,-2))*args.width**-.5
        if args.causal:
            mask=torch.arange(args.keys,device='cuda')[None,:]>torch.arange(args.queries,device='cuda')[:,None]
            scores=scores.masked_fill(mask,float('-inf'))
        output=scores.softmax(-1)@v
        output.backward(dy)
        return output
    expected=eager().detach();kernel.step();torch.cuda.synchronize()
    checks=dict(output=discrepancy(expected,kernel.output),
        dq=discrepancy(q.grad,kernel.query_gradient),dk=discrepancy(k.grad,kernel.key_gradient),
        dv=discrepancy(v.grad,kernel.value_gradient))
    print(json.dumps(checks),flush=True)
    assert all(check['relative_l2']<5e-5 for check in checks.values())
    def measure(fn):
        for _ in range(3):fn()
        torch.cuda.synchronize();samples=[]
        for _ in range(args.iterations):
            start=time.perf_counter();fn();torch.cuda.synchronize();samples.append((time.perf_counter()-start)*1000)
        return statistics.median(samples)
    result=dict(scope=__doc__,checks=checks,eager_ms=measure(eager),persistent_ms=measure(kernel.step),
        shape=dict(heads=args.heads,queries=args.queries,keys=args.keys,width=args.width),causal=args.causal,
        stages=len(kernel.stages),stage_clock_cycles=(kernel.clocks[1:]-kernel.clocks[:-1]).cpu().tolist())
    result['speedup']=result['eager_ms']/result['persistent_ms']
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        kernel.step();torch.cuda.synchronize()
    trace=args.output/'persistent-trace.json';prof.export_chrome_trace(str(trace))
    result['profiled_kernel_names']=[e['name'] for e in json.loads(trace.read_text())['traceEvents'] if e.get('cat')=='kernel']
    result['profiled_kernel_count']=len(result['profiled_kernel_names'])
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True);run.log(dict(objective=result['speedup'],**result));run.finish()


if __name__=='__main__':main()
