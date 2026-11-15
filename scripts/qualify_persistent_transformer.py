"""Correctness qualification of one complete single-launch Chisel core block.

This is an integrated milestone, not the complete Chisel/GPT offline update.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import torch
from torch import nn
from torch.nn.attention import sdpa_kernel,SDPBackend
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rewriter.persistent_transformer import PersistentTransformerTrainer,PersistentTransformerStackTrainer
from scripts.benchmark_persistent_mlp import discrepancy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--tokens',type=int,default=64)
    parser.add_argument('--updates',type=int,default=3)
    parser.add_argument('--layers',type=int,default=1)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    import wandb
    run=wandb.init(project='chisel-persistent-kernel',mode='offline',dir=str(args.output))
    torch.manual_seed(948);torch.backends.cuda.matmul.allow_tf32=False
    layers=[nn.TransformerEncoderLayer(128,4,512,dropout=0.,activation='gelu',
                                        batch_first=True,norm_first=True).cuda() for _ in range(args.layers)]
    reference=layers[0] if args.layers==1 else nn.Sequential(*layers)
    actual=copy.deepcopy(reference)
    x=torch.randn(args.tokens,128,device='cuda');rx=x.clone().requires_grad_()
    dy=torch.randn_like(x)/args.tokens
    start=time.perf_counter()
    persistent=(PersistentTransformerTrainer(actual,x,dy) if args.layers==1 else
                PersistentTransformerStackTrainer(actual,x,dy))
    torch.cuda.synchronize()
    setup=time.perf_counter()-start
    optimizer=torch.optim.Adam(reference.parameters(),lr=2e-4,betas=(.9,.99),eps=1e-8)
    checks=[]
    for iteration in range(args.updates):
        optimizer.zero_grad(set_to_none=True);rx.grad=None
        with sdpa_kernel(SDPBackend.MATH):output=reference(rx[None])[0]
        output.backward(dy)
        ref_grad=torch.cat([p.grad.flatten() for p in reference.parameters()]).detach().clone()
        optimizer.step();persistent.step();torch.cuda.synchronize()
        act_grad=torch.cat([p.flatten() for p in persistent.gradients.values()])
        ref_values=torch.cat([p.detach().flatten() for p in reference.parameters()])
        act_values=torch.cat([p.detach().flatten() for p in actual.parameters()])
        check=dict(iteration=iteration,output=discrepancy(output.detach(),persistent.output),
            input_gradient=discrepancy(rx.grad,persistent.input_gradient),
            parameter_gradient=discrepancy(ref_grad,act_grad),updated_parameters=discrepancy(ref_values,act_values),
            individual_gradients={name:discrepancy(p.grad,persistent.gradients[name])
                                  for name,p in reference.named_parameters()})
        checks.append(check);print(json.dumps(check),flush=True)
        (args.output/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
        assert check['output']['relative_l2']<1e-4
        assert check['input_gradient']['relative_l2']<1e-4
        assert check['parameter_gradient']['relative_l2']<1e-4
        assert check['updated_parameters']['relative_l2']<1e-4
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        persistent.step();torch.cuda.synchronize()
    trace=args.output/'persistent-trace.json';prof.export_chrome_trace(str(trace))
    names=[e['name'] for e in json.loads(trace.read_text())['traceEvents'] if e.get('cat')=='kernel']
    result=dict(coverage=f'{args.layers} Chisel prenorm TransformerEncoderLayers forward/all first-order gradients/Adam; supplied output cotangent',
        complete_offline_update=False,checks=checks,setup_seconds=setup,stages=len(persistent.plan.stages),
        profiled_kernel_names=names,profiled_kernel_count=len(names),
        stage_clock_cycles=(persistent.plan.clocks[1:]-persistent.plan.clocks[:-1]).cpu().tolist())
    assert names==['persistent_training']
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}),flush=True)
    run.log(dict(objective=checks[-1]['parameter_gradient']['relative_l2'],kernel_count=len(names),stages=len(persistent.plan.stages)))
    run.finish()


if __name__=='__main__':main()
