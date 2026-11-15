"""Full-actor persistent-kernel correctness; supplied action cotangent, no GPT yet."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rewriter.controller import GlobalRewriter
from rewriter.algebra import Factors,RawAction
from rewriter.persistent_controller import PersistentGlobalRewriterTrainer
from scripts.benchmark_persistent_mlp import discrepancy
from scripts.profile_controller import move_tree
from tests.test_controller import make_snapshot,small_controller


def flatten(action):
    return torch.cat([v.flatten() for factor in action.matrices.values() for v in (factor.P,factor.Q)]
                     +[v.flatten() for v in action.vectors.values()])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--full',action='store_true');p.add_argument('--updates',type=int,default=3)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    import wandb
    run=wandb.init(project='chisel-persistent-kernel',mode='offline',dir=str(args.output))
    torch.manual_seed(193);torch.backends.cuda.matmul.allow_tf32=False
    if args.full:
        from baseline.model import GPT
        from scripts.benchmark_chisel_kernels import synthetic_snapshot
        with torch.device('meta'):shape_model=GPT(50304,12,768)
        snapshot=synthetic_snapshot(shape_model);reference=GlobalRewriter().cuda()
    else:
        snapshot=move_tree(make_snapshot(),'cuda');reference=small_controller().cuda()
    with torch.no_grad():
        for decoder in (reference.row_decoder,reference.column_decoder,reference.vector_decoder):
            decoder[-1].weight.normal_(std=.01)
    actual=copy.deepcopy(reference)
    count=sum(o.anchors.A.numel()+o.anchors.B.numel() for o in snapshot.matrices.values())+sum(o.weight.numel() for o in snapshot.vectors.values())
    cotangents=RawAction({name:Factors(torch.randn(o.anchors.A.shape,device='cuda')/count,
        torch.randn(o.anchors.B.shape,device='cuda')/count,o.anchors) for name,o in snapshot.matrices.items()},
        {name:torch.randn(o.weight.shape,device='cuda')/count for name,o in snapshot.vectors.items()})
    flat_cotangent=flatten(cotangents)
    start=time.perf_counter();persistent=PersistentGlobalRewriterTrainer(actual,snapshot,cotangents)
    torch.cuda.synchronize();setup=time.perf_counter()-start
    optimizer=torch.optim.Adam(reference.parameters(),lr=2e-4,betas=(.9,.99),eps=1e-8)
    checks=[]
    for iteration in range(args.updates):
        optimizer.zero_grad(set_to_none=True)
        action=reference(snapshot);expected=flatten(action)
        (expected*flat_cotangent).sum().backward()
        expected_grad=torch.cat([value.grad.flatten() for value in reference.parameters()]).detach().clone()
        optimizer.step();persistent.step();torch.cuda.synchronize()
        actual_grad=torch.cat([value.flatten() for value in persistent.gradients.values()])
        expected_parameters=torch.cat([value.detach().flatten() for value in reference.parameters()])
        actual_parameters=torch.cat([value.detach().flatten() for value in actual.parameters()])
        check=dict(iteration=iteration,action=discrepancy(expected.detach(),flatten(persistent.output)),
            parameter_gradient=discrepancy(expected_grad,actual_grad),
            updated_parameters=discrepancy(expected_parameters,actual_parameters),
            reference_gradient_norm=float(expected_grad.norm()),
            individual_gradients={name:discrepancy(value.grad,persistent.gradients[name])
                                  for name,value in reference.named_parameters()})
        checks.append(check);print(json.dumps(check),flush=True)
        (args.output/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
        assert check['action']['relative_l2']<1e-4
        assert check['parameter_gradient']['relative_l2']<1e-4
        assert check['updated_parameters']['relative_l2']<2e-4
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        persistent.step();torch.cuda.synchronize()
    trace=args.output/'persistent-trace.json';prof.export_chrome_trace(str(trace))
    names=[e['name'] for e in json.loads(trace.read_text())['traceEvents'] if e.get('cat')=='kernel']
    result=dict(coverage='complete GlobalRewriter actor forward, all parameter gradients and Adam; supplied RawAction cotangent',
        full_pr323_shapes=args.full,complete_offline_update=False,checks=checks,setup_seconds=setup,
        stages=len(persistent.plan.stages),actor_parameters=sum(p.numel() for p in actual.parameters()),
        profiled_kernel_names=names,profiled_kernel_count=len(names),peak_allocated_bytes=torch.cuda.max_memory_allocated())
    assert names==['persistent_training']
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}),flush=True)
    run.log(dict(objective=checks[-1]['parameter_gradient']['relative_l2'],stages=len(persistent.plan.stages),kernel_count=len(names)))
    run.finish()


if __name__=='__main__':main()
