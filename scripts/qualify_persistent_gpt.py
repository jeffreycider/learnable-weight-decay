"""Compare literal student GPT forward/VJP against unchanged baseline math."""
import argparse,json,sys,time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baseline.model import GPT
from rewriter.function_preservation import gpt_logits
from rewriter.persistent_gpt import PersistentGPTVJP,append_rms_norm,rounded
from rewriter.persistent_plan import PersistentTrainingPlan
from scripts.benchmark_persistent_mlp import discrepancy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--tokens',type=int,default=17)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    import wandb
    run=wandb.init(project='chisel-persistent-kernel',mode='offline',dir=str(args.output))
    from baseline.execution import enable_deterministic_execution
    enable_deterministic_execution()
    torch.manual_seed(319);torch.backends.cuda.matmul.allow_tf32=False
    model=GPT(50304,12,768).cuda() if args.checkpoint else GPT(257,1,128).cuda()
    if args.checkpoint:
        state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(state.get('model',state.get('model_state_dict',state)))
    inputs=torch.randint(0,model.embed.weight.shape[0],(1,args.tokens),device='cuda')
    output=gpt_logits(model,inputs)
    cotangent=torch.randn_like(output)/output.numel()
    (output*cotangent).sum().backward()
    start=time.perf_counter();program=PersistentGPTVJP(model,inputs,cotangent.view(args.tokens,-1))
    torch.cuda.synchronize();setup=time.perf_counter()-start
    program.step();torch.cuda.synchronize()
    actual=program.logits.view_as(output)
    teacher=output.detach().double().log_softmax(-1)
    zero_kl=(teacher.exp()*(teacher-actual.double().log_softmax(-1))).sum(-1).mean()
    gradients={n:discrepancy(p.grad.float(),program.gradients[n]) for n,p in model.named_parameters() if n in program.gradients}
    whole=discrepancy(torch.cat([p.grad.float().flatten() for n,p in model.named_parameters() if n in program.gradients]),
                      torch.cat([program.gradients[n].flatten() for n,p in model.named_parameters() if n in program.gradients]))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        program.step();torch.cuda.synchronize()
    trace=args.output/'trace.json';prof.export_chrome_trace(str(trace))
    names=[e['name'] for e in json.loads(trace.read_text())['traceEvents'] if e.get('cat')=='kernel']
    result=dict(coverage='student GPT forward and all editable-parameter VJPs; supplied logit cotangent',
        complete_offline_update=False,logits=discrepancy(output.detach(),actual),zero_edit_baseline_teacher_kl=float(zero_kl),
        zero_edit_within_preservation_budget=float(zero_kl)<=1e-5,
        acceptance='finite outputs and one kernel are checked; numerical qualification requires the reported zero-edit floor and derivative tolerances',
        gradients=gradients,all_gradient=whole,setup_seconds=setup,stages=len(program.plan.stages),
        kernel_names=names,peak_allocated_bytes=torch.cuda.max_memory_allocated())
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    run.log(dict(objective=float(zero_kl),gradient_error=whole['relative_l2'],kernel_count=len(names)));run.finish()
    assert names==['persistent_training']
    assert torch.isfinite(actual).all() and torch.isfinite(zero_kl)


if __name__=='__main__':main()
