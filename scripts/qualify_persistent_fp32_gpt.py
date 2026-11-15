"""Literal FP32 GPT forward/VJP qualification, with fixed-teacher KL floor."""
from pathlib import Path
import argparse,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baseline.execution import enable_deterministic_execution
from baseline.model import GPT
from rewriter.function_preservation import FunctionalPreservation
from rewriter.persistent_fp32 import PersistentFP32GPTVJP
from rewriter.state_bank import StateBankReader


def metric(actual,reference):
    a,b=actual.double().flatten(),reference.double().flatten()
    return dict(relative=float((a-b).norm()/b.norm().clamp_min(1e-30)),
                maximum=float((a-b).abs().max()),reference_norm=float(b.norm()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--state-bank',type=Path)
    parser.add_argument('--state-id',type=int,default=999)
    parser.add_argument('--tokens',type=int,default=33)
    parser.add_argument('--batch',type=int,default=1)
    parser.add_argument('--compensated-backward',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);enable_deterministic_execution();torch.manual_seed(39)
    if args.state_bank:
        reader=StateBankReader(args.state_bank,verify_once=True,mmap_model=True)
        record=reader.read(args.state_id,model_device='cpu',device='cuda')
        model=GPT(**reader.manifest['provenance']['model_config']).cuda()
        record.load_model(model)
    else:
        model=GPT(257,2,128).cuda()
        # Baseline output/projection zero initialization is a trivial VJP test.
        with torch.no_grad():
            for name,p in model.named_parameters():
                if p.ndim==2:p.normal_(std=.03)
                elif 'bias' in name:p.normal_(std=.005)
    engine=FunctionalPreservation(model,gpt_precision='fp32',kl_dtype=torch.float64)
    inputs=torch.randint(model.embed.weight.shape[0],(args.batch,args.tokens),device='cuda')
    reference_parameters={n:p.detach().clone().requires_grad_(p.ndim in (1,2) and 'gains' not in n)
                          for n,p in engine.parameters.items()}
    output=engine._forward(reference_parameters,inputs)
    cotangent=torch.randn_like(output)/output.numel()
    (output*cotangent).sum().backward()
    import wandb
    run=wandb.init(project='chisel',mode='offline',dir=str(args.output),config=vars(args))
    # Model values are taken from the promoted engine, including embedding FP32.
    model=model.float();model.load_state_dict({**engine.parameters,**engine.buffers})
    started=time.perf_counter()
    program=PersistentFP32GPTVJP(model,inputs,cotangent.flatten(0,1),compensated_backward=args.compensated_backward)
    torch.cuda.synchronize();setup=time.perf_counter()-started
    program.step();torch.cuda.synchronize()
    actual=program.logits.reshape_as(output)
    teacher=output.detach().double().log_softmax(-1)
    floor=float((teacher.exp()*(teacher-actual.double().log_softmax(-1))).sum(-1).mean())
    gradients={n:metric(program.gradients[n],p.grad) for n,p in reference_parameters.items() if n in program.gradients}
    ref_grad=torch.cat([reference_parameters[n].grad.flatten() for n in program.gradients])
    actual_grad=torch.cat([program.gradients[n].flatten() for n in program.gradients])
    clocks=program.plan.clocks.detach().cpu().tolist()
    elapsed={}
    for i,instruction in enumerate(program.plan.instructions):
        op=str(instruction[0]);elapsed[op]=elapsed.get(op,0)+clocks[i+1]-clocks[i]
    def measure(fn):
        torch.cuda.synchronize();started=time.perf_counter()
        for _ in range(3):fn()
        torch.cuda.synchronize();return (time.perf_counter()-started)/3
    def reference():
        for p in reference_parameters.values():p.grad=None
        (engine._forward(reference_parameters,inputs)*cotangent).sum().backward()
    result=dict(logits=metric(actual,output),zero_edit_kl=floor,all_gradient=metric(actual_grad,ref_grad),
                gradients=gradients,literal_seconds=measure(program.step),reference_seconds=measure(reference),
                setup_seconds=setup,stage_clock_ticks=elapsed,stages=len(program.plan.instructions),
                single_launch=True,scope='GPT forward and supplied-logit VJP; no actor or optimizer',
                compensated_backward=args.compensated_backward,tokens=args.tokens,batch=args.batch)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True);run.log({'objective':floor,'gradient_error':result['all_gradient']['relative']});run.finish()


if __name__=='__main__':main()
