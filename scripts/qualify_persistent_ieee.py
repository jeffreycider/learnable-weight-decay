"""Qualify IEEE FP32 persistent GEMM and its tiny finite-edit responses."""
from pathlib import Path
import argparse,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baseline.execution import enable_deterministic_execution
from rewriter.gpt_precision import configure_precision
from rewriter.persistent_mlp import CooperativeTrainingKernel
from rewriter.persistent_plan import PersistentTrainingPlan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);enable_deterministic_execution();configure_precision('fp32');torch.manual_seed(331)
    import wandb
    wb=wandb.init(project='chisel',mode='offline',dir=str(args.output),config={'backend':'literal_ieee_gemm','optimizer_updates':0})
    started=time.perf_counter();kernel=CooperativeTrainingKernel();cold=time.perf_counter()-started
    rows=[]
    for m,n,k in [(65,67,33),(256,768,768),(256,3072,768),(256,768,3072),(128,50304,768)]:
        for ta,tb in [(False,False),(False,True),(True,False)]:
            a=torch.randn((k,m) if ta else (m,k),device='cuda')/k**.5
            b=torch.randn((n,k) if tb else (k,n),device='cuda')
            original=b.clone();direction=torch.randn_like(b)*1e-5
            plan=PersistentTrainingPlan(a.device,kernel=kernel)
            actual=plan.gemm(a,b,ta=ta,tb=tb,ieee=True);plan.finalize()
            def reference():return (a.T if ta else a)@(b.T if tb else b)
            base=reference();plan.step();torch.cuda.synchronize();base_actual=actual.clone()
            for amplitude in (0.,.25,1.,4.):
                b.copy_(original+amplitude*direction)
                expected=reference();plan.step();torch.cuda.synchronize()
                difference=actual-expected;response=expected-base;actual_response=actual-base_actual
                row=dict(m=m,n=n,k=k,ta=ta,tb=tb,amplitude=amplitude,
                         output_relative_error=float(difference.norm()/expected.norm()),
                         response_relative_error=None if amplitude==0 else float((actual_response-response).norm()/response.norm()),
                         realized_edit=float((b-original).norm()/original.norm()),single_launch=True)
                rows.append(row);print(json.dumps(row),flush=True);wb.log(row)
            b.copy_(original)
            for _ in range(2):plan.step();reference()
            def measure(fn):
                torch.cuda.synchronize();start=time.perf_counter()
                for _ in range(5):fn()
                torch.cuda.synchronize();return (time.perf_counter()-start)/5
            timing=dict(m=m,n=n,k=k,ta=ta,tb=tb,reference_seconds=measure(reference),literal_seconds=measure(plan.step))
            rows.append(timing);print(json.dumps(timing),flush=True);wb.log(timing)
    (args.output/'summary.json').write_text(json.dumps(dict(rows=rows,cold_compile_seconds=cold,
        registers=kernel.registers,shared_bytes=kernel.shared_bytes,blocks=kernel.blocks),indent=2)+'\n')
    wb.finish()


if __name__=='__main__':main()
