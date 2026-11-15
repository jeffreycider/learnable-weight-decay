"""Isolated compensated head-input VJP split-K tuning; IEEE reference."""
import argparse,json,statistics,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baseline.execution import enable_deterministic_execution
from rewriter.persistent_mlp import CooperativeTrainingKernel
from rewriter.persistent_plan import PersistentTrainingPlan
from scripts.qualify_compensated_dx import comparison


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);enable_deterministic_execution();torch.manual_seed(832)
    torch.backends.cuda.matmul.allow_tf32=False
    a=torch.randn(256,50304,device='cuda')*.0001;b=torch.randn(50304,768,device='cuda')*.03
    reference=a@b;kernel=CooperativeTrainingKernel();rows=[]
    import wandb
    wb=wandb.init(project='chisel',mode='offline',dir=str(args.output),config=dict(shape=[256,768,50304],registers=kernel.registers,blocks=kernel.blocks))
    for chunk in (512,1024,2048,4096,8192):
        plan=PersistentTrainingPlan(a.device,kernel=kernel)
        output=plan.gemm(a,b,split=True,split_chunk=chunk);plan.finalize();plan.step();torch.cuda.synchronize()
        metric=comparison(output,reference);samples=[]
        for _ in range(7):
            torch.cuda.synchronize();start=time.perf_counter();plan.step();torch.cuda.synchronize()
            samples.append(time.perf_counter()-start)
        row=dict(split_k=chunk,seconds=statistics.median(samples),error=metric)
        rows.append(row);print(json.dumps(row),flush=True);wb.log(dict(split_k=chunk,seconds=row['seconds'],relative_error=metric['relative_error']))
        del plan,output
    result=dict(rows=rows,registers=kernel.registers,blocks=kernel.blocks,shared_bytes=kernel.shared_bytes,
                scope='single compensated head input VJP plus split reduction; one cooperative launch; not an offline step')
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');wb.finish()


if __name__=='__main__':main()
