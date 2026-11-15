"""Measure rank-eight literal contractions against strict cuBLAS on same operands."""
import argparse,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baseline.execution import enable_deterministic_execution
from rewriter.persistent_mlp import CooperativeTrainingKernel,_check
from rewriter.persistent_plan import PersistentTrainingPlan


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--tiles',default='32,64,128');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);enable_deterministic_execution()
    import wandb
    wb=wandb.init(project='chisel',mode='offline',dir=str(args.output),config={'scope':__doc__})
    torch.manual_seed(915);results=[]
    shapes=[(16384,8,768,False),(16384,8,3072,False),(768,8,16384,True),(256,8,50304,False),(50304,8,256,True)]
    def elapsed(fn):
        for _ in range(5):fn()
        torch.cuda.synchronize();values=[]
        for _ in range(5):
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
            for _ in range(10):fn()
            b.record();b.synchronize();values.append(a.elapsed_time(b)/10)
        return sorted(values)[2]
    for tile in map(int,args.tiles.split(',')):
        os.environ['CHISEL_IEEE_SKINNY_K']=str(tile);kernel=CooperativeTrainingKernel()
        for m,n,k,ta in shapes:
            torch.manual_seed(915+m+k)
            a=torch.randn((k,m) if ta else (m,k),device='cuda')*.03;b=torch.randn(k,n,device='cuda')*.03
            plan=PersistentTrainingPlan(a.device,kernel=kernel);out=plan.gemm(a,b,ta=ta,ieee=True);plan.finalize()
            expected=(a.T if ta else a)@b;plan.step();torch.cuda.synchronize()
            error=float((out-expected).double().norm()/expected.double().norm())
            refout=torch.empty_like(expected)
            reference_ms=elapsed(lambda:torch.mm(a.T if ta else a,b,out=refout));literal_ms=elapsed(plan.step)
            row=dict(tile_k=tile,m=m,n=n,k=k,transpose_a=ta,relative_error=error,reference_ms=reference_ms,literal_ms=literal_ms,
                     slowdown=literal_ms/reference_ms,registers=kernel.registers,shared_bytes=kernel.shared_bytes,blocks=kernel.blocks,
                     stage_ticks=int((plan.clocks[-1]-plan.clocks[0]).item()),
                     local_bytes=_check(kernel.driver.cuFuncGetAttribute(kernel.driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,kernel.function),"local memory"))
            results.append(row);print(json.dumps(row),flush=True);wb.log(dict(row,objective=literal_ms))
            del plan,a,b,out,expected,refout
    (args.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n');wb.finish()

if __name__=='__main__':main()
