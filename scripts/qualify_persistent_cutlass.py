"""Check optional in-kernel CUTLASS tiles against BF16 cuBLAS on identical inputs."""
import json,sys,time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rewriter.persistent_plan import PersistentTrainingPlan
from rewriter.persistent_mlp import CooperativeTrainingKernel

torch.manual_seed(31)
kernel=CooperativeTrainingKernel()
print(json.dumps(dict(registers=kernel.registers,shared=kernel.shared_bytes,blocks=kernel.blocks)),flush=True)
for m,n,k in [(256,768,768),(768,768,1024),(256,50304,768),(256,768,50304)]:
    for ta,tb,split in [(False,False,False),(False,True,False),(True,False,True)]:
        a=torch.randn((k,m) if ta else (m,k),device='cuda').bfloat16().float()
        b=torch.randn((n,k) if tb else (k,n),device='cuda').bfloat16().float()
        bias=torch.randn(n,device='cuda') if not split else None
        plan=PersistentTrainingPlan(a.device,kernel=kernel)
        out=plan.gemm(a,b,ta=ta,tb=tb,split=split,bf16=True,bias=bias,epilogue=int(bias is not None));plan.finalize()
        plan.step();torch.cuda.synchronize()
        expected=(a.T if ta else a).bfloat16()@(b.T if tb else b).bfloat16()
        # cuBLAS BF16 output rounds; comparison rounds this kernel's FP32 sum too.
        if bias is None:measured=out.bfloat16().float();target=expected.float()
        else:measured=(out-bias).bfloat16().float();target=expected.float()
        error=float((measured-target).norm()/target.norm())
        started=time.perf_counter()
        for _ in range(5):plan.step()
        torch.cuda.synchronize();seconds=(time.perf_counter()-started)/5
        print(json.dumps(dict(m=m,n=n,k=k,ta=ta,tb=tb,split=split,error=error,seconds=seconds)),flush=True)
        assert error<.003
