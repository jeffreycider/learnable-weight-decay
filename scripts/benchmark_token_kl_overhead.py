"""Isolated actual-shape loss overhead; synthetic fixed logits, no GPT or optimizer."""
from pathlib import Path
import argparse,json,statistics,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baseline.execution import enable_deterministic_execution
from rewriter.tiled_teacher_kl import tiled_teacher_kl
from scripts.qualify_graph_offline import record_qualification_source

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    record_qualification_source(a.output,vars(a));enable_deterministic_execution();torch.set_num_threads(1);torch.manual_seed(711)
    import wandb
    run=wandb.init(project='chisel',mode='offline',dir=str(a.output),config={'scope':'synthetic actual-shape loss component; no GPT/Adam','rows':16384,'vocab':50304})
    device='cuda';teacher=torch.randn(16384,50304,device=device)*3
    student=(teacher+torch.randn_like(teacher)*1e-4).requires_grad_()
    def body(kind):
        if kind=='mean':loss=tiled_teacher_kl(student,teacher)
        else:
            rows=tiled_teacher_kl(student,teacher,reduction='none');loss=rows.sum()/rows.numel()
            if kind=='token_tail':loss=loss+((rows-1e-8).clamp_min(0).square()/1e-8).mean()
        return torch.autograd.grad(loss,student)[0],loss.detach()
    mean_grad,mean_loss=body('mean');token_grad,token_loss=body('token_mean')
    equality={'loss_exact':bool(torch.equal(mean_loss,token_loss)),'gradient_exact':bool(torch.equal(mean_grad,token_grad))}
    if not all(equality.values()):raise RuntimeError('token mean changed default mean loss/gradient')
    del mean_grad,token_grad
    # Tiny FP64 oracle checks the actual CUDA row-weighted kernel independently.
    x=student[:7,:2049].detach().contiguous().requires_grad_();t=teacher[:7,:2049].contiguous();rows=tiled_teacher_kl(x,t,reduction='none')
    q=t.double().log_softmax(-1);ref=(q.exp()*(q-x.double().log_softmax(-1))).sum(-1);w=torch.arange(1,8,device=device,dtype=torch.float64)
    actual=torch.autograd.grad((rows*w).sum(),x)[0];expected=torch.autograd.grad((ref*w).sum(),x)[0]
    torch.testing.assert_close(rows,ref,atol=5e-14,rtol=1e-8);torch.testing.assert_close(actual,expected,atol=3e-11,rtol=1e-5)
    del x,t,rows,ref,q,actual,expected
    timings={k:[] for k in ('mean','token_mean','token_tail','teacher_top10','student_top10','teacher_top10_gather')}
    _,indices=torch.topk(teacher,10,dim=-1)
    for repeat in range(6):
        order=list(timings) if repeat%2==0 else list(reversed(timings))
        for name in order:
            torch.cuda.synchronize();start=time.perf_counter()
            if name.startswith('teacher_top10_gather'):out=student.gather(-1,indices)
            elif name=='teacher_top10':out=torch.topk(teacher,10,dim=-1)
            elif name=='student_top10':out=torch.topk(student,10,dim=-1)
            else:out=body(name)
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            if repeat:timings[name].append(elapsed)
            del out
    result=dict(scope='Isolated16384x50304 FP32 logit loss F+B; synthetic fixed logits; no GPT/actor/Adam; topk standalone read-only component, no ranking objective selected',exact_default_mean=equality,median_seconds={k:statistics.median(v) for k,v in timings.items()},timings=timings,teacher_top10_cache_bytes=indices.numel()*(indices.element_size()+teacher.element_size()),compact_int32_top10_cache_bytes=indices.numel()*(4+teacher.element_size()),peak_bytes=torch.cuda.max_memory_allocated())
    (a.output/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);run.log(result['median_seconds']);run.finish()
if __name__=='__main__':main()
