"""Qualify tiled ranking against the dense formula and time actual-shape logits."""
from pathlib import Path
import argparse,json,statistics,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baseline.execution import enable_deterministic_execution
from rewriter.ranking_loss import topk_margin_loss
from rewriter.tiled_ranking_loss import tiled_topk_margin_loss
from scripts.qualify_graph_offline import record_qualification_source

ARITHMETIC='fp32'
def candidate(t,s,**kw):return tiled_topk_margin_loss(t,s,arithmetic=ARITHMETIC,**kw)
def reference(t,s,**kw):
    if ARITHMETIC=='fp64':t,s=t.double(),s.double()
    return topk_margin_loss(t,s,**kw)
def check(t,s,k):
    a=candidate(t,s,k=k);r=reference(t,s,k=k)
    w=torch.linspace(-2,3,a.numel(),device=s.device).reshape(a.shape)
    ga,=torch.autograd.grad((a*w).sum(),s,retain_graph=True);gr,=torch.autograd.grad((r*w).sum(),s)
    torch.testing.assert_close(a,r,atol=2e-6,rtol=2e-6);torch.testing.assert_close(ga,gr,atol=2e-6,rtol=2e-6)
    return dict(k=k,loss_max_absolute=float((a-r).abs().max()),gradient_max_absolute=float((ga-gr).abs().max()))

def main():
    global ARITHMETIC
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--arithmetic",choices=("fp32","fp64"),default="fp32");p.add_argument('--output',type=Path,required=True);args=p.parse_args();ARITHMETIC=args.arithmetic;args.output.mkdir(parents=True,exist_ok=False);record_qualification_source(args.output,vars(args));enable_deterministic_execution();torch.set_num_threads(1);torch.manual_seed(194)
    import wandb
    run=wandb.init(project='chisel',mode='offline',dir=str(args.output),config={'scope':'synthetic loss component only; no GPT/optimizer','k':10,'margin_fraction':.5})
    checks=[]
    for k in (1,3,10):
        t=torch.randn(2,3,37,device='cuda');s=torch.randn_like(t,requires_grad=True);checks.append(check(t,s,k))
    t=torch.tensor([[5.,4.,3.,2.,1.]],device='cuda');s=torch.tensor([[0.,0.,6.,5.5,0.]],device='cuda',requires_grad=True);checks.append(dict(tied_outsiders=True,**check(t,s,2)))
    t=torch.ones(2,37,device='cuda');s=t.clone().requires_grad_();checks.append(dict(teacher_ties=True,**check(t,s,10)))
    for base in (.001,1.):
        t=(base-torch.arange(37,device='cuda')*1e-6).repeat(3,1)
        s=t.clone();s[:,5]=t[:,4]+1e-7;s[:,10]=t[:,9]+1e-7;s.requires_grad_()
        for logprob in (False,True) if ARITHMETIC=='fp64' else (False,):
            teacher=t.double().log_softmax(-1) if logprob else t
            a=candidate(teacher,s,k=10);r=reference(teacher,s,k=10);w=torch.tensor([-3.,.01,100.],device='cuda',dtype=a.dtype)
            ga,=torch.autograd.grad((a*w).sum(),s,retain_graph=True);gr,=torch.autograd.grad((r*w).sum(),s)
            assert bool((a>0).all()) and torch.equal(ga!=0,gr!=0)
            torch.testing.assert_close(a,r,atol=1e-25 if ARITHMETIC=='fp64' else 1e-20,rtol=1e-12 if ARITHMETIC=='fp64' else 3e-6)
            torch.testing.assert_close(ga,gr,atol=2e-14,rtol=3e-6)
            checks.append(dict(microgap_base=base,teacher_logprobs=logprob,reference_losses=r.tolist(),loss_max_absolute=float((a-r).abs().max()),gradient_relative_error=float((ga-gr).double().norm()/gr.double().norm())))
    del t,s,teacher,a,r,ga,gr
    t=torch.randn(16384,50304,device='cuda')*3;s=(t+torch.randn_like(t)*.2).requires_grad_()
    timings={'dense':[],'tiled':[]};peaks={};losses={}
    for repeat in range(6):
        for name in (('dense','tiled') if repeat%2==0 else ('tiled','dense')):
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            row=(reference if name=='dense' else candidate)(t,s)
            g,=torch.autograd.grad(row.sum()/row.numel(),s)
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            if repeat:timings[name].append(elapsed)
            if repeat:peaks[name]=max(peaks.get(name,0),torch.cuda.max_memory_allocated())
            losses[name]=float(row.double().mean())
            if repeat==0:
                # Compare full-shape values and gradients once outside timers.
                if name=='dense':reference_g=g.detach().clone();reference_row=row.detach().clone()
                else:
                    torch.testing.assert_close(row,reference_row,atol=2e-6,rtol=2e-6);torch.testing.assert_close(g,reference_g,atol=2e-10,rtol=2e-5)
                    checks.append(dict(full_shape=True,loss_max_absolute=float((row-reference_row).abs().max()),gradient_max_absolute=float((g-reference_g).abs().max())))
                    del reference_g,reference_row
            del row,g
    result=dict(arithmetic=ARITHMETIC,scope='Standalone16384x50304 syntheticFP32 ranking loss forward+logitbackward; teacher top10 recomputed inside both timers; no GPT/actor/Adam; no convergence claim',checks=checks,median_seconds={k:statistics.median(v) for k,v in timings.items()},timings=timings,peak_bytes=peaks,mean_losses=losses)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2));run.log(result['median_seconds']);print(json.dumps(result),flush=True);run.finish()
if __name__=='__main__':main()
