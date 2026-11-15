"""Isolated exact-tie ranking loss/cotangent qualification; no model or optimizer."""
from pathlib import Path
import argparse
import hashlib
import json
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from baseline.execution import enable_deterministic_execution
from rewriter.ranking_loss import tie_aware_topk_margin_loss
from rewriter.tiled_tie_ranking import tiled_tie_aware_topk_margin_loss


def check(teacher, student, k, *, margin=.5, tiny=False):
    student = student.detach().requires_grad_()
    actual = tiled_tie_aware_topk_margin_loss(teacher, student, k=k, margin_fraction=margin)
    expected = tie_aware_topk_margin_loss(teacher.double(), student.double(), k=k, margin_fraction=margin)
    weight = torch.linspace(-3., 4., actual.numel(), device=student.device,
                            dtype=torch.float64).reshape(actual.shape)
    ga = torch.autograd.grad((actual*weight).sum(), student, retain_graph=True)[0]
    ge = torch.autograd.grad((expected*weight).sum(), student)[0]
    torch.testing.assert_close(actual, expected, rtol=3e-13, atol=1e-26 if tiny else 1e-14)
    torch.testing.assert_close(ga, ge, rtol=3e-7, atol=1e-16 if tiny else 1e-8)
    assert torch.equal(ga != 0, ge != 0)
    return dict(k=k, shape=list(student.shape), max_loss_error=float((actual-expected).abs().max()),
                gradient_relative_error=float((ga-ge).double().norm()/ge.double().norm().clamp_min(1e-300)),
                max_gradient_error=float((ga-ge).abs().max()), nonzero_entries=int(ge.count_nonzero()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rows', type=int, default=16384)
    p.add_argument('--vocabulary', type=int, default=50304)
    p.add_argument('--k', type=int, default=100)
    p.add_argument('--repeats', type=int, default=3)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    enable_deterministic_execution();torch.set_num_threads(1);torch.manual_seed(417)
    import wandb
    run = wandb.init(project='chisel', mode='offline', dir=str(args.output),
                     config={**{k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                             'scope':'logit loss and cotangent only; no GPT/actor/Adam'})
    root = Path(__file__).resolve().parents[1]
    sources = ['rewriter/ranking_loss.py', 'rewriter/tiled_tie_ranking.py',
               'scripts/qualify_tiled_tie_ranking.py', 'tests/test_tiled_tie_ranking.py']
    (args.output/'source_hashes.json').write_text(json.dumps({name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources}, indent=2))
    checks=[]
    for k in (1,10,100,137):
        t=torch.randint(0,17,(6,137),device='cuda').double()
        s=torch.randn(6,137,device='cuda')
        checks.append(check(t,s,k,margin=.37))
    t=torch.tensor([[5.,5.,4.,4.,4.,3.,3.,2.],[5.,4.,4.,4.,4.,3.,2.,1.]],device='cuda')
    s=torch.tensor([[1.,1.,4.,4.,4.,6.,6.,0.],[3.,2.,2.,2.,2.,4.,0.,0.]],device='cuda')
    checks.append(check(t,s,4))
    t=(1.-torch.arange(137,device='cuda')*1e-6).repeat(3,1)
    t[:,50]=t[:,49];t[:,100]=t[:,99]
    s=t.clone();s[:,50]-=2e-6;s[:,101]=t[:,99]+1e-7
    checks.append(check(t,s,100,tiny=True))
    for v in (1,10):checks.append(check(torch.ones(2,v,device='cuda'),torch.randn(2,v,device='cuda'),v))
    (args.output/'small_checks.json').write_text(json.dumps(checks,indent=2))
    del t,s
    student=torch.randn(args.rows,args.vocabulary,device='cuda',dtype=torch.float32)
    teacher=torch.empty_like(student,dtype=torch.float64)
    for start in range(0,args.rows,128):
        teacher[start:start+128]=student[start:start+128].double().log_softmax(-1)
        student[start:start+128].add_(torch.randn_like(student[start:start+128])*1e-3)
    student.requires_grad_()
    # Chunk the independent full-shape oracle outside timing to bound validation memory.
    for start in range(0,args.rows,128):
        checks.append(check(teacher[start:start+128],student[start:start+128],args.k))
    (args.output/'checks.json').write_text(json.dumps(checks,indent=2))
    records=[]
    for repeat in range(args.repeats+1):
        for name in (('reference','tiled') if repeat%2==0 else ('tiled','reference')):
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter()
            loss=(tie_aware_topk_margin_loss(teacher,student.double(),k=args.k) if name=='reference'
                  else tiled_tie_aware_topk_margin_loss(teacher,student,k=args.k))
            grad=torch.autograd.grad(loss.mean(),student)[0]
            torch.cuda.synchronize()
            record=dict(backend=name,repeat=repeat,cold=repeat==0,seconds=time.perf_counter()-start,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(),loss_mean=float(loss.mean()),
                        gradient_norm=float(grad.double().norm()))
            records.append(record)
            with (args.output/'timings.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            run.log({**record,'objective':record['seconds']});del loss,grad
    summary=dict(scope='Synthetic logit loss plus logit cotangent only; no GPT, teacher head, actor or Adam',
                 shape=[args.rows,args.vocabulary],k=args.k,arithmetic='FP64 loss; FP32 student cotangent',
                 teacher_dtype='float64',student_dtype='float32',margin_fraction=.5,
                 cutoff='Exact torch.topk over FP64 boundary score buffer, temporary freed before backward',
                 temporary_boundary_score_bytes=args.rows*args.vocabulary*8,
                 max_gradient_relative_error=max(x['gradient_relative_error'] for x in checks),
                 checks=len(checks),timings=records,
                 median_seconds={name:statistics.median(x['seconds'] for x in records if not x['cold'] and x['backend']==name)
                                 for name in ('reference','tiled')})
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
    run.finish()


if __name__=='__main__':main()
