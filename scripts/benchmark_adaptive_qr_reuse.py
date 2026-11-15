"""Single-thread actual state6 query+actor CPU microbenchmark; excludes GPT/loss."""
import argparse
import json
import os
from pathlib import Path
import resource
import statistics
import sys
import time
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from rewriter.adaptive_coordinate_chisel import AdaptiveCoordinateChisel
from rewriter.adaptive_qr_reuse import evaluate_adaptive_with_qr_reuse
from rewriter.adaptive_sensing import query_weight_cores
from rewriter.prefix_cache import unpack_record


def run(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(451)
    data=torch.load(args.state,map_location='cpu',weights_only=False,mmap=True)
    snapshot=unpack_record(data['observation'],device='cpu').snapshot
    names=set(snapshot.matrices)|set(snapshot.vectors)
    weights={n:v.detach().float() for n,v in data['model'].items() if n in names}
    actor=AdaptiveCoordinateChisel(rank=8,trajectory_dim=400,model_dim=256,num_layers=2,
        latent_tokens=8,fresh_weight_core=True,factor_precision=args.precision,residual_precision='fp32')
    private=actor.make_private(snapshot,state_key=('qr-bench',6),seed=19)
    with torch.no_grad():
        actor.matrix_output.weight.normal_(std=.001)
        actor.vector_output.weight.normal_(std=.001)
        actor.blocks[0].attn.proj.weight.normal_(std=.001)
    # Fixed factor cotangents exercise every output and all QR branches without
    # materializing giant dense edits or executing student/teacher GPT.
    def forward(reuse):
        if reuse:return evaluate_adaptive_with_qr_reuse(actor,snapshot,private,weights)
        return actor(snapshot,private,weight_cores=query_weight_cores(weights,private))
    def tensors(action):
        return [t for f in action.matrices.values() for t in (f.U,f.C,f.V)]+list(action.vectors.values())
    modes=[False,True] if args.mode=='paired' else [args.mode=='reuse']
    with torch.no_grad(): cotangents=[torch.randn_like(t) for t in tensors(forward(modes[0]))]
    counts={}
    for mode in modes:
        with patch('torch.linalg.qr',wraps=torch.linalg.qr) as qr:
            action=forward(mode)
        counts[str(mode)]=qr.call_count
        del action
    parity = None
    if args.mode == 'paired':
        references = None
        max_output_abs = max_gradient_abs = 0.
        parameter_count = none_gradient_count = 0
        for mode in (False, True):
            actor.zero_grad(set_to_none=True);private.zero_grad(set_to_none=True)
            out=tensors(forward(mode))
            torch.autograd.backward(out,cotangents)
            grads=[p.grad for module in (actor,private) for p in module.parameters()]
            if references is None:
                references=([t.detach().clone() for t in out],
                            [None if g is None else g.clone() for g in grads])
            else:
                for t,r in zip(out,references[0]):
                    torch.testing.assert_close(t,r,rtol=0,atol=0)
                    max_output_abs=max(max_output_abs,float((t.detach()-r).abs().max()))
                for g,r in zip(grads,references[1]):
                    parameter_count+=1
                    assert (g is None)==(r is None)
                    if g is None: none_gradient_count+=1;continue
                    strict=args.precision=='fp64'
                    torch.testing.assert_close(g,r,rtol=1e-9 if strict else 3e-4,
                                               atol=1e-9 if strict else 2e-5)
                    max_gradient_abs=max(max_gradient_abs,float((g-r).abs().max()))
        parity=dict(max_output_abs=max_output_abs,max_gradient_abs=max_gradient_abs,
                    parameter_count=parameter_count,none_gradient_count=none_gradient_count)
        del references,out,grads
    rows={str(m):[] for m in modes}
    def iteration(mode):
        actor.zero_grad(set_to_none=True);private.zero_grad(set_to_none=True)
        start=time.perf_counter()
        out=tensors(forward(mode))
        middle=time.perf_counter()
        torch.autograd.backward(out,cotangents)
        end=time.perf_counter()
        return {'forward_s':middle-start,'backward_s':end-middle,'total_s':end-start}
    for _ in range(args.warmups):
        for mode in modes:iteration(mode)
    for i in range(args.iterations):
        for mode in modes if i%2==0 else list(reversed(modes)):
            rows[str(mode)].append(iteration(mode))
    result=dict(pid=os.getpid(),state=str(args.state),completed_steps=data['completed_steps'],
        precision=args.precision,mode=args.mode,threads=torch.get_num_threads(),
        matrices=len(snapshot.matrices),vectors=len(snapshot.vectors),qr_counts=counts,
        parity=parity,embedding_names=['embed.weight'],
        warmups=args.warmups,iterations=args.iterations,rows=rows,
        medians={mode:{k:statistics.median(r[k] for r in values) for k in values[0]}
                 for mode,values in rows.items()},max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='CPU query+actor only; fixed factor cotangents; no GPT, teacher, loss, Adam, or full training loop')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--state',type=Path,default=Path('data/chisel-teaching-diagnostic/state_0006.pt'))
    p.add_argument('--precision',choices=['fp32','fp64'],default='fp64')
    p.add_argument('--mode',choices=['paired','baseline','reuse'],default='paired')
    p.add_argument('--warmups',type=int,default=2)
    p.add_argument('--iterations',type=int,default=5)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
