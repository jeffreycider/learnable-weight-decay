"""CPU surrogate of the offline Chisel primal-dual loop; candidate ranking only.

This reproduces the CONTROL structure of scripts/train_multistate_chisel.py with
the production PrimalDual class and a real torch Adam, but replaces the
Transformer policy and GPT by a scalar magnitude per objective with a
quadratic KL law. It can rank (rho, dual_lr, target) candidates for cycle
amplitude; it cannot verify causality in the full policy. Calibration numbers
come from the corrected fit log (runs/chisel-fp32-eps23-features-v2):
15-group shrink speed ~0.007 %/update, KL/shrink^2 ~1.4e-5 at update 61,
200 of 800 checkpoints per update, gradient clip 1.0 active on every update.
"""
from __future__ import annotations
import argparse,json,math,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rewriter.primal_dual import PrimalDual


def simulate(*,rho,dual_lr,tolerance=1e-6,target=None,updates=240,states=800,batch=200,lr=.007,
             efficiency=1.4e-5,spread=.35,early_factor=1.5,seed=0,clip=1.,betas=(.9,.99),shrink_coefficient=1.,
             initial_shrink=.32,warm_start_multiplier=0.):
    """One surrogate fit. Returns per-update rows and summary statistics.

    The primal variable is the 15-group shrink percentage ``m`` (a scalar policy
    magnitude). Checkpoint i has KL_i = a_i m^2 with heterogeneous a_i: lognormal
    spread ``spread`` and an ``early_factor`` excess on the first 2% of states,
    matching the observed early-state excess. Each update visits ``batch``
    checkpoints of ``states`` (an epoch is states/batch updates), uses the
    production PrimalDual.terms for the penalty, sums state_weight 1/batch
    contributions, clips the gradient to ``clip`` before Adam exactly like the
    trainer, then performs the production dual step with pre-update KL.
    ``target`` is an optional training target below the evaluation tolerance;
    feasibility statistics are always reported against ``tolerance``.
    """
    generator=torch.Generator().manual_seed(seed)
    ids=list(range(1,states+1))
    a=efficiency*torch.exp(spread*torch.randn(states,generator=generator))
    a[:max(1,states//50)]*=early_factor
    dual=PrimalDual(ids,tolerance=target or tolerance,rho=rho,dual_lr=dual_lr)
    if warm_start_multiplier:dual.multipliers={i:warm_start_multiplier for i in ids}
    m=torch.tensor(float(initial_shrink))
    optimizer=torch.optim.Adam([m],lr=lr,betas=betas,eps=1e-8)
    rows=[]
    for update in range(updates):
        if update%(states//batch)==0:order=torch.randperm(states,generator=generator).tolist()
        visited=order[(update%(states//batch))*batch:(update%(states//batch)+1)*batch]
        kls=[];gradient=0.
        for index in visited:
            kl=float(a[index])*float(m)**2
            coefficient=dual.terms(kl,ids[index])['kl_coefficient']
            # exact d/dm of [ -c*m*w + coefficient*a_i*m^2*w ] with state_weight w=1/batch
            gradient+=(-shrink_coefficient+coefficient*2*float(a[index])*float(m))/batch
            kls.append((ids[index],kl))
        grad=abs(gradient)
        if grad>clip:gradient*=clip/grad
        m.grad=torch.tensor(gradient)
        optimizer.step()
        with torch.no_grad():m.clamp_(min=0.)
        dual.step(kls,expected_ids=[i for i,_ in kls])
        batch_kl=[k for _,k in kls]
        rows.append(dict(update=update+1,shrink=float(m.detach()),mean_kl=sum(batch_kl)/len(batch_kl),max_kl=max(batch_kl),
                         feasible_fraction=sum(k<=tolerance for k in batch_kl)/len(batch_kl),
                         multiplier_mean=sum(dual.multipliers.values())/states,gradient_norm=grad))
    tail=rows[len(rows)//2:]
    kl_tail=[r['mean_kl'] for r in tail]
    all_kl=(a*float(m.detach())**2)
    return dict(rows=rows,summary=dict(rho=rho,dual_lr=dual_lr,target=target,tolerance=tolerance,lr=lr,
        tail_mean_kl=sum(kl_tail)/len(kl_tail),tail_peak_to_trough=max(kl_tail)/max(min(kl_tail),1e-30),
        tail_fraction_updates_mean_above_tolerance=sum(k>tolerance for k in kl_tail)/len(kl_tail),
        tail_mean_shrink=sum(r['shrink'] for r in tail)/len(tail),
        final_fraction_states_at_or_below_tolerance=float((all_kl<=tolerance).double().mean()),
        final_shrink=float(m.detach()),final_multiplier_mean=rows[-1]['multiplier_mean']))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=240)
    parser.add_argument('--rhos',nargs='*',type=float,default=[.01,.1,1.,3.,10.])
    parser.add_argument('--dual-lrs',nargs='*',type=float,default=[.03,.1,.3])
    parser.add_argument('--targets',nargs='*',type=float,default=[1e-6,6e-7])
    parser.add_argument('--lrs',nargs='*',type=float,default=[.007,.0035])
    parser.add_argument('--seeds',nargs='*',type=int,default=[0,1])
    parser.add_argument('--plots',action='store_true')
    args=parser.parse_args(argv)
    args.output.mkdir(parents=True,exist_ok=True)
    results=[]
    for lr in args.lrs:
        for target in args.targets:
            for rho in args.rhos:
                for dual_lr in args.dual_lrs:
                    for seed in args.seeds:
                        run=simulate(rho=rho,dual_lr=dual_lr,target=None if target==1e-6 else target,lr=lr,seed=seed,updates=args.updates)
                        run['summary'].update(seed=seed);results.append(run)
    table=sorted((r['summary'] for r in results),key=lambda s:(s['tail_peak_to_trough'],-s['tail_mean_shrink']))
    (args.output/'summary.json').write_text(json.dumps(dict(calibration=dict(shrink_speed_percent_per_update=.007,kl_per_shrink_squared=1.4e-5,
        states=800,batch=200,clip=1.,betas=[.9,.99]),caveat='scalar surrogate; ranks candidates, does not verify full-policy causality',
        ranked=table),indent=2)+'\n')
    lines=['| lr | target | rho | dual_lr | seed | tail mean KL | tail peak/trough | tail frac updates > tol | tail shrink % | final states <= tol |','|---|---|---|---|---|---|---|---|---|---|']
    for s in table:
        lines.append(f"| {s['lr']:g} | {s['target'] or s['tolerance']:g} | {s['rho']:g} | {s['dual_lr']:g} | {s['seed']} | {s['tail_mean_kl']:.2e} | {s['tail_peak_to_trough']:.2f} | {s['tail_fraction_updates_mean_above_tolerance']:.2f} | {s['tail_mean_shrink']:.3f} | {s['final_fraction_states_at_or_below_tolerance']:.2f} |")
    (args.output/'summary.md').write_text('\n'.join(lines)+'\n')
    if args.plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(12,4.2))
        for rho in args.rhos:
            run=next(r for r in results if r['summary']['rho']==rho and r['summary']['dual_lr']==.1 and r['summary']['target'] is None and r['summary']['lr']==args.lrs[0] and r['summary']['seed']==args.seeds[0])
            axes[0].plot([r['update'] for r in run['rows']],[r['mean_kl'] for r in run['rows']],label=f'rho={rho:g}')
            axes[1].plot([r['update'] for r in run['rows']],[r['shrink'] for r in run['rows']],label=f'rho={rho:g}')
        axes[0].axhline(1e-6,color='k',ls='--',lw=.8);axes[0].set_yscale('log');axes[0].set_xlabel('surrogate update');axes[0].set_ylabel('batch mean KL (surrogate)');axes[0].legend(fontsize=7)
        axes[0].set_title(f'Surrogate: dual_lr 0.1, lr {args.lrs[0]:g} %/update, target = tolerance 1e-6')
        axes[1].set_xlabel('surrogate update');axes[1].set_ylabel('15-group shrink % (surrogate magnitude)');axes[1].legend(fontsize=7)
        fig.tight_layout();fig.savefig(args.output/'surrogate_dynamics.png',dpi=130);plt.close(fig)
    print('\n'.join(lines[:14]))


if __name__=='__main__':main()
