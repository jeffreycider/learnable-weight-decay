"""Weight-only bounds/coordinate search for binary channel gauge diagnostics.

Never imported by training. No outputs, losses, or query examples are inspected.
The search gives a feasible reward, NOT a global optimum. The upper bound is
analytic (triangle inequality + Holder), not a numerical optimizer's result.
"""
import math
import time
import numpy as np


def normalized_energies(first,bias,second):
    columns=[np.asarray(v,dtype=np.float64) for v in (first,bias,second)]
    if not columns[0].ndim==1 or any(v.shape!=columns[0].shape for v in columns):raise ValueError('matching energy vectors required')
    if any(not np.isfinite(v).all() or (v<0).any() for v in columns):raise ValueError('finite nonnegative energies required')
    totals=np.array([v.sum() for v in columns]);active=totals>0
    normalized=np.array([v/t if t>0 else np.zeros_like(v) for v,t in zip(columns,totals)])
    return normalized,totals,active


def group_cost(energies,exponents,inverse_power):
    factors=np.exp2(2*np.asarray(exponents,dtype=np.float64))
    return np.sqrt(np.array([np.dot(energies[0],factors),np.dot(energies[1],factors),
                            np.dot(energies[2],factors**(-inverse_power))]))


def analytic_reward_upper_bound(energies,*,inverse_power,groups=15):
    """Valid for all positive scales, hence also {-1,0,1} binary exponents.

    A=sum a_i d_i^2, B=sum b_i d_i^2, C=sum c_i d_i^(-2p).
    D_i=(sqrt(a_i)+sqrt(b_i))^2; sqrt(A)+sqrt(B)>=x=sqrt(sum D_i d_i^2).
    Holder gives x^p sqrt(C)>=S^((p+1)/2), where
    S=sum D_i^(p/(p+1))*c_i^(1/(p+1)). Therefore total cost >=
    (p+1)*p^(-p/(p+1))*sqrt(S). Zero groups contribute zero.
    """
    if inverse_power not in (1,2):raise ValueError('attention p=1 or ReLU-squared p=2 required')
    p=inverse_power;a,b,c=energies
    d=(np.sqrt(a)+np.sqrt(b))**2
    s=float(np.sum(d**(p/(p+1))*c**(1/(p+1))))
    lower=(p+1)*p**(-p/(p+1))*math.sqrt(s)
    active=int(np.count_nonzero(np.sum(energies,axis=1)))
    return dict(normalized_cost_lower_bound=lower,
        mean_group_l2_percentage_reduction_upper_bound=max(0.,100*(active-lower)/groups),
        derivation='triangle inequality plus Holder; valid for all positive per-channel scales',active_groups=active)


def binary_coordinate_search(first,bias,second,*,inverse_power,max_sweeps=8,groups=15):
    """Three deterministic starts, cyclic exact single-coordinate improvements.

    Identity, all-halved and all-doubled starts are evaluated from weights only.
    Changes can be reverted. At a zero-change sweep the result is coordinatewise
    optimal, not globally optimal. Every candidate has powers in {-1,0,1}.
    """
    if inverse_power not in (1,2) or max_sweeps<1:raise ValueError('invalid bounded search')
    started=time.perf_counter();e,total,active=normalized_energies(first,bias,second)
    n=e.shape[1];base=int(active.sum());factors=(.25,1.,4.);out=tuple(v**(-inverse_power) for v in factors)
    runs=[];best=None
    for initial in (1,0,2):
        choice=np.full(n,initial,dtype=np.int8)
        norm=group_cost(e,choice-1,inverse_power)**2;cost=float(np.sqrt(norm).sum());moves=[]
        for sweep in range(max_sweeps):
            changed=0
            for i in range(n):
                old=int(choice[i]);a,b,c=e[:,i];selected=old;selected_cost=cost;selected_norm=None
                for candidate in (1,0,2):
                    if candidate==old:continue
                    f=factors[candidate]-factors[old];o=out[candidate]-out[old]
                    norms=(max(0.,norm[0]+a*f),max(0.,norm[1]+b*f),max(0.,norm[2]+c*o))
                    value=sum(math.sqrt(v) for v in norms)
                    if value<selected_cost-1e-14:selected=candidate;selected_cost=value;selected_norm=norms
                if selected!=old:choice[i]=selected;norm=np.array(selected_norm);cost=selected_cost;changed+=1
            # Recontract from the original energies to remove incremental drift.
            norm=group_cost(e,choice-1,inverse_power)**2;cost=float(np.sqrt(norm).sum());moves.append(changed)
            if not changed:break
        reward=100*(base-cost)/groups
        record=dict(initial_exponent=initial-1,mean_group_l2_percentage_reduction=reward,
            sweeps=len(moves),coordinate_moves_per_sweep=moves,coordinatewise_converged=moves[-1]==0,
            selected_channels=int(np.count_nonzero(choice!=1)),halved_channels=int(np.sum(choice==0)),doubled_channels=int(np.sum(choice==2)),
            group_l2_percentage_reductions=(100*(active.astype(float)-np.sqrt(norm))).tolist())
        runs.append(record)
        if best is None or reward>best[0]:best=(reward,choice.copy(),record)
    # The unchanged gauge is always a valid fallback.
    if best[0]<0:best=(0.,np.ones(n,dtype=np.int8),dict(mean_group_l2_percentage_reduction=0.,selection='identity fallback'))
    bound=analytic_reward_upper_bound(e,inverse_power=inverse_power,groups=groups)
    if best[0]>bound['mean_group_l2_percentage_reduction_upper_bound']+1e-9:raise AssertionError('analytic upper bound violated')
    return dict(best=best[2],channel_exponents=(best[1]-1).tolist(),starts=runs,upper_bound=bound,
        reference_group_squared_norms=total.tolist(),inverse_power=inverse_power,
        elapsed_seconds=time.perf_counter()-started,objective_group_denominator=groups,
        diagnostic_only=True,global_optimum_claim=False,gpt_forwards=0,gpt_backwards=0)
