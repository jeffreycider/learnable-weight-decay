"""Weight-only diagnostic for binary gauges representable by chosen rank-r anchors.

No training code imports this. The greedy selection is constructive, not optimal;
finite-precision prediction preservation must be measured separately.
"""
import numpy as np
from .gauge_energy_diagnostic import normalized_energies, group_cost


def capped_binary_greedy(first,bias,second,block_ids,*,inverse_power,rank=8,groups=15,
                         require_matrix_improvement=False):
    """At most rank nonidentity channels per block, picked from identity.

    Exact global functional-group energies score every candidate. The best
    improving unselected channel is added until capacities fill or none improves.
    Ties use input order and half before double. No outputs/query data are read.
    Optional require_matrix_improvement additionally requires each selected
    move to reduce the sum of the two matrix groups' relative L2 norms. This
    selects diagnostic teaching targets; it does not change Chisel's objective
    or require every affected group to shrink individually.
    """
    if inverse_power not in (1,2) or type(rank) is not int or rank<1:raise ValueError('positive rank and inverse power1/2 required')
    if type(require_matrix_improvement) is not bool:raise ValueError('explicit matrix-improvement boolean required')
    e,totals,active=normalized_energies(first,bias,second)
    if len(block_ids)!=e.shape[1]:raise ValueError('one block identity per channel required')
    blocks=sorted(set(block_ids));index=np.array([blocks.index(x) for x in block_ids]);counts=np.zeros(len(blocks),dtype=int)
    choice=np.zeros(e.shape[1],dtype=np.int8);norm=e.sum(axis=1);cost=float(np.sqrt(norm).sum());moves=[]
    scales=np.array([.25,4.]);delta=np.stack((scales-1,scales-1,scales**(-inverse_power)-1))
    for _ in range(rank*len(blocks)):
        candidate_norms=np.sqrt(np.maximum(0,norm[:,None,None]+e[:,:,None]*delta[:,None,:]))
        candidates=candidate_norms.sum(axis=0)
        if require_matrix_improvement:
            matrix_cost=float(np.sqrt(norm[[0,2]]).sum())
            candidates[candidate_norms[[0,2]].sum(axis=0)>=matrix_cost-1e-14]=np.inf
        candidates[(choice!=0)|(counts[index]>=rank),:]=np.inf
        flat=int(np.argmin(candidates));channel,option=divmod(flat,2);newcost=float(candidates[channel,option])
        if newcost>=cost-1e-14:break
        choice[channel]=(-1,1)[option];counts[index[channel]]+=1
        norm=group_cost(e,choice,inverse_power)**2;cost=float(np.sqrt(norm).sum())
        move=dict(channel=channel,exponent=int(choice[channel]),mean_group_reduction_percent=100*(int(active.sum())-cost)/groups)
        if require_matrix_improvement:
            move['matrix_contribution_percent']=100*(int(active[[0,2]].sum())-np.sqrt(norm[[0,2]]).sum())/groups
        moves.append(move)
    return dict(channel_exponents=choice.tolist(),selected_per_block=dict(zip(blocks,counts.tolist())),selected_channels=int(np.count_nonzero(choice)),
        mean_group_l2_percentage_reduction=100*(int(active.sum())-cost)/groups,
        group_l2_percentage_reductions=(100*(active.astype(float)-np.sqrt(norm))).tolist(),reference_group_squared_norms=totals.tolist(),
        moves=moves,rank_limit=rank,inverse_power=inverse_power,global_optimum_claim=False,
        action_anchor_change_required=True,finite_logits_qualified=False,gpt_forwards=0,gpt_backwards=0)
