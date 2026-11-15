"""Read-only alignment diagnostics, including rounding outside the action subspace."""
import math
import torch


@torch.no_grad()
def tensor_alignment(base,new,anchors=None,*,chunk_elements=1<<20):
    """Computational orientation; no dense projector or extra model evaluation.

    Returns extensive quantities so disjoint tensors can be summed into groups.
    Projected weights are the canonical anchored projection, computed from actual
    base values; projected edit energy exposes stored-dtype rounding components.
    """
    if base.shape!=new.shape:raise ValueError('shape mismatch')
    if base.ndim==1:
        w=base.double();d=new.double()-w
        return dict(base_squared=float(w.square().sum()),edit_squared=float(d.square().sum()),
                    weight_dot_edit=float((w*d).sum()),projected_weight_squared=float(w.square().sum()),
                    projected_weight_dot_edit=float((w*d).sum()),projected_edit_squared=float(d.square().sum()))
    if base.ndim!=2 or anchors is None:raise ValueError('matrix anchors required')
    A=anchors.A.double();B=anchors.B.double();n,m=base.shape;r=B.shape[1]
    wb=torch.empty((n,r),device=base.device,dtype=torch.float64);db=torch.empty_like(wb)
    wta=torch.zeros((m,r),device=base.device,dtype=torch.float64);dta=torch.zeros_like(wta)
    totals=torch.zeros(3,device=base.device,dtype=torch.float64);chunk=max(1,chunk_elements//m)
    for start in range(0,n,chunk):
        end=min(start+chunk,n);w=base[start:end].double();d=new[start:end].double()-w
        wb[start:end]=w@B;db[start:end]=d@B;wta.add_(w.T@A[start:end]);dta.add_(d.T@A[start:end])
        totals+=torch.stack((w.square().sum(),d.square().sum(),(w*d).sum()))
    pw=wb-A@(A.T@wb);pd=db-A@(A.T@db)
    projected_w=pw.square().sum()+wta.square().sum();projected_d=pd.square().sum()+dta.square().sum()
    dot=(db*pw).sum()+(dta*wta).sum()
    return dict(base_squared=float(totals[0]),edit_squared=float(totals[1]),weight_dot_edit=float(totals[2]),
                projected_weight_squared=float(projected_w),projected_weight_dot_edit=float(dot),projected_edit_squared=float(projected_d))


def summarize_alignment(totals):
    denominator=math.sqrt(max(totals['projected_weight_squared'],0)*max(totals['edit_squared'],0))
    result=dict(totals)
    result.update(norm_squared_change=2*totals['weight_dot_edit']+totals['edit_squared'],
        cosine_to_negative_projected_weight=-totals['projected_weight_dot_edit']/denominator if denominator else None,
        radial_ratio_using_full_weight=-totals['weight_dot_edit']/denominator if denominator else None,
        realized_off_subspace_fraction=(1-totals['projected_edit_squared']/totals['edit_squared']) if totals['edit_squared'] else 0.)
    return result
