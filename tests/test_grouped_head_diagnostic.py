import torch
from scripts.diagnose_grouped_heads import summarize
from scripts.fit_cpu_teacher_heads_lstsq import solve


def test_functional_split_removes_incompatible_shared_rows_but_keeps_span_floor():
    # Same frozen feature, contradictory nonzero/zero targets in different roles.
    # Independent role heads interpolate; a shared head cannot do so.
    A=torch.tensor([[1.,2.,1.]],dtype=torch.float64)
    targets=[torch.tensor([2.],dtype=torch.float64),torch.zeros(1,dtype=torch.float64)]
    shared,_=solve(torch.cat([A,A]),torch.cat(targets))
    assert float(sum((A@shared-b).square().sum() for b in targets))>1.9
    groups={}
    for name,b,floor in zip(('role1','role2'),targets,(.25,0.)):
        coef,report=solve(A,b)
        assert report['rank']==1 and report['shape']==[1,3]
        groups[name]={'physical_error_squared':float((A@coef-b).square().sum())+floor}
    result=summarize(groups,4.25,4.25/2.,{'role1':2.,'role2':10.})
    assert abs(result['normalized_edit_error']-.25/4.25)<1e-14
    assert abs(result['relative_normalized_edit_error']-.25/4.25)<1e-14
