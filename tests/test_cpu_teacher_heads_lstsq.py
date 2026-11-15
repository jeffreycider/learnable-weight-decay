import torch
import pytest
from scripts.fit_cpu_teacher_heads_lstsq import matrix_design,vector_design,solve,orthonormal_error


def test_design_dense_values_and_gradients_include_transpose_bias_and_scales():
    torch.manual_seed(35);dtype=torch.float64;r=3;d=4
    row=torch.randn(r,d,dtype=dtype,requires_grad=True);col=torch.randn(r,d,dtype=dtype,requires_grad=True)
    coefficients=torch.randn(r,d+1,dtype=dtype,requires_grad=True)
    u=torch.linalg.qr(torch.randn(7,r,dtype=dtype)).Q;v=torch.linalg.qr(torch.randn(5,r,dtype=dtype)).Q
    target=torch.randn(7,5,dtype=dtype);scale=.37
    core=(torch.nn.functional.linear(row,coefficients[:,:-1],coefficients[:,-1])+torch.nn.functional.linear(col,coefficients[:,:-1],coefficients[:,-1]).T)*scale
    design=matrix_design(row,col,scale)
    torch.testing.assert_close(design@coefficients.flatten(),core.flatten(),rtol=1e-13,atol=1e-13)
    dense_loss=(u@core@v.T-target).square().sum()
    projected=u.T@target@v
    design_loss=((design@coefficients.flatten()).reshape(r,r)-projected).square().sum()+target.square().sum()-projected.square().sum()
    torch.testing.assert_close(dense_loss,design_loss,rtol=1e-13,atol=1e-13)
    actual=torch.autograd.grad(design_loss,(coefficients,row,col),retain_graph=True)
    expected=torch.autograd.grad(dense_loss,(coefficients,row,col))
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
    assert orthonormal_error(u)<1e-12
    with pytest.raises(ValueError,match='column'):orthonormal_error(u.T)


def test_zero_target_rows_change_least_squares_solution_and_rank_report():
    h=torch.tensor([[1.],[1.]],dtype=torch.float64)
    design=vector_design(h,2.)
    target=torch.tensor([1.,0.],dtype=torch.float64)
    coeff,report=solve(design,target)
    torch.testing.assert_close(design@coeff,torch.tensor([.5,.5],dtype=torch.float64))
    assert report['rank']==1 and report['shape']==[2,2]
    assert report['prediction_squared']<=report['target_squared']
    assert report['rcond']==torch.finfo(torch.float64).eps*2
    ignored,_=solve(design[:1],target[:1])
    assert (design@ignored)[1]>.99


def test_control_state_identity_handles_extra_metadata_and_resume_rejects(tmp_path):
    from scripts.fit_cpu_teacher_heads_lstsq import state_equal
    from scripts.fit_cpu_edit_teacher import resume_adam
    state={'tensor':torch.ones(2),'extra':{'action_family':'adaptive','shape':(2,3)}}
    assert state_equal(state,state)
    assert not state_equal(state,{**state,'tensor':torch.zeros(2)})
    path=tmp_path/'control.pt';torch.save({'diagnostic_control':True},path)
    with pytest.raises(ValueError,match='analytic control'):resume_adam(path,{},None,None,None,None)


def test_balanced_weighted_design_matches_dense_mixed_loss_and_gradient():
    from scripts.fit_cpu_teacher_heads_lstsq import balanced_sse_weight
    torch.manual_seed(68)
    design=torch.randn(9,3,dtype=torch.float64)
    target=torch.randn(9,dtype=torch.float64)
    # Three groups with very different reference sizes, including zero target
    # rows. No group is silently removed just because its desired edit is zero.
    target[6:]=0
    references=torch.tensor([1.,100.,1e6],dtype=torch.float64).repeat_interleave(3)
    target_sq=target.square().sum();relative_sq=(target.square()/references).sum()
    weights=torch.stack([balanced_sse_weight(target_sq,relative_sq,x) for x in references])
    coeff=torch.randn(3,dtype=torch.float64,requires_grad=True)
    residual=design@coeff-target
    direct=.5*(residual.square().sum()/target_sq+(residual.square()/references).sum()/relative_sq)
    weighted=(residual*weights.sqrt()).square().sum()
    torch.testing.assert_close(direct,weighted,rtol=1e-14,atol=1e-14)
    g1=torch.autograd.grad(direct,coeff,retain_graph=True)[0]
    g2=torch.autograd.grad(weighted,coeff)[0]
    torch.testing.assert_close(g1,g2,rtol=1e-14,atol=1e-14)
    solution,_=solve(design*weights.sqrt()[:,None],target*weights.sqrt())
    assert float(((design@solution-target).square()*weights).sum())<=1.
    for bad in (0.,-1.,float('nan')):
        with pytest.raises(ValueError):balanced_sse_weight(target_sq,relative_sq,bad)
