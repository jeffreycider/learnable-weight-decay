import torch
from scripts.fit_cpu_grouped_heads_gradient import fit, objective
from scripts.fit_cpu_grouped_heads_gradient import FrozenFeaturePreconditioner
from scripts.fit_cpu_grouped_heads_gradient import merge_designs
from scripts.fit_cpu_grouped_heads_gradient import verify_shared_bodies
from scripts.fit_cpu_edit_teacher import sha
import pytest


def test_cached_map_matches_linear_head_gradient_with_bias_and_output_scaling():
    torch.manual_seed(81)
    features=torch.randn(7,3,dtype=torch.float64)
    joined=torch.cat((features,torch.ones(7,1,dtype=torch.float64)),dim=1)
    head=torch.nn.Linear(3,1,dtype=torch.float64)
    coeff=torch.cat((head.weight.flatten(),head.bias)).detach().requires_grad_()
    target=torch.randn(7,dtype=torch.float64)
    scale=2.3
    groups={'a':dict(A=joined*scale,b=target,weight=.7)}
    cached=objective(groups,{'a':coeff})
    direct=.7*(head(features).flatten()*scale-target).square().sum()
    cached.backward();direct.backward()
    torch.testing.assert_close(cached,direct)
    torch.testing.assert_close(coeff.grad,torch.cat((head.weight.grad.flatten(),head.bias.grad)))


def test_iterative_fit_from_zero_learns_conflicting_groups_and_keeps_zero_target_zero():
    A=torch.eye(3,dtype=torch.float64)
    groups={n:dict(A=A,b=b,weight=.5) for n,b in
            [('positive',torch.tensor([1.,-2.,3.],dtype=torch.float64)),
             ('zero',torch.zeros(3,dtype=torch.float64))]}
    c,state,rows=fit(groups,algorithm='adam',lr=.05,steps=500)
    assert rows[-1]['objective'] < rows[0]['objective']*1e-12
    assert torch.equal(c['zero'],torch.zeros(3,dtype=torch.float64))
    assert int(next(iter(state['state'].values()))['step'])==500


def test_preconditioner_depends_only_on_design_and_matches_damped_dense_curvature():
    torch.manual_seed(5)
    A=torch.randn(4,3,dtype=torch.float64);weight=.7;damping=.01
    groups={'a':dict(A=A,b=torch.randn(4,dtype=torch.float64),weight=weight)}
    pre=FrozenFeaturePreconditioner(groups,damping)
    groups['a']['b'].mul_(100.)
    other=FrozenFeaturePreconditioner(groups,damping)
    for a,b in zip(pre.banks['a'],other.banks['a']):assert torch.equal(a,b)
    c=torch.nn.Parameter(torch.randn(3,dtype=torch.float64));initial=c.detach().clone()
    c.grad=torch.randn_like(c)
    curvature=2*weight*(A.T@A+damping*torch.linalg.svdvals(A)[0].square()*torch.eye(3,dtype=A.dtype))
    expected=initial-.5*torch.linalg.solve(curvature,c.grad)
    pre.step({'a':c},.5)
    torch.testing.assert_close(c,expected)


def test_preconditioning_recovers_ill_conditioned_target_without_lstsq_seed():
    A=torch.diag(torch.tensor([1.,1e-3,2e-6],dtype=torch.float64))
    truth=torch.tensor([1.,-2.,3.],dtype=torch.float64)
    groups={'a':dict(A=A,b=A@truth,weight=.7)}
    coeff,_,rows=fit(groups,algorithm='preconditioned',lr=1.,steps=64)
    assert rows[0]['objective']>.1
    assert rows[-1]['objective']<1e-28
    torch.testing.assert_close(coeff['a'],truth,rtol=0,atol=1e-10)


def test_shared_head_merge_is_mean_of_checkpoint_losses_and_gradients():
    torch.manual_seed(19)
    datasets=[dict(groups={'a':dict(A=torch.randn(rows,3,dtype=torch.float64),
                   b=torch.randn(rows,dtype=torch.float64),weight=weight)})
              for rows,weight in [(2,.1),(5,7.)]]
    c={'a':torch.randn(3,dtype=torch.float64,requires_grad=True)}
    separate=sum(objective(d['groups'],c) for d in datasets)/2
    merged=objective(merge_designs(datasets),c)
    torch.testing.assert_close(merged,separate)
    torch.testing.assert_close(torch.autograd.grad(merged,c['a'])[0],torch.autograd.grad(separate,c['a'])[0])
    datasets[1]['groups']['a']['A']=torch.randn(5,4,dtype=torch.float64)
    with pytest.raises(ValueError,match='incompatible'):merge_designs(datasets)


def test_cross_checkpoint_fit_requires_identical_body_and_distinct_states(tmp_path):
    datasets=[]
    for i in range(2):
        path=tmp_path/f'{i}.pt'
        torch.save(dict(actor={'trunk':torch.ones(3),'matrix_output.weight':torch.tensor([float(i)])},
                        metadata={'state_sha256':str(i),'architecture':{'name':'test'}}),path)
        datasets.append(dict(checkpoint=str(path),source_sha256={str(path):sha(path)}))
    assert len(verify_shared_bodies(datasets))==2
    with pytest.raises(ValueError,match='duplicate'):verify_shared_bodies([datasets[0],datasets[0]])
    path=tmp_path/'1.pt';saved=torch.load(path,weights_only=False);saved['actor']['trunk'].zero_();torch.save(saved,path)
    datasets[1]['source_sha256'][str(path)]=sha(path)
    with pytest.raises(ValueError,match='bodies differ'):verify_shared_bodies(datasets)


def test_shared_body_identity_rejects_equal_values_at_different_precision(tmp_path):
    datasets=[]
    for i,dtype in enumerate((torch.float32,torch.float64)):
        path=tmp_path/f'precision{i}.pt'
        torch.save(dict(actor={'trunk':torch.ones(3,dtype=dtype)},
                        metadata={'state_sha256':str(i),'architecture':{'name':'test'}}),path)
        datasets.append(dict(checkpoint=str(path),source_sha256={str(path):sha(path)}))
    with pytest.raises(ValueError,match='body dtypes differ'):
        verify_shared_bodies(datasets)
