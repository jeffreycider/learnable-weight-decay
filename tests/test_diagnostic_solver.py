import dataclasses
from unittest.mock import patch
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from rewriter.algebra import Anchors, Measurements, auxiliary_objective
from rewriter.controller import ControllerSnapshot, MatrixObservation, VectorObservation
from rewriter.diagnostics import solve_auxiliary_quadratic


def fixture(shapes=((7, 5), (4, 11)), rank=2, probes=4, dtype=torch.float64):
    gen = torch.Generator().manual_seed(135)
    matrices, banks = {}, {}
    norm_sq = torch.zeros((), dtype=dtype)
    for i, (n,m) in enumerate(shapes):
        a = Anchors.random(n,m,rank,generator=gen,dtype=dtype)
        w = torch.randn(n,m,generator=gen,dtype=dtype)
        g = torch.randn(probes,n,m,generator=gen,dtype=dtype)
        matrices[str(i)] = MatrixObservation(a, torch.empty(0), torch.empty(0), (w @ a.B,w.T @ a.A))
        banks[str(i)] = (g @ a.B,g.transpose(-1,-2) @ a.A)
        norm_sq += w.square().sum()
    v = torch.randn(3,generator=gen,dtype=dtype)
    vg = torch.randn(probes,3,generator=gen,dtype=dtype)
    norm_sq += v.square().sum()
    snapshot = ControllerSnapshot(matrices,{'gain':VectorObservation(torch.empty(0),v)},
        torch.empty(0),norm_sq.sqrt(),torch.tensor(.7,dtype=dtype))
    return snapshot,Measurements(banks,{'gain':vg})


def dense_coordinates(snapshot, measured, action=None):
    """Independent tiny-test orthonormal complement parameterization."""
    weights, banks, values = [], [], []
    for name, obs in snapshot.matrices.items():
        a = obs.anchors.A
        complement = torch.linalg.qr(a, mode='complete').Q[:,a.shape[1]:]
        wp,wq = obs.weight_probes
        zp,zq = measured.matrices[name]
        weights += [(complement.T @ wp).flatten(), wq.flatten()]
        banks += [(complement.T @ zp).flatten(1),zq.flatten(1)]
        if action is not None:
            f=action.matrices[name]
            values += [(complement.T @ f.P).flatten(),f.Q.flatten()]
    for name, obs in snapshot.vectors.items():
        weights.append(obs.weight.flatten());banks.append(measured.vectors[name].flatten(1))
        if action is not None:values.append(action.vectors[name].flatten())
    return torch.cat(weights),torch.cat(banks,dim=1),torch.cat(values) if values else None


@pytest.mark.parametrize('beta,gamma,budget', [(1.,.01,.001),(0.,0.,1.),(.5,.3,.2)])
def test_woodbury_matches_independent_dense_canonical_solve(beta,gamma,budget):
    snapshot,train=fixture()
    action,info=solve_auxiliary_quadratic(snapshot,train,beta=beta,gamma=gamma,loss_budget=budget)
    w,z,x=dense_coordinates(snapshot,train,action)
    radius=snapshot.radius
    c=2*snapshot.weight_norm*radius+radius.square()+1e-24
    lam=1/c+gamma/(radius.square()+1e-24)
    alpha=beta/(z.shape[0]*budget**2)
    expected=torch.linalg.solve(lam*torch.eye(len(w),dtype=w.dtype)+alpha*z.T@z,-w/c)
    # The explicit D-by-D solve is less stable than the K-by-K solve at beta/budget^2=1e6.
    torch.testing.assert_close(x,expected,rtol=2e-7,atol=2e-8)
    objective=auxiliary_objective(action,snapshot.weights,snapshot.vector_weights,train,
        weight_norm=snapshot.weight_norm,radius=radius,beta=beta,gamma=gamma,loss_budget=budget)
    assert info['objective']==pytest.approx(float(objective),rel=1e-10,abs=1e-10)
    assert info['canonical_accessible_dimension']==len(x)
    assert info['stored_action_coordinates']>len(x)
    assert info['objective']<=0
    assert info['small_system_relative_residual']<1e-12
    assert info['max_anchor_constraint_error']<1e-12
    assert info['diagnostic_only'] and info['extra_gpt_evaluations']==0


def test_stationarity_by_independent_autograd_and_random_perturbations():
    snapshot,train=fixture()
    action,info=solve_auxiliary_quadratic(snapshot,train,beta=.7,gamma=.1,loss_budget=.2)
    w,z,x=dense_coordinates(snapshot,train,action)
    x=x.detach().requires_grad_(True)
    c=2*snapshot.weight_norm*snapshot.radius+snapshot.radius.square()+1e-24
    def objective(t):
        return ((2*w.dot(t)+t.square().sum())/c
                +.7*(z@t).square().mean()/.2**2
                +.1*t.square().sum()/(snapshot.radius.square()+1e-24))
    grad=torch.autograd.grad(objective(x),x)[0]
    assert float(grad.norm())<1e-10
    assert info['stationarity_relative_norm']<1e-10
    generator=torch.Generator().manual_seed(99)
    for _ in range(5):
        candidate=x.detach()+.1*torch.randn(x.shape,generator=generator,dtype=x.dtype)
        assert objective(candidate)>objective(x)


def test_global_cross_layer_cancellation_survives_solver():
    dtype=torch.float64
    snapshot=ControllerSnapshot({}, {
        'first':VectorObservation(torch.empty(0),torch.tensor([1.],dtype=dtype)),
        'second':VectorObservation(torch.empty(0),torch.tensor([-1.],dtype=dtype))},
        torch.empty(0),torch.tensor(2.,dtype=dtype).sqrt(),torch.tensor(1.,dtype=dtype))
    train=Measurements({}, {name:torch.ones(4,1,dtype=dtype) for name in snapshot.vectors})
    action,info=solve_auxiliary_quadratic(snapshot,train,beta=1.,gamma=0.,loss_budget=.001)
    torch.testing.assert_close(action.vectors['first'],torch.tensor([-1.],dtype=dtype))
    torch.testing.assert_close(action.vectors['second'],torch.tensor([1.],dtype=dtype))
    assert info['train_proxy']==0
    assert sum((train.vectors[name]*v).sum(-1).square().mean() for name,v in action.vectors.items())==2


def test_fp32_inputs_are_promoted_and_duplicate_probes_are_stable():
    snapshot,train=fixture(dtype=torch.float32)
    train=Measurements({n:(a[:1].expand_as(a).clone(),b[:1].expand_as(b).clone()) for n,(a,b) in train.matrices.items()},
                       {n:v[:1].expand_as(v).clone() for n,v in train.vectors.items()})
    action,info=solve_auxiliary_quadratic(snapshot,train,loss_budget=.01)
    assert all(f.P.dtype==torch.float64 for f in action.matrices.values())
    assert all(not f.P.requires_grad for f in action.matrices.values())
    assert info['small_system_relative_residual']<1e-12
    assert info['stationarity_relative_norm']<1e-7
    assert info['max_anchor_constraint_error']<1e-5
    assert info['objective']<0


def test_full_anchor_rank_spans_matrix_and_zero_row_factor():
    snapshot,train=fixture(shapes=((2,5),),rank=2)
    action,info=solve_auxiliary_quadratic(snapshot,train,beta=0.,gamma=0.)
    assert action.matrices['0'].P.norm()<1e-13
    assert info['canonical_accessible_dimension']==2*5+3
    w,z,x=dense_coordinates(snapshot,train,action)
    torch.testing.assert_close(x,-w)


def test_solver_never_materializes_gpt_matrices_or_projectors():
    snapshot,train=fixture(shapes=((13,11),),probes=3)
    class NoDense(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            result=func(*args,**(kwargs or {}))
            def check(x):
                if isinstance(x,torch.Tensor):
                    assert tuple(x.shape) not in {(13,11),(13,13),(11,11),(47,47)}
                elif isinstance(x,(tuple,list)):
                    for y in x:check(y)
            check(result)
            return result
    with NoDense(),patch('rewriter.algebra.Factors.dense',side_effect=AssertionError('dense GPT action')):
        action,info=solve_auxiliary_quadratic(snapshot,train)
    assert info['training_probes']==3


def test_zero_cached_weights_returns_identity_without_mutating_inputs():
    snapshot,train=fixture()
    matrices={n:dataclasses.replace(obs,weight_probes=tuple(torch.zeros_like(w) for w in obs.weight_probes)) for n,obs in snapshot.matrices.items()}
    vectors={n:dataclasses.replace(obs,weight=torch.zeros_like(obs.weight)) for n,obs in snapshot.vectors.items()}
    snapshot=dataclasses.replace(snapshot,matrices=matrices,vectors=vectors,weight_norm=torch.tensor(0.,dtype=torch.float64))
    old={n:tuple(x.clone() for x in pair) for n,pair in train.matrices.items()}
    action,info=solve_auxiliary_quadratic(snapshot,train)
    assert action.norm_sq()==0 and info['objective']==0
    assert info['stationarity_relative_norm']==0
    for name,pair in train.matrices.items():
        for x,y in zip(pair,old[name]):torch.testing.assert_close(x,y,atol=0,rtol=0)


@pytest.mark.parametrize('kwargs', [{'beta':-1},{'gamma':-1},{'loss_budget':0},{'loss_budget':float('nan')},{'squared_norm_floor':0}])
def test_invalid_objective_coefficients_fail_closed(kwargs):
    snapshot,train=fixture()
    with pytest.raises(ValueError):solve_auxiliary_quadratic(snapshot,train,**kwargs)


def test_malformed_and_nonorthonormal_banks_are_rejected():
    snapshot,train=fixture()
    with pytest.raises(ValueError,match='exactly'):
        solve_auxiliary_quadratic(snapshot,Measurements({},train.vectors))
    bad=dict(train.vectors);bad['gain']=bad['gain'][:1]
    with pytest.raises(ValueError,match='same probe count'):
        solve_auxiliary_quadratic(snapshot,Measurements(train.matrices,bad))
    matrices=dict(snapshot.matrices);obs=matrices['0']
    matrices['0']=dataclasses.replace(obs,anchors=Anchors(obs.anchors.A*2,obs.anchors.B))
    with pytest.raises(ValueError,match='orthonormal'):
        solve_auxiliary_quadratic(dataclasses.replace(snapshot,matrices=matrices),train)
