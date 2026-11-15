from types import SimpleNamespace
import pytest
import torch
from rewriter.soap_freeze import freeze_residual_learning,install_checkpoint_policy,FROZEN_SCHEMA


def hyper():
    h=torch.nn.Parameter(torch.ones(2,2));r=torch.nn.Parameter(torch.eye(2))
    return SimpleNamespace(learned=True,entries={'p':dict(hl=h,hr=r,pending_record='old')},meta_steps=999)


def test_freeze_keeps_coordinates_and_optimizer_clock_and_discards_pending_credit():
    x=hyper();before=[v.detach().clone() for k,v in x.entries['p'].items() if k in ('hl','hr')]
    freeze_residual_learning(x,1000)
    for step in [1000,1001,2200]:
        x.consume(step);x.install_pending(x.entries['p'],object(),1.4455,step)
    assert x.meta_steps==999 and x.metrics['credit_seconds']==0
    assert x.entries['p']['pending_record'] is None
    assert all(torch.equal(a,b) for a,b in zip(before,[x.entries['p']['hl'],x.entries['p']['hr']]))
    # The actual detached scale may still change while H is held fixed.
    assert torch.equal(2*x.entries['p']['hl'],2*before[0])
    with torch.no_grad():x.entries['p']['hl'].add_(1)
    with pytest.raises(RuntimeError,match='mutated'):x.consume(2201)


def test_frozen_checkpoint_requires_explicit_branch_policy():
    mod=SimpleNamespace(restore=lambda scope,x,state:state['completed_steps'],
                        payload=lambda scope,x,step:dict(schema=2,completed_steps=step))
    install_checkpoint_policy(mod,1000)
    x=hyper();assert mod.restore({},x,dict(schema=2,completed_steps=1000))==1000
    state=mod.payload({},x,1500);assert state['schema']==FROZEN_SCHEMA
    y=hyper();assert mod.restore({},y,state)==1500
    with pytest.raises(ValueError,match='declared checkpoint'):mod.restore({},hyper(),dict(schema=2,completed_steps=999))
    with pytest.raises(ValueError,match='mismatch'):mod.restore({},hyper(),dict(state,frozen_at_step=500))


def test_restored_real_hyperoptimizer_freeze_preserves_adam_and_actual_proposal(monkeypatch):
    from rewriter.soap_online import SoapResidualOnline
    from rewriter.soap_residual import FrozenMap,committed_proposal
    from rewriter.lag_one import LagOneHypergradient
    model=torch.nn.Linear(4,3,bias=False).double();p=model.weight
    h=SoapResidualOnline.__new__(SoapResidualOnline)
    h.scope={'model':model,'batch_size':8};h.lr=1e-4;h.stop=2900;h.learned=True;h.map=committed_proposal
    h.entries={};h.optimizer=None;h.meta_steps=999;h.metrics={}
    row={'hl':torch.nn.Parameter(torch.randn(3,3,dtype=torch.float64)*.01),
         'hr':torch.nn.Parameter(torch.randn(4,4,dtype=torch.float64)*.01),'lag':LagOneHypergradient()}
    h.entries[p]=row;h.optimizer=torch.optim.Adam([row['hl'],row['hr']],lr=h.lr)
    for v in (row['hl'],row['hr']):v.grad=torch.randn_like(v)
    h.optimizer.step()
    record=FrozenMap(p.detach().clone(),torch.randn_like(p),torch.eye(3,dtype=torch.float64),torch.eye(4,dtype=torch.float64),
                     torch.tensor(.2,dtype=torch.float64),torch.tensor(.3,dtype=torch.float64),.01,polar_dtype=torch.float64)
    h.install_pending(row,record,1.003,999)
    expected=committed_proposal(record,row['hl'],row['hr']).detach()
    saved_steps=[state['step'].clone() for state in h.optimizer.state.values()]
    freeze_residual_learning(h,1000)
    p.grad=torch.full_like(p,float('nan')) # no GPT-gradient path is consulted after freeze
    h.consume(1000)
    assert torch.equal(expected,committed_proposal(record,row['hl'],row['hr']))
    assert all(torch.equal(a,b['step']) for a,b in zip(saved_steps,h.optimizer.state.values()))
    assert not row['lag'].pending
