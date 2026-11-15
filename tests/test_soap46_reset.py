"""A separately selected warmup-history intervention; active sources are immutable."""
import copy
from pathlib import Path
import pytest
import torch

from test_soap46_online import bare,initial_state


def test_activation_reset_consumes_real_credit_then_restarts_only_adam(monkeypatch):
    monkeypatch.setattr(torch.distributed,'all_reduce',lambda _:None)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    torch.manual_seed(8127);model=torch.nn.Linear(7,4,bias=False);p=model.weight
    h=bare(model,lr=1e-4,variant='outer-only');h.warmup_steps=4;h.meta_ramp_steps=3
    h.residual_decay=.1;h.reset_meta_at_activation=True;st=initial_state(p);group={'lr':.0375,'mu':.85}
    for step in range(4):
        p.grad=torch.randn_like(p);h.consume(step);h.propose(p,st,group,step,True,True)
    assert h.meta_steps==2 and not h.meta_reset_done
    assert all(float(s['step'])==2 for s in h.optimizer.state.values())
    assert any(s['exp_avg_sq'].sum()>0 for s in h.optimizer.state.values())
    saved=copy.deepcopy(h.state_dict())
    oracle=bare(model,lr=1e-4,variant='outer-only');oracle.warmup_steps=4;oracle.meta_ramp_steps=3
    oracle.residual_decay=.1;oracle.reset_meta_at_activation=True;oracle.load_state_dict(copy.deepcopy(saved))
    # An independent freshly initialized Adam history with the exact pending
    # action/gradient provides the expected boundary update.
    oracle.reset_meta_at_activation=False;oracle.optimizer.state.clear()
    model_before=p.detach().clone();base_before=copy.deepcopy(st)
    p.grad=torch.randn_like(p);h.consume(4);oracle.consume(4)
    assert torch.equal(p,model_before)
    assert all(torch.equal(v,base_before[k]) if isinstance(v,torch.Tensor) else v==base_before[k] for k,v in st.items())
    assert h.meta_reset_done and h.metrics['meta_adam_reset_this_step']
    assert h.metrics['meta_adam_step']==1 and h.meta_steps==3
    assert h.metrics['meta_effective_lr']==0
    for key in ('xl','xr'):assert torch.count_nonzero(h.entries[p][key])==0
    for a,b in zip(h.optimizer.state.values(),oracle.optimizer.state.values()):
        for key in a:assert torch.equal(a[key],b[key])
    h.propose(p,st,group,4,True,True)
    resumed=bare(model,lr=1e-4,variant='outer-only');resumed.warmup_steps=4;resumed.meta_ramp_steps=3
    resumed.residual_decay=.1;resumed.reset_meta_at_activation=True
    resumed.load_state_dict(copy.deepcopy(h.state_dict()))
    p.grad=torch.randn_like(p);h.consume(5);resumed.consume(5)
    assert h.metrics['meta_adam_step']==2 and not h.metrics['meta_adam_reset_this_step']
    assert h.meta_steps==4 and h.metrics['meta_effective_lr']==pytest.approx(1e-4/3)
    for key in ('xl','xr'):
        assert torch.equal(h.entries[p][key],resumed.entries[p][key])
        assert torch.count_nonzero(h.entries[p][key])>0
    for a,b in zip(h.optimizer.state.values(),resumed.optimizer.state.values()):
        for key in a:assert torch.equal(a[key],b[key])


def test_reset_resume_refuses_policy_change():
    model=torch.nn.Linear(7,4,bias=False);h=bare(model);h.reset_meta_at_activation=True
    other=bare(model)
    with pytest.raises(ValueError,match='reset policy'):other.load_state_dict(h.state_dict())


def test_reset_driver_records_choice_without_extra_gpt_computation(tmp_path):
    from scripts.train_soap46 import patched_source
    source=Path('baseline/record46/upstream.py').read_text()
    actual=patched_source(source,variant='outer-only',meta_lr=1e-5,output=tmp_path,
        steps=20,microbatch=64,warmup_steps=4,meta_ramp_steps=4,reset_meta_at_activation=True)
    compile(actual,'<reset qualification>','exec')
    assert 'reset_meta_at_activation=True' in actual
    assert actual.count('loss.backward()')==source.count('loss.backward()')==1
    assert 'FINAL_TRAIN_STEPS = 2900' in actual
