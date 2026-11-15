import copy
import pytest
import torch
from rewriter.transformer_chisel import TransformerChisel
from rewriter.meta_optimizer import build_meta_adam,set_meta_learning_rate,meta_learning_rates


def actor():
    torch.manual_seed(123)
    return TransformerChisel(rank=2,trajectory_dim=12,model_dim=128,num_layers=1,patch_size=8)


def gradients(model,step):
    for index,p in enumerate(model.parameters()):p.grad=torch.full_like(p,(index+1)*(step+1)*.001)


def test_default_optimizer_is_the_original_adam():
    a=actor();b=copy.deepcopy(a)
    actual=build_meta_adam(a,lr=2e-6)
    reference=torch.optim.Adam(b.parameters(),lr=2e-6,betas=(.9,.99),eps=1e-8)
    for step in range(3):
        gradients(a,step);gradients(b,step);actual.step();reference.step()
    assert all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))
    assert actual.state_dict()['param_groups']==reference.state_dict()['param_groups']


def test_representation_split_preserves_head_lr_and_schedule_ratio():
    model=actor();opt=build_meta_adam(model,lr=2e-6,representation_lr_multiplier=10.)
    heads={id(p) for head in model.output_heads for p in head.parameters()}
    assert {id(p) for p in opt.param_groups[0]['params']}==heads
    ids=[id(p) for group in opt.param_groups for p in group['params']]
    assert len(ids)==len(set(ids))==len(list(model.parameters()))
    assert set(ids)=={id(p) for p in model.parameters()}
    assert meta_learning_rates(opt)==pytest.approx({'action_heads':2e-6,'representation':2e-5})
    set_meta_learning_rate(opt,1e-6)
    assert meta_learning_rates(opt)==pytest.approx({'action_heads':1e-6,'representation':1e-5})


def test_split_optimizer_resume_preserves_moments_counters_and_lr_ratio():
    a=actor();oa=build_meta_adam(a,lr=2e-6,representation_lr_multiplier=10.)
    gradients(a,0);oa.step();set_meta_learning_rate(oa,1e-6)
    b=copy.deepcopy(a);ob=build_meta_adam(b,lr=2e-6,representation_lr_multiplier=10.)
    ob.load_state_dict(copy.deepcopy(oa.state_dict()))
    gradients(a,1);gradients(b,1);oa.step();ob.step()
    assert all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))
    assert all(float(state['step'])==2 for state in ob.state.values())
    assert meta_learning_rates(ob)==meta_learning_rates(oa)


@pytest.mark.parametrize('value',[0.,-1.,float('nan'),float('inf')])
def test_bad_representation_lr_fails_closed(value):
    with pytest.raises(ValueError,match='positive finite'):
        build_meta_adam(actor(),lr=2e-6,representation_lr_multiplier=value)
