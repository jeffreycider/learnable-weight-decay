"""Private adaptation must not erase earlier domains' accumulated shared grads."""
import copy
from types import SimpleNamespace
import pytest
import torch
from scripts.train_coordinate_chisel import adapt_private,validate_recipe
from tests.test_coordinate_training_driver import recipe


class Contexts:
    def __init__(self,fail=False):self.calls=0;self.fail=fail
    def measure(self,context,actor,private,config,dual,*,learning,weight):
        self.calls+=1
        assert learning and weight==1.
        assert not any(p.requires_grad for p in actor.parameters())
        if self.fail:raise RuntimeError('injected fitting failure')
        loss=(actor(private(torch.ones(2,2)))-1).square().mean()
        loss.backward()
        return dict(local_id=1,mean_kl=float(loss.detach()))


def test_private_inner_preserves_shared_weights_mixed_flags_and_accumulated_grads():
    torch.manual_seed(9)
    actor=torch.nn.Linear(2,1);actor.bias.requires_grad_(False)
    private=torch.nn.Linear(2,2);optimizer=torch.optim.Adam(private.parameters(),lr=.01)
    actor.weight.grad=torch.randn_like(actor.weight)
    before=copy.deepcopy(actor.state_dict());grad=actor.weight.grad.clone()
    private_before=private.weight.detach().clone();calls=[]
    dual=SimpleNamespace(step=lambda rows,expected_ids:calls.append((rows,expected_ids)))
    contexts=Contexts()
    rows=adapt_private(contexts,{},actor,private,optimizer,None,dual,steps=3,clip=1.)
    assert len(rows)==contexts.calls==len(calls)==3
    assert not torch.equal(private.weight,private_before)
    assert all(torch.equal(actor.state_dict()[k],v) for k,v in before.items())
    assert torch.equal(actor.weight.grad,grad)
    assert actor.weight.requires_grad and not actor.bias.requires_grad
    assert all(int(s['step'])==3 for s in optimizer.state.values())
    with pytest.raises(RuntimeError,match='injected'):
        adapt_private(Contexts(fail=True),{},actor,private,optimizer,None,dual,steps=1,clip=1.)
    assert actor.weight.requires_grad and not actor.bias.requires_grad
    assert torch.equal(actor.weight.grad,grad)


@pytest.mark.parametrize('value',[-1,1.5,True,'4'])
def test_private_inner_budget_explicit_nonnegative_integer(value):
    config=recipe();config['private_inner_steps']=value
    with pytest.raises(ValueError,match='private_inner_steps'):validate_recipe(config)


def test_zero_private_inner_is_noop():
    assert adapt_private(None,None,None,None,None,None,None,steps=0,clip=1.)==[]


def test_reused_real_engine_counter_accounting_is_linear():
    from scripts.train_coordinate_chisel import measure_context
    counter={'student_forwards':0};total={}
    class C:
        compute=total
        def measure(self,context):
            counter['student_forwards']+=1
            for k,v in counter.items():self.compute[k]=self.compute.get(k,0)+v
            return {}
    context={'domain':'gpt','engine':SimpleNamespace(compute=SimpleNamespace(as_dict=lambda:dict(counter)))}
    for _ in range(5):measure_context(C(),context)
    assert total=={'student_forwards':5}


def test_actor_only_import_checks_pin_and_does_not_import_optimizer(tmp_path):
    from scripts.train_coordinate_chisel import initialize_shared_actor,build_shared_adam,set_private_lr
    from rewriter.coordinate_chisel import CoordinateChisel
    from rewriter.multibank import file_sha
    cfg=recipe();actor=CoordinateChisel(**cfg['architecture'])
    saved=copy.deepcopy(actor.state_dict());saved['matrix_output.weight'].fill_(.25)
    path=tmp_path/'parent.pt';torch.save(dict(actor=saved,update=1024,shared_optimizer={'must_not_load':True}),path)
    cfg['shared_initialization']=dict(mode='checkpoint_actor_only',path=str(path),sha256=file_sha(path),update=1024)
    initialize_shared_actor(actor,cfg)
    assert torch.equal(actor.matrix_output.weight,saved['matrix_output.weight'])
    opt=build_shared_adam(actor,cfg['shared_optimizer']);assert not opt.state
    cfg['shared_initialization']['sha256']='0'*64
    with pytest.raises(ValueError,match='SHA'):initialize_shared_actor(actor,cfg)
    schedule=dict(lr=1e-4,tick_schedule=dict(cold_steps=64,warm_lr=1e-5))
    assert set_private_lr(opt,schedule,63)==1e-4
    assert set_private_lr(opt,schedule,64)==1e-5
    assert set_private_lr(opt,schedule,64)==1e-5


def test_private_actor_backend_explicit_and_default_off():
    config=recipe();validate_recipe(config)
    config['private_actor_backend']='unknown'
    with pytest.raises(ValueError,match='private actor backend'):validate_recipe(config)
    config['private_actor_backend']='eager_actor_graph'
    with pytest.raises(ValueError,match='inner fitting'):validate_recipe(config)
    config['private_inner_steps']=2;validate_recipe(config)


def test_private_dispatch_sees_frozen_shared_and_selected_panel():
    actor=torch.nn.Linear(2,1);private=torch.nn.Linear(2,2)
    optimizer=torch.optim.Adam(private.parameters(),lr=.01)
    actor.weight.grad=torch.ones_like(actor.weight);before=actor.weight.grad.clone();seen=[]
    class Dispatch:
        def for_visit(self,domain,context,p):
            assert domain=='gpt' and p is private and not any(x.requires_grad for x in actor.parameters())
            seen.append(context['index']);return actor
    dual=SimpleNamespace(step=lambda *args,**kwargs:None)
    adapt_private(Contexts(),{},actor,private,optimizer,None,dual,steps=2,clip=1.,
        inner_contexts=[dict(domain='gpt',index=i) for i in range(2)],actor_dispatch=Dispatch())
    assert seen==[0,1] and actor.weight.requires_grad
    assert torch.equal(actor.weight.grad,before)
