"""Alternating norm Adam is opt-in and precedes the fresh committed graph."""
import copy
from dataclasses import is_dataclass, fields
import pytest
import torch
from tests.test_transformer_action import fixture,FactorActor
from tests.test_transformer_online import task_gradients
from rewriter.transformer_online import TransformerOnlineLifecycle
from rewriter.transformer_action import prepare_transformer_action
from rewriter.online_norm import continuous_group_norm_objective,validate_norm_auxiliary

# Test fixture values only, not experiment settings.
AUX=dict(mode='alternating_continuous_l2',coefficient=.2,learning_rate=2e-4,
         representation_lr_multiplier=3.,gradient_clip=10.)


def make(aux=None,forward=None):
    base,snapshot=fixture(torch.float64);actor=FactorActor()
    opt=torch.optim.Adam(actor.parameters(),lr=1e-4,betas=(.9,.99))
    life=TransformerOnlineLifecycle(actor,opt,global_token_count=8192,final_action_step=1003,
        radius_fraction=.001,gradient_clip=100.,norm_auxiliary=aux,
        actor_forward=(lambda snap:forward(actor,snap)) if forward else None,
        actor_execution={'backend':'test'} if forward else None)
    return life,{n:p.clone().requires_grad_() for n,p in base.items()},snapshot


def assert_tree(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif is_dataclass(a):
        for f in fields(a):assert_tree(getattr(a,f.name),getattr(b,f.name))
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:assert_tree(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_tree(x,y)
    else:assert a==b


def test_explicit_off_is_exact_default_trajectory_and_state():
    a,wa,sa=make();b,wb,sb=make({'mode':'off'})
    for t in range(1000,1004):
        if t>1000:
            assert_tree(a.consume(task_gradients(wa),gradient_step=t),b.consume(task_gradients(wb),gradient_step=t))
        assert_tree(a.commit(sa,wa,action_step=t),b.commit(sb,wb,action_step=t))
        assert_tree(wa,wb);assert_tree(a.state_dict(),b.state_dict())
    assert a.state_dict()['schema']==1 and 'norm_updates' not in a.state_dict()


def test_aux_adam_reference_second_tick_and_regenerated_final_graph():
    calls=[]
    def record(actor,snapshot):calls.append(actor.x.detach().clone());return actor(snapshot)
    life,live,snapshot=make(AUX,record)
    actor=FactorActor();opt=torch.optim.Adam(actor.parameters(),lr=1e-4,betas=(.9,.99))
    base={n:p.detach().clone() for n,p in live.items()}
    candidate=prepare_transformer_action(actor,snapshot,base,radius_fraction=.001)
    obj,_=continuous_group_norm_objective(candidate);opt.zero_grad();(AUX['coefficient']*obj).backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(),AUX['gradient_clip']);opt.param_groups[0]['lr']=AUX['learning_rate'];opt.step();opt.param_groups[0]['lr']=1e-4
    expected=prepare_transformer_action(actor,snapshot,base,radius_fraction=.001).materialize()
    metrics=life.commit(snapshot,live,action_step=1000)
    assert len(calls)==2 and not torch.equal(calls[0],calls[1])
    assert_tree(live,expected);assert_tree(life.actor.x,actor.x);assert_tree(life.optimizer.state_dict(),opt.state_dict())
    assert life.optimizer.state[life.actor.x]['step']==1
    assert metrics['chisel_norm_auxiliary_actor_backward_calls']==1
    life.consume(task_gradients(live),gradient_step=1001)
    assert life.optimizer.state[life.actor.x]['step']==2
    life.commit(snapshot,live,action_step=1001)
    assert life.optimizer.state[life.actor.x]['step']==3
    assert life.meta_updates==1 and life.norm_updates==2
    assert len(calls)==4
    assert life.optimizer.param_groups[0]['lr']==1e-4


def test_pending_resume_does_not_repeat_norm_step_and_terminal_only_consumes_task():
    a,live,snapshot=make(AUX);a.commit(snapshot,live,action_step=1000);saved=a.state_dict()
    b,restored,sb=make(AUX)
    with torch.no_grad():
        for n,p in restored.items():p.copy_(live[n])
    b.load_state_dict(saved,restored)
    assert b.norm_updates==1 and b.optimizer.state[b.actor.x]['step']==1
    for t in range(1001,1004):
        a.consume(task_gradients(live),gradient_step=t);b.consume(task_gradients(restored),gradient_step=t)
        ma=a.commit(snapshot,live,action_step=t);mb=b.commit(sb,restored,action_step=t)
        assert_tree(live,restored);assert_tree(a.state_dict(),b.state_dict())
    assert ma['terminal_identity'] and mb['terminal_identity']
    assert a.norm_updates==a.meta_updates==3
    assert a.optimizer.state[a.actor.x]['step']==6 and not a.lag.pending


def test_pending_actor_mutation_config_and_clock_fail_closed():
    a,live,snapshot=make(AUX);a.commit(snapshot,live,action_step=1000)
    old=a.actor.x.detach().clone()
    with pytest.raises(RuntimeError,match='pending'):a.commit(snapshot,live,action_step=1001)
    assert_tree(a.actor.x,old)
    saved=a.state_dict();bad=copy.deepcopy(saved);bad['norm_updates']+=1
    b,_,_=make(AUX)
    with pytest.raises(ValueError,match='clocks'):b.load_state_dict(bad,live)
    b,_,_=make({**AUX,'learning_rate':3e-4})
    with pytest.raises(ValueError,match='configuration'):b.load_state_dict(saved,live)
    b,_,_=make()
    with pytest.raises(ValueError,match='configuration'):b.load_state_dict(saved,live)
    with torch.no_grad():a.actor.x.add_(.1)
    with pytest.raises(RuntimeError,match='mutated'):a.consume(task_gradients(live),gradient_step=1001)


@pytest.mark.parametrize('patch',[{'coefficient':0},{'learning_rate':float('nan')},{'mode':'joint_next_task'},{'gradient_clip':True}])
def test_invalid_settings_rejected(patch):
    with pytest.raises(ValueError):validate_norm_auxiliary({**AUX,**patch})


def test_actual_transformer_split_adam_rates_restore_after_auxiliary():
    from rewriter.transformer_chisel import TransformerChisel
    from rewriter.meta_optimizer import build_meta_adam
    base,snapshot=fixture();torch.manual_seed(824)
    actor=TransformerChisel(rank=2,trajectory_dim=12,model_dim=128,num_layers=1,patch_size=4)
    with torch.no_grad():
        for head in actor.output_heads:head.weight.normal_(std=.005)
    optimizer=build_meta_adam(actor,lr=1e-6,representation_lr_multiplier=10.)
    task_lrs=[g['lr'] for g in optimizer.param_groups]
    life=TransformerOnlineLifecycle(actor,optimizer,global_token_count=8192,final_action_step=1002,
                                   radius_fraction=.01,norm_auxiliary=AUX)
    live={n:p.clone().requires_grad_() for n,p in base.items()}
    metrics=life.commit(snapshot,live,action_step=1000)
    assert metrics['chisel_norm_auxiliary_learning_rates']==[AUX['learning_rate'],AUX['learning_rate']*3]
    assert [g['lr'] for g in optimizer.param_groups]==task_lrs
    assert max(float(st['step']) for st in optimizer.state.values())==1
    life.consume(task_gradients(live),gradient_step=1001)
    life.commit(snapshot,live,action_step=1001)
    assert max(float(st['step']) for st in optimizer.state.values())==3
    assert [g['lr'] for g in optimizer.param_groups]==task_lrs
    assert all(torch.isfinite(p).all() for p in live.values())
