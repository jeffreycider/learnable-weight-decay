"""Optional actual-base-motion budget: lifecycle, exact restart and authority."""
import copy

import pytest
import torch

from rewriter.transformer_action import prepare_transformer_action
from rewriter.transformer_online import TransformerOnlineLifecycle, TransformerChiselAdapter, displacement_norms
from tests.test_transformer_action import FactorActor, fixture
from tests.test_transformer_online import task_gradients, _adapter_fixture, _ordinary_step
from tests.test_online_norm_lifecycle import assert_tree


def make(fraction=.25):
    base,snapshot=fixture(torch.float64)
    actor=FactorActor()
    lifecycle=TransformerOnlineLifecycle(actor,torch.optim.Adam(actor.parameters(),lr=1e-4,betas=(.9,.99)),
        global_token_count=8192,final_action_step=1003,radius_fraction=.001,gradient_clip=100.,
        action_budget=None if fraction is None else dict(mode='base_motion',fraction=fraction))
    # Production materialization intentionally uses base.float(); represent FP32
    # stored values exactly in this FP64 VJP fixture, avoiding baseline rounding.
    return lifecycle,{n:p.float().double().requires_grad_() for n,p in base.items()},snapshot


@pytest.mark.parametrize('motion',[0.,.001,100.])
def test_cap_dense_vjp_and_exact_pending_restart(motion):
    online,live,snapshot=make()
    statistic=torch.tensor(motion,dtype=torch.float64)
    online.bind_action_budget(statistic,action_step=1000)
    statistic.fill_(777.)
    online.commit(snapshot,live,action_step=1000)
    action=online.action
    actual=sum((live[n]-action.parameters[n]).square().sum() for n in live).sqrt()
    assert actual<=motion*.25+1e-12
    assert action.action_budget['base_update_norm'].item()==motion
    proposal=action.materialize()
    targets={n:torch.sin(p.detach()) for n,p in proposal.items()}
    expected,=torch.autograd.grad(sum(.5*(p-targets[n]).square().sum() for n,p in proposal.items()),online.actor.x,retain_graph=True)
    if motion==0:
        assert actual==0 and torch.equal(expected,torch.zeros_like(expected))
    saved=copy.deepcopy(online.state_dict())
    assert saved['schema']==3 and saved['pending']['action_budget']['action_step']==1000
    restarted,restored,_=make()
    with torch.no_grad():
        for n,p in restored.items():p.copy_(live[n])
    rng=torch.get_rng_state().clone();restarted.load_state_dict(saved,restored)
    assert torch.equal(rng,torch.get_rng_state())
    a=online.consume(task_gradients(live),gradient_step=1001)
    b=restarted.consume(task_gradients(restored),gradient_step=1001)
    assert a==b
    torch.testing.assert_close(online.actor.x.grad,expected,rtol=1e-12,atol=1e-12)
    assert_tree(online.state_dict(),restarted.state_dict())


def test_disabled_and_nonbinding_budget_preserve_proposal_and_credit():
    off,live,snapshot=make(None);on,other,snapshot2=make()
    on.bind_action_budget(100.,action_step=1000)
    off.commit(snapshot,live,action_step=1000);on.commit(snapshot2,other,action_step=1000)
    for n in live:torch.testing.assert_close(live[n],other[n],rtol=0,atol=0)
    assert off.state_dict()['schema']==1 and 'action_budget' not in off.config
    assert_tree(off.action.snapshot,on.action.snapshot)
    assert 'action_budget' not in off.state_dict()['pending']
    off.consume(task_gradients(live),gradient_step=1001);on.consume(task_gradients(other),gradient_step=1001)
    torch.testing.assert_close(off.actor.x,on.actor.x,rtol=0,atol=0)
    assert_tree(off.optimizer.state_dict(),on.optimizer.state_dict())


def test_budget_bind_clock_resume_and_missing_replay_hook_fail_closed():
    online,live,snapshot=make()
    with pytest.raises(ValueError,match='not bound'):online.commit(snapshot,live,action_step=1000)
    online.bind_action_budget(.001,action_step=1000)
    with pytest.raises(ValueError,match='already bound'):online.bind_action_budget(.002,action_step=1000)
    base={n:p.detach().clone() for n,p in live.items()}
    unbudgeted=prepare_transformer_action(online.actor,snapshot,base,radius_fraction=.001)
    with pytest.raises(ValueError,match='budget differs'):online.commit_prepared(unbudgeted,live,action_step=1000)
    online.commit(snapshot,live,action_step=1000)
    saved=copy.deepcopy(online.state_dict())
    for change in ['radius','clock','fraction','missing']:
        bad=copy.deepcopy(saved);budget=bad['pending']['action_budget']
        if change=='radius':budget['radius']*=2
        elif change=='clock':budget['action_step']+=1
        elif change=='fraction':budget['fraction']=.5;budget['radius']*=2
        else:bad['pending'].pop('action_budget')
        restored,_,_=make()
        with pytest.raises(ValueError):restored.load_state_dict(bad,live)
    disabled,_,_=make(None)
    with pytest.raises(ValueError,match='configuration'):disabled.load_state_dict(saved,live)


def test_full_actual_displacement_includes_uneditable_parameters():
    base,_=fixture(torch.float64);after={n:p.clone() for n,p in base.items()}
    after['norm1.gains'].add_(2.)
    norms=displacement_norms(after,base)
    assert norms['all_parameters']==4. and norms['grouped']==0.
    online,_,_=make();online.bind_action_budget(norms['all_parameters'],action_step=1000)
    assert online.current_action_budget['radius']==1.


def test_restore_no_pending_boundary_clears_used_instance_budget():
    online,live,snapshot=make()
    initial=copy.deepcopy(online.state_dict())
    online.bind_action_budget(.01,action_step=1001)
    online.commit(snapshot,live,action_step=1001)
    online.consume(task_gradients(live),gradient_step=1002)
    assert online.action is None and online.current_action_budget['action_step']==1001
    online.load_state_dict(initial,live)
    assert online.current_action_budget is None and online.last_consumed_step is None
    online.bind_action_budget(.002,action_step=1000)
    online.commit(snapshot,live,action_step=1000)
    assert online.action.action_budget['action_step']==1000


def test_norm_candidate_and_final_reuse_same_budget_and_restart():
    from tests.test_online_norm_lifecycle import AUX
    online,live,snapshot=make()
    # Construct the public lifecycle with both optional settings enabled.
    online=TransformerOnlineLifecycle(online.actor,online.optimizer,global_token_count=8192,
        final_action_step=1003,radius_fraction=.001,gradient_clip=100.,norm_auxiliary=AUX,
        action_budget=dict(mode='base_motion',fraction=.25))
    online.bind_action_budget(.001,action_step=1000)
    online.commit(snapshot,live,action_step=1000)
    assert online.norm_updates==1
    assert online.action.action_budget['radius']==.00025
    saved=copy.deepcopy(online.state_dict())
    restarted,_,_=make()
    restarted=TransformerOnlineLifecycle(restarted.actor,restarted.optimizer,global_token_count=8192,
        final_action_step=1003,radius_fraction=.001,gradient_clip=100.,norm_auxiliary=AUX,
        action_budget=dict(mode='base_motion',fraction=.25))
    restarted.load_state_dict(saved,live)
    assert_tree(online.state_dict(),restarted.state_dict())
    assert online.consume(task_gradients(live),gradient_step=1001)==restarted.consume(task_gradients(live),gradient_step=1001)
    assert_tree(online.optimizer.state_dict(),restarted.optimizer.state_dict())


@pytest.mark.parametrize('base_lr',[0.,.001])
def test_actual_adapter_binds_complete_stored_base_motion_and_terminal(base_lr):
    model,opt,old,config=_adapter_fixture()
    actor,actoropt=old.lifecycle.actor,old.lifecycle.optimizer;old.close()
    config['transformer_chisel']['action_budget']=dict(mode='base_motion',fraction=.0001)
    adapter=TransformerChiselAdapter(model,[opt],config,mode='learned',actor=actor,actor_optimizer=actoropt)
    model._fp32_sensors=adapter.collector
    opt.param_groups[0]['lr']=base_lr
    try:
        for t in range(4):
            _ordinary_step(model,opt,adapter,t)
            budget=adapter.lifecycle.current_action_budget
            assert budget['action_step']==t
            if t<3:
                assert budget['base_update_norm'].item()==adapter.metrics['chisel_base_displacement_norm_all_parameters']
                # Storage rounding may change the real norm slightly.
                assert adapter.metrics['chisel_realized_edit_norm_grouped']<=float(budget['radius'])+1e-5
                if base_lr==0:
                    assert budget['radius']==0 and adapter.metrics['chisel_realized_edit_norm_grouped']==0
            else:
                assert adapter.metrics['terminal_identity'] and adapter.lifecycle.action is None
    finally:adapter.close()
