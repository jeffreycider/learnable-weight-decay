import copy

import pytest
import torch

from tests.test_transformer_action import FactorActor, fixture
from rewriter.transformer_online import TransformerOnlineLifecycle


def make():
    base, snapshot = fixture(torch.float64)
    actor = FactorActor()
    optimizer = torch.optim.Adam(actor.parameters(),lr=1e-4,betas=(.9,.99))
    lifecycle = TransformerOnlineLifecycle(actor,optimizer,global_token_count=8192,
                                         final_action_step=1003,radius_fraction=.001,gradient_clip=100.)
    return lifecycle, {n:p.clone().requires_grad_() for n,p in base.items()}, snapshot


def task_gradients(live):
    # Globally SUM-reduced gradients for a simple mean quadratic task.
    return {n: (p.detach() - torch.sin(p.detach())) * 8192 for n,p in live.items()}


def test_checkpoint_reconstruction_matches_uninterrupted_credit_and_adam():
    online, live, snapshot = make()
    metrics = online.commit(snapshot,live,action_step=1000)
    assert metrics['committed_edit_norm'] > 0
    # Snapshot/anchors may be recycled by a collector after committing.
    snapshot.trajectories.zero_()
    saved = online.state_dict()
    restarted, restored, _ = make()
    for n,p in restored.items():
        with torch.no_grad(): p.copy_(live[n])
    rng = torch.get_rng_state().clone()
    restarted.load_state_dict(saved,restored)
    assert torch.equal(rng,torch.get_rng_state())
    gradients = task_gradients(live)
    copies = {n:g.clone() for n,g in gradients.items()}
    first = online.consume(gradients,gradient_step=1001)
    second = restarted.consume(gradients,gradient_step=1001)
    assert first == second
    for n,g in gradients.items(): torch.testing.assert_close(g,copies[n],rtol=0,atol=0)
    torch.testing.assert_close(online.actor.x,restarted.actor.x,rtol=0,atol=0)
    for key in ('exp_avg','exp_avg_sq','step'):
        torch.testing.assert_close(online.optimizer.state[online.actor.x][key],restarted.optimizer.state[restarted.actor.x][key],rtol=0,atol=0)


def test_dense_task_gradient_agrees_with_lag_one_and_no_double_normalization():
    online, live, snapshot = make()
    online.commit(snapshot,live,action_step=1000)
    proposal = online.action.materialize()
    targets = {n: torch.sin(p.detach()) for n,p in proposal.items()}
    expected, = torch.autograd.grad(sum(.5*(p-targets[n]).square().sum() for n,p in proposal.items()),
                                   (online.actor.x,),retain_graph=True)
    online.consume(task_gradients(live),gradient_step=1001)
    torch.testing.assert_close(online.actor.x.grad,expected,rtol=1e-12,atol=1e-12)


def test_clock_pending_and_terminal_contracts():
    online, live, snapshot = make()
    online.commit(snapshot,live,action_step=1001)
    with pytest.raises(RuntimeError,match='pending'):
        online.commit(snapshot,live,action_step=1002)
    with pytest.raises(ValueError,match='next ordinary'):
        online.consume(task_gradients(live),gradient_step=1003)
    online.consume(task_gradients(live),gradient_step=1002)
    with pytest.raises(ValueError,match='current ordinary'):
        online.commit(snapshot,live,action_step=1001)
    online.commit(snapshot,live,action_step=1002)
    online.consume(task_gradients(live),gradient_step=1003)
    before={n:p.clone() for n,p in live.items()}
    assert online.commit(snapshot,live,action_step=1003)['terminal_identity']
    assert not online.lag.pending
    for n,p in live.items(): torch.testing.assert_close(p,before[n],rtol=0,atol=0)


def test_restore_rejects_changed_live_weights_and_resume_settings():
    online, live, snapshot = make()
    online.commit(snapshot,live,action_step=1000)
    saved=online.state_dict()
    restart, _, _=make()
    bad=copy.deepcopy(saved);bad['config']['global_token_count']=1
    with pytest.raises(ValueError,match='configuration'):
        restart.load_state_dict(bad,live)
    with torch.no_grad(): live['embed.weight'].add_(.1)
    with pytest.raises(ValueError,match='executed proposal'):
        restart.load_state_dict(saved,live)


def test_resume_rejects_different_actor_adam_and_provenance():
    online, live, snapshot = make()
    online.commit(snapshot,live,action_step=1000)
    saved=online.state_dict()
    for key in ('optimizer_groups','policy_provenance'):
        changed=copy.deepcopy(saved)
        if key=='optimizer_groups': changed['config'][key][0]['betas']=(.9,.999)
        else: changed['config'][key]={'source_checkpoint_sha256':'a'*64}
        restart, _, _=make()
        with pytest.raises(ValueError,match='configuration'):
            restart.load_state_dict(changed,live)


def test_actor_mutation_is_rejected_before_credit():
    online, live, snapshot = make()
    online.commit(snapshot,live,action_step=1000)
    with torch.no_grad(): online.actor.x.add_(.01)
    with pytest.raises(RuntimeError,match='mutated'):
        online.consume(task_gradients(live),gradient_step=1001)


def test_nonfinite_proposal_cannot_partially_edit_live_weights():
    online, live, snapshot = make()
    before={n:p.detach().clone() for n,p in live.items()}
    with torch.no_grad(): online.actor.x.fill_(float('nan'))
    with pytest.raises(FloatingPointError,match='nonfinite proposal'):
        online.commit(snapshot,live,action_step=1000)
    for n,p in live.items(): torch.testing.assert_close(p,before[n],rtol=0,atol=0)
    assert not online.lag.pending


def _distributed_worker(rank, rendezvous, output):
    import torch.distributed as dist
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    try:
        online, live, snapshot=make()
        if rank==0: online.commit(snapshot,live,action_step=1000)
        else:
            with pytest.raises(RuntimeError,match='only rank zero'):
                online.commit(snapshot,live,action_step=1000)
        for p in live.values(): dist.broadcast(p.detach(),src=0)
        # Unequal local cotangents make a missing SUM observable. Together they
        # equal the single-process global token-sum gradient, normalized once.
        gradients={n:g*(.25 if rank==0 else .75) for n,g in task_gradients(live).items()}
        for g in gradients.values(): dist.all_reduce(g,op=dist.ReduceOp.SUM)
        if rank==0:
            online.consume(gradients,gradient_step=1001)
            torch.save(dict(actor=online.actor.state_dict(),grad=online.actor.x.grad,
                            live={n:p.detach() for n,p in live.items()}),output)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_two_rank_sum_and_rank_zero_actor_match_single_process(tmp_path):
    import torch.multiprocessing as mp
    rendezvous=str(tmp_path/'rendezvous')
    output=str(tmp_path/'distributed.pt')
    mp.spawn(_distributed_worker,args=(rendezvous,output),nprocs=2,join=True)
    actual=torch.load(output,weights_only=False)
    online, live, snapshot=make()
    online.commit(snapshot,live,action_step=1000)
    online.consume(task_gradients(live),gradient_step=1001)
    torch.testing.assert_close(actual['actor']['x'],online.actor.x,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(actual['grad'],online.actor.x.grad,rtol=1e-12,atol=1e-12)
    for n,p in live.items(): torch.testing.assert_close(actual['live'][n],p,rtol=0,atol=0)


def test_stable_forward_callable_used_for_commit_and_reconstruction():
    online, live, snapshot = make()
    calls=[]
    def forward(obs):
        calls.append('forward')
        return online.actor(obs)
    online.actor_forward=forward
    online.commit(snapshot,live,action_step=1000)
    saved=online.state_dict()
    assert calls==['forward']
    restart, _, _=make()
    def restored_forward(obs):
        calls.append('restore')
        return restart.actor(obs)
    restart.actor_forward=restored_forward
    restart.load_state_dict(saved,live)
    assert calls==['forward','restore']
    bad=copy.deepcopy(saved);bad['config']['actor_execution']={'backend':'other'}
    another, _, _=make()
    with pytest.raises(ValueError,match='configuration'):another.load_state_dict(bad,live)


def _adapter_fixture(mode='learned',world=1):
    from baseline.model import GPT
    from baseline.fp32_execution import enable_fp32_execution
    from rewriter.transformer_online import TransformerChiselAdapter
    torch.set_num_threads(1);torch.manual_seed(100)
    model=enable_fp32_execution(GPT(vocab_size=32,num_layers=1,model_dim=128).float())
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    config=dict(batch_tokens=8,seq_len=4,max_steps=4,schedule_steps=3250,unscored=True,
                gpt_precision='fp32',compile=False,world_size=world,
                transformer_chisel=dict(prefix_steps=0,task_lr=1e-5,representation_lr_multiplier=1.,
                    radius_fraction=.001,gradient_clip=1.,actor_seed=5,architecture={},policy_provenance={},
                    actor_execution={'backend':'eager'},sensor_config=dict(probe_rank=2,anchor_refresh_period=1,trajectory_positions=4)))
    actor=FactorActor().float();actor_optimizer=torch.optim.Adam(actor.parameters(),lr=1e-5,betas=(.9,.99))
    adapter=TransformerChiselAdapter(model,[optimizer],config,mode=mode,actor=actor,actor_optimizer=actor_optimizer)
    model._fp32_sensors=adapter.collector
    return model,optimizer,adapter,config


def _ordinary_step(model,optimizer,adapter,step,rank=0,world=1,batch=None):
    import torch.distributed as dist
    inputs=torch.arange(8).reshape(2,4)%32;targets=(inputs+1)%32
    if batch is not None:inputs,targets=batch
    ids=torch.arange(rank*(2//world),(rank+1)*(2//world))
    adapter.before_step(step,model,[optimizer])
    adapter.before_microbatch(step,ids,inputs[ids])
    loss=model(inputs[ids],targets[ids]);loss.backward()
    if dist.is_initialized():
        for p in model.parameters():dist.all_reduce(p.grad)
    gradients={n:p.grad.clone() for n,p in model.named_parameters()}
    adapter.after_backward(step,model)
    for n,p in model.named_parameters():torch.testing.assert_close(p.grad,gradients[n],rtol=0,atol=0)
    optimizer.step();adapter.after_step(step,model,[optimizer]);model.zero_grad(set_to_none=True)
    return loss.detach()


def test_fp32_shadow_observation_is_identity_and_has_no_validation_side_effect():
    model,opt,adapter,_=_adapter_fixture('shadow')
    from baseline.model import GPT
    from baseline.fp32_execution import enable_fp32_execution
    plain=enable_fp32_execution(GPT(vocab_size=32,num_layers=1,model_dim=128).float())
    plain.load_state_dict(model.state_dict());base=torch.optim.AdamW(plain.parameters(),lr=.001)
    inputs=torch.arange(8).reshape(2,4);targets=inputs+1
    # Initial validation must not populate a collector without sequence IDs.
    with torch.no_grad():torch.testing.assert_close(model(inputs,targets),plain(inputs,targets),rtol=0,atol=0)
    for step in range(2):
        observed=_ordinary_step(model,opt,adapter,step)
        expected=plain(inputs,targets);expected.backward();base.step();plain.zero_grad(set_to_none=True)
        torch.testing.assert_close(observed,expected,rtol=0,atol=0)
        for p,q in zip(model.parameters(),plain.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
    assert not adapter.lifecycle.lag.pending
    adapter.close()


def test_full_adapter_pending_checkpoint_resume_replays_next_gpt_step():
    model,opt,adapter,_=_adapter_fixture()
    _ordinary_step(model,opt,adapter,0)
    assert adapter.metrics['committed_edit_norm']>0
    saved=copy.deepcopy(dict(model=model.state_dict(),optimizer=opt.state_dict(),adapter=adapter.state_dict()))
    restarted,opt2,adapter2,_=_adapter_fixture()
    restarted.load_state_dict(saved['model']);opt2.load_state_dict(saved['optimizer'])
    rng=torch.get_rng_state().clone();adapter2.load_state_dict(saved['adapter'])
    assert torch.equal(rng,torch.get_rng_state())
    for step in range(1,4):
        loss1=_ordinary_step(model,opt,adapter,step);loss2=_ordinary_step(restarted,opt2,adapter2,step)
        torch.testing.assert_close(loss1,loss2,rtol=0,atol=0)
        for p,q in zip(model.parameters(),restarted.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
        torch.testing.assert_close(adapter.lifecycle.actor.x,adapter2.lifecycle.actor.x,rtol=0,atol=0)
        # Timing is observational and cannot match bit-for-bit across reruns.
        comparable=lambda m:{k:v for k,v in m.items() if not k.endswith('_wall_seconds')}
        assert comparable(adapter.metrics)==comparable(adapter2.metrics)
    assert adapter.metrics['terminal_identity'] and not adapter.lifecycle.lag.pending
    adapter.close();adapter2.close()


def _full_adapter_worker(rank,rendezvous):
    import torch.distributed as dist
    from rewriter.transformer_online import authoritative_call, collective_local_call, assert_replicated_before_edit
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    try:
        model,opt,adapter,_=_adapter_fixture(world=2)
        _ordinary_step(model,opt,adapter,0,rank,2)
        saved=copy.deepcopy(dict(model=model.state_dict(),optimizer=opt.state_dict(),adapter=adapter.state_dict()))
        restarted,opt2,adapter2,_=_adapter_fixture(world=2)
        restarted.load_state_dict(saved['model']);opt2.load_state_dict(saved['optimizer']);adapter2.load_state_dict(saved['adapter'])
        for step in range(1,4):
            loss1=_ordinary_step(model,opt,adapter,step,rank,2)
            loss2=_ordinary_step(restarted,opt2,adapter2,step,rank,2)
            torch.testing.assert_close(loss1,loss2,rtol=0,atol=0)
            for p,q in zip(model.parameters(),restarted.parameters()):
                torch.testing.assert_close(p,q,rtol=0,atol=0)
                root=p.detach().clone();dist.broadcast(root,src=0)
                torch.testing.assert_close(root,p,rtol=0,atol=0)
            if rank==0:
                torch.testing.assert_close(adapter.lifecycle.actor.x,adapter2.lifecycle.actor.x,rtol=0,atol=0)
        # Root failure is observed on both ranks before any later broadcasts.
        with pytest.raises(RuntimeError,match='injected root error'):
            authoritative_call(lambda:(_ for _ in ()).throw(ValueError('injected root error')),device=torch.device('cpu'))
        with pytest.raises(RuntimeError,match='injected rank1 error'):
            collective_local_call(lambda:(_ for _ in ()).throw(ValueError('injected rank1 error')) if rank==1 else None,device=torch.device('cpu'))
        # A divergent non-root base weight must raise BEFORE any overwrite.
        probe=next(model.parameters());original=probe.detach().clone()
        if rank==1:
            with torch.no_grad():probe.flatten()[0].add_(1.)
        with pytest.raises(RuntimeError,match='replicas differ before Chisel'):
            assert_replicated_before_edit(model,device=torch.device('cpu'))
        if rank==1:assert not torch.equal(probe,original)
        with torch.no_grad():probe.copy_(original)
        dist.barrier()
        adapter.close();adapter2.close()
    finally:dist.destroy_process_group()


def test_two_rank_full_fp32_adapter_resume_and_collective_failure(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_full_adapter_worker,args=(str(tmp_path/'full-rendezvous'),),nprocs=2,join=True)



def test_baseline_checkpoint_restores_pending_actor_cursor_rng_and_base_adam(tmp_path):
    import numpy as np
    from baseline.data import FinewebStream
    from baseline.checkpoint import build_checkpoint,restore_checkpoint
    from scripts.bundle_baseline_checkpoint import _record_equal
    path=tmp_path/'fineweb_train_000001.bin'
    header=np.zeros(256,dtype=np.int32);header[:3]=[20240520,1,128]
    with path.open('wb') as f:
        f.write(header.tobytes());f.write((np.arange(128,dtype=np.uint16)%32).tobytes())
    def loader():return FinewebStream(str(path),8,4,rank=0,world_size=1,device='cpu')
    stream=loader();model,opt,adapter,config=_adapter_fixture()
    _ordinary_step(model,opt,adapter,0,batch=next(stream))
    saved=copy.deepcopy(build_checkpoint(model,[opt],adapter,stream,config,1,{'forward':1,'backward':1},0.))
    expected_rng_draw=torch.rand(8)
    batch=next(stream);expected_loss=_ordinary_step(model,opt,adapter,1,batch=batch)
    restored,opt2,adapter2,config2=_adapter_fixture();stream2=loader()
    step,counters,_=restore_checkpoint(saved,restored,[opt2],adapter2,stream2,config2)
    assert step==1 and counters=={'forward':1,'backward':1}
    torch.testing.assert_close(torch.rand(8),expected_rng_draw,rtol=0,atol=0)
    batch2=next(stream2)
    for a,b in zip(batch,batch2):torch.testing.assert_close(a,b,rtol=0,atol=0)
    actual_loss=_ordinary_step(restored,opt2,adapter2,1,batch=batch2)
    torch.testing.assert_close(actual_loss,expected_loss,rtol=0,atol=0)
    _record_equal(model.state_dict(),restored.state_dict(),'GPT weights')
    _record_equal(opt.state_dict(),opt2.state_dict(),'base Adam')
    _record_equal(adapter.lifecycle.optimizer.state_dict(),adapter2.lifecycle.optimizer.state_dict(),'actor Adam')
    _record_equal(stream.state_dict(),stream2.state_dict(),'data cursor')
    adapter.close();adapter2.close()


def test_actual_base_displacement_ratios_have_no_silent_floor_or_cap():
    from rewriter.transformer_online import displacement_norms,edit_base_ratios
    base,_=fixture();after={n:p+.001 for n,p in base.items()}
    edited={n:p+.002 for n,p in after.items()}
    motion=displacement_norms(after,base);edit=displacement_norms(edited,after)
    metrics=edit_base_ratios(edit,motion)
    assert metrics['chisel_edit_to_base_norm_ratio_all_parameters']==pytest.approx(2.,rel=2e-4)
    assert len(metrics['chisel_edit_to_base_norm_ratio_by_group'])==15
    zero=displacement_norms(base,base)
    undefined=edit_base_ratios(edit,zero)
    assert undefined['chisel_edit_to_base_norm_ratio_grouped'] is None
    assert all(v is None for v in undefined['chisel_edit_to_base_norm_ratio_by_group'].values())


def test_lag_one_metrics_report_actual_adam_parameter_displacement():
    online,live,snapshot=make()
    online.commit(snapshot,live,action_step=1000)
    before=online.actor.x.detach().clone()
    result=online.consume(task_gradients(live),gradient_step=1001)
    expected=float((online.actor.x.detach()-before).double().norm())
    assert result['chisel_actor_step_norm']==expected
    assert result['chisel_actor_step_norm_by_family']['input_and_norm']==expected
    assert result['chisel_actor_learning_rates']=={'0':1e-4}


def test_realized_group_shrinkage_has_signed_percent_units_and_frozen_zero_groups():
    from rewriter.transformer_online import realized_group_norm_metrics
    from rewriter.grouped_norm import functional_groups
    base,_=fixture(torch.float32)
    groups=functional_groups(base)
    proposed={n:p.clone() for n,p in base.items()}
    for group,names in groups.items():
        factor=1.01 if group=='embedding.weight' else .98
        for name in names:proposed[name].mul_(factor)
    metrics=realized_group_norm_metrics(proposed,base)
    changes=metrics['chisel_action_group_l2_percentage_reduction']
    assert len(changes)==15
    assert changes['embedding.weight']==pytest.approx(-1.,abs=1e-5)
    assert changes['head.weight']==pytest.approx(2.,abs=1e-5)
    assert changes['head.bias']==0
    assert metrics['chisel_action_zero_reference_groups']==['head.bias']
    assert metrics['chisel_action_mean_group_l2_percentage_reduction']==pytest.approx(25/15,abs=1e-5)
    assert torch.equal(proposed['norm1.gains'],base['norm1.gains'])


def test_checkpoint_absolute_group_norms_preserve_rng_and_actual_weights():
    from rewriter.transformer_online import checkpoint_group_norms
    from rewriter.grouped_norm import functional_groups
    parameters,_=fixture(torch.float32)
    before={n:p.clone() for n,p in parameters.items()};rng=torch.get_rng_state().clone()
    result=checkpoint_group_norms(parameters)
    for group,names in functional_groups(parameters).items():
        expected=torch.cat([parameters[n].flatten().double() for n in names]).norm()
        assert result['group_l2_norm'][group]==pytest.approx(float(expected),rel=1e-15)
    for name,p in parameters.items():assert torch.equal(p,before[name])
    assert torch.equal(rng,torch.get_rng_state())
