"""CPU integration of the opt-in schedule through actual GPT/observer hooks."""
import copy
from datetime import timedelta
import torch
from tests.test_transformer_action import FactorActor
from tests.test_transformer_online import _ordinary_step
from tests.test_online_norm_lifecycle import AUX,assert_tree
from baseline.model import GPT
from baseline.fp32_execution import enable_fp32_execution
from rewriter.transformer_online import TransformerChiselAdapter


def make(world=1):
    torch.set_num_threads(1);torch.manual_seed(100)
    model=enable_fp32_execution(GPT(vocab_size=32,num_layers=1,model_dim=128).float())
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    config=dict(batch_tokens=8,seq_len=4,max_steps=4,schedule_steps=3250,unscored=True,
        gpt_precision='fp32',compile=False,world_size=world,
        transformer_chisel=dict(prefix_steps=0,task_lr=1e-5,representation_lr_multiplier=1.,
            radius_fraction=.001,gradient_clip=1.,actor_seed=5,architecture={},policy_provenance={},
            actor_execution={'backend':'eager'},sensor_config=dict(probe_rank=2,anchor_refresh_period=1,trajectory_positions=4),
            replica_check_every=1,group_norm_metrics_every=1,norm_auxiliary=copy.deepcopy(AUX)))
    actor=FactorActor().float();actor_optimizer=torch.optim.Adam(actor.parameters(),lr=1e-5,betas=(.9,.99))
    adapter=TransformerChiselAdapter(model,[optimizer],config,mode='learned',actor=actor,actor_optimizer=actor_optimizer)
    model._fp32_sensors=adapter.collector
    return model,optimizer,adapter


def trace(rank=0,world=1):
    model,opt,adapter=make(world)
    _ordinary_step(model,opt,adapter,0,rank,world)
    saved=copy.deepcopy(dict(model=model.state_dict(),optimizer=opt.state_dict(),adapter=adapter.state_dict()))
    restored,opt2,adapter2=make(world)
    restored.load_state_dict(saved['model']);opt2.load_state_dict(saved['optimizer'])
    rng=torch.get_rng_state().clone();adapter2.load_state_dict(saved['adapter'])
    assert torch.equal(rng,torch.get_rng_state())
    if rank==0:
        assert adapter2.lifecycle.norm_updates==1
        assert adapter2.lifecycle.optimizer.state[adapter2.lifecycle.actor.x]['step']==1
    for t in range(1,4):
        first=_ordinary_step(model,opt,adapter,t,rank,world)
        second=_ordinary_step(restored,opt2,adapter2,t,rank,world)
        torch.testing.assert_close(first,second,rtol=0,atol=0)
        assert_tree(model.state_dict(),restored.state_dict())
        assert_tree(opt.state_dict(),opt2.state_dict())
        if world>1:
            for value in model.state_dict().values():
                root=value.clone();torch.distributed.broadcast(root,0)
                torch.testing.assert_close(value,root,rtol=0,atol=0)
        if rank==0:
            assert_tree(adapter.lifecycle.state_dict(),adapter2.lifecycle.state_dict())
            metric={k:v for k,v in adapter.metrics.items() if not k.endswith('_wall_seconds')}
            metric2={k:v for k,v in adapter2.metrics.items() if not k.endswith('_wall_seconds')}
            assert_tree(metric,metric2)
            assert metric['meta_updates']==t
            assert metric['chisel_norm_auxiliary_updates']==min(t+1,3)
            assert metric['chisel_actor_optimizer_step_calls']==t+min(t+1,3)
            assert metric['chisel_extra_gpt_forwards']==metric['chisel_extra_gpt_backwards']==0
            if t<3:
                assert metric['chisel_norm_auxiliary_actor_forward_calls']==metric['chisel_norm_auxiliary_actor_backward_calls']==1
                assert metric['chisel_norm_auxiliary_action_step']==t
                assert metric['chisel_task_consumed_action_step']==t-1
                assert metric['chisel_norm_auxiliary_final_prediction_gap_percentage_points']==(
                    -100*metric['chisel_norm_auxiliary_final_fractional_change']-metric['chisel_action_mean_group_l2_percentage_reduction'])
            else:
                assert metric['terminal_identity']
                assert 'chisel_norm_auxiliary_action_step' not in metric
                assert not adapter.lifecycle.lag.pending
                assert adapter.lifecycle.optimizer.state[adapter.lifecycle.actor.x]['step']==6
        else:
            assert adapter.lifecycle is None and adapter2.lifecycle is None
    adapter.close();adapter2.close()


def test_full_fp32_norm_adapter_resume_metrics_and_terminal():
    trace()


def worker(rank,rendezvous):
    torch.distributed.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2,timeout=timedelta(seconds=60))
    try:trace(rank,2)
    finally:torch.distributed.destroy_process_group()


def test_two_rank_norm_adapter_broadcast_and_pending_resume(tmp_path):
    torch.multiprocessing.spawn(worker,args=(str(tmp_path/'norm-rendezvous'),),nprocs=2,join=True)
