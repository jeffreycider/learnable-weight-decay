import copy
import torch
import pytest
from baseline.model import GPT
from baseline.fp32_execution import enable_fp32_execution
from baseline.checkpoint import build_checkpoint,restore_checkpoint
from baseline.adaptive_transition import build_transition,sha256
from rewriter.adaptive_adapter import AdaptiveChiselAdapter
from rewriter.grouped_coordinate_chisel import GroupedCoordinateChisel
from rewriter.state_capture import DistributedStateObserver
from rewriter.prefix_cache import pack_record
from rewriter.gpt_precision import RMS_EPSILON
from scripts.fit_cpu_edit_teacher import exact_state_equal


class Cursor:
    def __init__(self,batches=0):self.batches=batches
    def state_dict(self):return dict(batches=self.batches,rank=0)
    def load_state_dict(self,s):self.batches=s['batches']


def fixture(tmp_path,mode='learned'):
    torch.set_num_threads(1);torch.manual_seed(724)
    model=enable_fp32_execution(GPT(32,1,128).float());optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    config=dict(mode='baseline',world_size=1,gpt_precision='fp32',compile=False,batch_tokens=8,seq_len=4,max_steps=4,schedule_steps=4,
        fp32_prefix_transition=dict(completed_steps=1,rms_epsilon=RMS_EPSILON,tf32=False))
    sensors=dict(probe_rank=2,anchor_refresh_period=0,trajectory_positions=4,train_sequence_probes=4,audit_sequence_probes=4,radius_fraction=.01)
    observer=DistributedStateObserver(model,[optimizer],config,sensor_config=sensors)
    model._fp32_sensors=observer.collector
    x=torch.arange(8).reshape(2,4);y=(x+1)%32
    observer.before_step(0,model,[optimizer]);observer.before_microbatch(0,torch.arange(2),x)
    model(x,y).backward();observer.after_backward(0,model);optimizer.step()
    snapshot,measurements,_=observer.snapshot_from_momentum(0,observer.momentum_projections())
    observer.release_step(0);observer.close();optimizer.zero_grad(set_to_none=True)
    teaching=dict(model=copy.deepcopy(model.state_dict()),completed_steps=1,observation=pack_record(0,snapshot,measurements,train_probe_count=4))
    statepath=tmp_path/'state.pt';torch.save(teaching,statepath)
    actor=GroupedCoordinateChisel(rank=2,trajectory_dim=snapshot.trajectories.shape[-1],model_dim=128,num_layers=1,latent_tokens=2,
        fresh_weight_core=True,factor_precision='fp64',residual_precision='fp32')
    private=actor.make_private(snapshot,state_key=('cpu-teaching',0,1,sha256(statepath)),seed=75)
    with torch.no_grad():
        for bank in actor.output_heads:
            for head in bank.values():head.weight.normal_(std=1e-4)
    policy=dict(actor=actor.state_dict(),private={1:private.state_dict()},metadata=dict(cases=[dict(state_id=1,sha256={'state':sha256(statepath)},private_identity=private.get_extra_state())]))
    path=tmp_path/'policy.pt';torch.save(policy,path)
    baseline=copy.deepcopy(build_checkpoint(model,[optimizer],None,Cursor(1),config,1,{'tokens':8},.1))
    settings=dict(run_id='tiny-online',parameter_view='unwrapped_stored_fp32',actor_execution={'backend':'eager'},online_auxiliary='none',
        bootstrap='teaching_snapshot_at_activation',head_lr=1e-9,body_lr=1e-7,private_lr=1e-4,shared_eps=1e-8,private_eps=1e-12,
        shared_clip=1.,private_clip=1.,radius_fraction=.01,betas=[.9,.99],sensor_config=sensors)
    transition=build_transition(baseline,policy_path=path,teaching_state_path=statepath,case_id=1,settings=settings,mode=mode)
    return model,optimizer,baseline,transition,x,y


def restore(model,opt,state):
    adapter=AdaptiveChiselAdapter(model,[opt],state['config'],mode=state['config']['mode']);cursor=Cursor()
    model._fp32_sensors=adapter.collector
    restore_checkpoint(state,model,[opt],adapter,cursor,state['config'])
    return adapter,cursor


def step(adapter,model,opt,cursor,t,x,y):
    adapter.before_step(t,model,[opt]);adapter.before_microbatch(t,torch.arange(2),x)
    loss=model(x,y);loss.backward();adapter.after_backward(t,model)
    opt.step();adapter.after_step(t,model,[opt]);opt.zero_grad(set_to_none=True);cursor.batches+=1
    return loss.detach()


def test_actual_ordinary_pass_pending_resume_and_terminal(tmp_path):
    model,opt,baseline,transition,x,y=fixture(tmp_path)
    assert exact_state_equal(baseline['model'],transition['model'])
    assert exact_state_equal(baseline['optimizers'],transition['optimizers'])
    adapter,cursor=restore(model,opt,transition)
    assert adapter.lifecycle.pending['action_step']==0
    assert any(not torch.equal(v,baseline['model'][n]) for n,v in model.state_dict().items())
    forwards=[];handle=model.register_forward_hook(lambda *args:forwards.append(1))
    step(adapter,model,opt,cursor,1,x,y)
    assert len(forwards)==1 and adapter.lifecycle.meta_updates==1
    saved=copy.deepcopy(build_checkpoint(model,[opt],adapter,cursor,transition['config'],2,{'tokens':16},.2))
    other=enable_fp32_execution(GPT(32,1,128).float());other_opt=torch.optim.AdamW(other.parameters(),lr=1e-4)
    restored,other_cursor=restore(other,other_opt,saved)
    left=step(adapter,model,opt,cursor,2,x,y);right=step(restored,other,other_opt,other_cursor,2,x,y)
    assert torch.equal(left,right)
    assert exact_state_equal(adapter.lifecycle.binding.state_dict(),restored.lifecycle.binding.state_dict())
    assert exact_state_equal(adapter.lifecycle.shared_optimizer.state_dict(),restored.lifecycle.shared_optimizer.state_dict())
    assert exact_state_equal(adapter.lifecycle.private_optimizer.state_dict(),restored.lifecycle.private_optimizer.state_dict())
    assert exact_state_equal(model.state_dict(),other.state_dict())
    step(adapter,model,opt,cursor,3,x,y)
    assert len(forwards)==3 and adapter.lifecycle.meta_updates==3 and adapter.lifecycle.terminal
    assert adapter.lifecycle.pending is None
    assert adapter.metrics['chisel_extra_gpt_forwards']==adapter.metrics['chisel_extra_gpt_backwards']==0
    handle.remove();adapter.close();restored.close()


def test_shadow_identity_and_activation_rejection(tmp_path):
    model,opt,baseline,state,x,y=fixture(tmp_path,'shadow');adapter,cursor=restore(model,opt,state)
    assert exact_state_equal(model.state_dict(),baseline['model'])
    reference=copy.deepcopy(model);reference._fp32_sensors=None;base=torch.optim.AdamW(reference.parameters(),lr=1e-4);base.load_state_dict(copy.deepcopy(baseline['optimizers'][0]))
    step(adapter,model,opt,cursor,1,x,y);reference(x,y).backward();base.step()
    assert exact_state_equal(model.state_dict(),reference.state_dict())
    adapter.close()
    wrong=copy.deepcopy(state);next(iter(wrong['model'].values())).add_(1e-4)
    with pytest.raises(RuntimeError,match='activation'):restore(model,opt,wrong)


def test_transition_rejects_case_transfer_precision_and_world(tmp_path):
    _,_,baseline,state,_,_=fixture(tmp_path)
    c=state['config']['transformer_chisel']
    def build(b,case=1):return build_transition(b,policy_path=c['policy_path'],teaching_state_path=c['teaching_state_path'],case_id=case,settings=c,mode='learned')
    with pytest.raises(ValueError,match='transfer'):build(baseline,1000)
    wrong=copy.deepcopy(baseline);wrong['config']['world_size']=8
    with pytest.raises(ValueError,match='single-rank'):build(wrong)
    wrong=copy.deepcopy(baseline);wrong['model']['embed.weight']=wrong['model']['embed.weight'].bfloat16()
    with pytest.raises(ValueError,match='FP32'):build(wrong)
