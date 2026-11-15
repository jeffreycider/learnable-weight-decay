"""Bounded 2-rank gloo staging with an 8x4 lookup/MLP, not a GPT experiment."""
import copy
import datetime
import json
import time
from pathlib import Path
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from baseline.adaptive_transition import build_transition,sha256
from baseline.checkpoint import build_checkpoint,restore_checkpoint
from rewriter.adaptive_adapter import AdaptiveChiselAdapter
from rewriter.grouped_coordinate_chisel import GroupedCoordinateChisel
from rewriter.state_capture import DistributedStateObserver
from rewriter.prefix_cache import pack_record
from rewriter.gpt_precision import RMS_EPSILON
from scripts.fit_cpu_edit_teacher import exact_state_equal


class Block(nn.Module):
    def __init__(self):
        super().__init__();self.norm2=nn.RMSNorm(4);self.mlp=nn.Module()
        self.mlp.fc=nn.Linear(4,4);self.mlp.proj=nn.Linear(4,4)
        self.attn=nn.Module()
        for name in ('q','k','v','proj'):setattr(self.attn,name,nn.Linear(4,4))
    def forward(self,x):
        x=x+self.attn.proj((self.attn.q(x)+self.attn.k(x)+self.attn.v(x)).tanh())
        return x+self.mlp.proj(self.mlp.fc(self.norm2(x)).tanh())


class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.embed=nn.Embedding(8,4);self.norm1=nn.RMSNorm(4)
        self.blocks=nn.ModuleList([Block()]);self.proj=nn.Linear(4,8)
    def forward(self,x):
        x=self.norm1(self.embed(x))
        for block in self.blocks:x=block(x)
        return self.proj(x).square().sum()


class Cursor:
    def __init__(self,rank=0,batches=1):self.rank=rank;self.batches=batches
    def state_dict(self):return dict(rank=self.rank,batches=self.batches)
    def load_state_dict(self,s):self.rank=s['rank'];self.batches=s['batches']


def source_fixture(directory):
    torch.set_num_threads(1);torch.manual_seed(913)
    model=Tiny();opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
    config=dict(mode='baseline',world_size=1,gpt_precision='fp32',compile=False,batch_tokens=8,seq_len=4,max_steps=4,schedule_steps=4,
        fp32_prefix_transition=dict(completed_steps=1,rms_epsilon=RMS_EPSILON,tf32=False))
    sensors=dict(probe_rank=2,anchor_refresh_period=0,trajectory_positions=4,train_sequence_probes=0,audit_sequence_probes=0,radius_fraction=.01)
    observer=DistributedStateObserver(model,[opt],config,sensor_config=sensors)
    x=torch.arange(8).reshape(2,4)
    observer.before_step(0,model,[opt])
    for i in range(2):observer.before_microbatch(0,torch.tensor([i]),x[i:i+1]);model(x[i:i+1]).backward()
    observer.after_backward(0,model);opt.step()
    snapshot,measurements,_=observer.snapshot_from_momentum(0,observer.momentum_projections());observer.release_step(0);observer.close();opt.zero_grad(set_to_none=True)
    statepath=directory/'state.pt'
    torch.save(dict(model=copy.deepcopy(model.state_dict()),completed_steps=1,observation=pack_record(0,snapshot,measurements,train_probe_count=0)),statepath)
    actor=GroupedCoordinateChisel(rank=2,trajectory_dim=12,model_dim=128,num_layers=1,latent_tokens=2,
        fresh_weight_core=True,factor_precision='fp64',residual_precision='fp32')
    private=actor.make_private(snapshot,state_key=('cpu-teaching',0,1,sha256(statepath)),seed=52)
    with torch.no_grad():
        for bank in actor.output_heads:
            for head in bank.values():head.weight.normal_(std=1e-4)
    policy=directory/'policy.pt';torch.save(dict(actor=actor.state_dict(),private={1:private.state_dict()},metadata=dict(cases=[dict(state_id=1,sha256={'state':sha256(statepath)},private_identity=private.get_extra_state())])),policy)
    baseline=copy.deepcopy(build_checkpoint(model,[opt],None,Cursor(),config,1,{'tokens':8},.01))
    settings=dict(run_id='gloo-tiny',parameter_view='unwrapped_stored_fp32',actor_execution={'backend':'eager'},online_auxiliary='none',bootstrap='teaching_snapshot_at_activation',head_lr=1e-9,body_lr=1e-7,private_lr=1e-4,shared_eps=1e-8,private_eps=1e-12,shared_clip=1.,private_clip=1.,radius_fraction=.01,betas=[.9,.99],sensor_config=sensors,distributed_qualification='cpu_gloo_two_rank_staging')
    torch.save(dict(baseline=baseline,settings=settings),directory/'fixture.pt')
    return baseline,settings


def activation(directory,baseline,settings,rank=0,world=1):
    baseline=copy.deepcopy(baseline);baseline['config']['world_size']=world;baseline['data_cursor']['rank']=rank
    # This is a synthetic baseline fixture, never a resize of a real checkpoint.
    return build_transition(baseline,policy_path=directory/'policy.pt',teaching_state_path=directory/'state.pt',case_id=1,settings=settings,mode='learned')


def restore(saved):
    model=Tiny();opt=torch.optim.AdamW(model.parameters(),lr=1e-4);cursor=Cursor()
    adapter=AdaptiveChiselAdapter(model,[opt],saved['config'],mode='learned')
    restore_checkpoint(copy.deepcopy(saved),model,[opt],adapter,cursor,saved['config'])
    return model,opt,adapter,cursor


def ordinary(bundle,t,rank=0,world=1):
    model,opt,adapter,cursor=bundle;adapter.before_step(t,model,[opt]);x=torch.arange(8).reshape(2,4)
    loss=torch.zeros(())
    for i in range(rank,2,world):
        adapter.before_microbatch(t,torch.tensor([i]),x[i:i+1]);value=model(x[i:i+1]);loss+=value.detach();value.backward()
    if world>1:
        for p in model.parameters():dist.all_reduce(p.grad,op=dist.ReduceOp.SUM)
        dist.all_reduce(loss,op=dist.ReduceOp.SUM)
    adapter.after_backward(t,model);opt.step();adapter.after_step(t,model,[opt]);opt.zero_grad(set_to_none=True);cursor.batches+=1
    return loss


def checkpoint(bundle,t):
    model,opt,adapter,cursor=bundle
    return copy.deepcopy(build_checkpoint(model,[opt],adapter,cursor,adapter.config,t,{'tokens':t*8},t*.01))


def worker(rank,directory,init):
    torch.set_num_threads(1);torch.set_num_interop_threads(1);directory=Path(directory)
    dist.init_process_group('gloo',init_method='file://'+init,rank=rank,world_size=2,timeout=datetime.timedelta(seconds=40))
    try:
        fixture=torch.load(directory/'fixture.pt',weights_only=False)
        saved=activation(directory,fixture['baseline'],fixture['settings'],rank,2)
        bundle=restore(saved);model,opt,adapter,cursor=bundle
        assert (adapter.lifecycle is not None)==(rank==0)
        forwards=[];hook=model.register_forward_hook(lambda *args:forwards.append(1))
        initial={n:p.detach().clone() for n,p in model.named_parameters()}
        # Both replicas receive the initial S-1 edit before the ordinary S pass.
        gathered=[None,None];dist.all_gather_object(gathered,initial)
        assert exact_state_equal(gathered[0],gathered[1])
        ordinary(bundle,1,rank,2)
        saved=checkpoint(bundle,2)
        assert (saved['controller']['learner'] is not None)==(rank==0)
        assert saved['controller']['rank']==rank and saved['data_cursor']['rank']==rank
        rng=torch.get_rng_state().clone();resumed=restore(saved)
        assert torch.equal(torch.get_rng_state(),saved['rng']['cpu'])
        left=ordinary(bundle,2,rank,2);right=ordinary(resumed,2,rank,2)
        assert torch.equal(left,right) and exact_state_equal(bundle[0].state_dict(),resumed[0].state_dict())
        if rank==0:
            for field in ('binding','shared_optimizer','private_optimizer'):
                assert exact_state_equal(getattr(adapter.lifecycle,field).state_dict(),getattr(resumed[2].lifecycle,field).state_dict())
        ordinary(bundle,3,rank,2)
        assert len(forwards)==3
        final=checkpoint(bundle,4)
        allmodels=[None,None];dist.all_gather_object(allmodels,model.state_dict());assert exact_state_equal(*allmodels)
        if rank==0:
            assert adapter.lifecycle.terminal and adapter.lifecycle.pending is None and adapter.lifecycle.meta_updates==3
            assert adapter.metrics['chisel_extra_gpt_forwards']==adapter.metrics['chisel_extra_gpt_backwards']==0
            torch.save(dict(initial=initial,final=final,gradients={n:p.grad.detach().clone() for n,p in adapter.lifecycle.binding.named_parameters() if p.grad is not None}),directory/'distributed_result.pt')
        hook.remove();adapter.close();resumed[2].close()
        # Non-root restore failure must propagate to both ranks before root work.
        bad=copy.deepcopy(saved)
        if rank==1:bad['controller']['rank']=0
        failed=False
        try:restore(bad)
        except RuntimeError as e:failed='rank mismatch' in str(e)
        assert failed
        dist.barrier()
    finally:dist.destroy_process_group()


def test_two_rank_global_credit_replica_broadcast_and_pending_resume(tmp_path):
    baseline,settings=source_fixture(tmp_path)
    reference=restore(activation(tmp_path,baseline,settings))
    initial={n:p.detach().clone() for n,p in reference[0].named_parameters()}
    for t in (1,2,3):ordinary(reference,t)
    expected=checkpoint(reference,4);expected_gradients={n:p.grad.detach().clone() for n,p in reference[2].lifecycle.binding.named_parameters() if p.grad is not None};reference[2].close()
    ctx=mp.start_processes(worker,args=(str(tmp_path),str(tmp_path/'gloo-init')),nprocs=2,join=False,start_method='spawn')
    deadline=time.monotonic()+90
    try:
        while not ctx.join(timeout=1):
            if time.monotonic()>deadline:raise TimeoutError('bounded gloo staging exceeded90s')
    finally:
        for p in ctx.processes:
            if p.is_alive():p.terminate();p.join(timeout=5)
    actual=torch.load(tmp_path/'distributed_result.pt',weights_only=False)
    assert exact_state_equal(initial,actual['initial'])
    for n,v in expected['model'].items():torch.testing.assert_close(actual['final']['model'][n],v,atol=3e-7,rtol=3e-6)
    # Global SUM gradient credit, shared and private motion, against the same
    # single-rank two-microbatch ordinary computation. No GPT forward involved.
    reference_learner=expected['controller']['learner'];learner=actual['final']['controller']['learner']
    squared={role:[0.,0.] for role in ('shared','private')}
    for n,v in reference_learner['binding'].items():
        if isinstance(v,torch.Tensor):
            torch.testing.assert_close(learner['binding'][n],v,atol=2e-7,rtol=2e-5)
            role=n.split('.')[0];squared[role][0]+=float((learner['binding'][n].double()-v.double()).square().sum());squared[role][1]+=float(v.double().square().sum())
    for error,norm in squared.values():assert (error/norm)**.5<1e-6

    gradient_errors={role:[0.,0.] for role in ('shared','private')}
    assert actual['gradients'].keys()==expected_gradients.keys()
    for n,v in expected_gradients.items():
        observed=actual['gradients'][n]
        torch.testing.assert_close(observed,v,atol=1e-7,rtol=1e-4)
        role=n.split('.')[0];gradient_errors[role][0]+=float((observed.double()-v.double()).square().sum());gradient_errors[role][1]+=float(v.double().square().sum())
    for error,norm in gradient_errors.values():assert norm>0 and (error/norm)**.5<1e-4
    (tmp_path/'measured.json').write_text(json.dumps(dict(parameter_relative_l2={r:(e/n)**.5 for r,(e,n) in squared.items()},gradient_relative_l2={r:(e/n)**.5 for r,(e,n) in gradient_errors.items()},ordinary_forwards_per_rank=3,meta_updates=3,extra_gpt_calls=0),indent=2)+'\n')


@pytest.mark.parametrize('mutation',['private','snapshot','base'])
def test_post_broadcast_rebind_checks_values_and_retained_graph(tmp_path,mutation):
    baseline,settings=source_fixture(tmp_path);model,opt,adapter,cursor=restore(activation(tmp_path,baseline,settings))
    life=adapter.lifecycle;params=dict(model.named_parameters());p=next(iter(params.values()))
    with torch.no_grad():p.copy_(p)  # collective-like version change without value change
    life.rebind_after_broadcast(params)
    life._check_versions(life.pending)
    with torch.no_grad():p.add_(1e-3)
    with pytest.raises(ValueError,match='executed'):life.rebind_after_broadcast(params)
    with torch.no_grad():
        p.copy_(life.pending['executed'][next(iter(params))])
        retained={'private':next(life.binding.private.parameters()),'snapshot':life.pending['snapshot'].trajectories,
                  'base':next(iter(life.pending['base'].values()))}[mutation]
        retained.add_(1e-6)
    with pytest.raises(RuntimeError,match='mutated'):life.rebind_after_broadcast(params)
    adapter.close()
