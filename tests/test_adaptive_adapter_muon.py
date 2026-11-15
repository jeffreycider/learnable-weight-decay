"""Real PR323 Muon/AdamW on two tiny CPU/gloo ranks; no GPT or GPU run."""
import copy
import datetime
import json
import resource
import time
from pathlib import Path
from unittest.mock import patch
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from baseline import optim as pr323
from baseline.train import make_optimizers
from baseline.checkpoint import build_checkpoint,restore_checkpoint
from baseline.adaptive_transition import build_transition,sha256
from rewriter.adaptive_adapter import AdaptiveChiselAdapter
from rewriter.state_capture import DistributedStateObserver
from rewriter.grouped_coordinate_chisel import GroupedCoordinateChisel
from rewriter.prefix_cache import pack_record
from rewriter.gpt_precision import RMS_EPSILON
from scripts.fit_cpu_edit_teacher import exact_state_equal
from tests.test_adaptive_adapter_distributed import Tiny,Cursor


def raw_backward(model,observer,t,rank):
    observer.before_step(t,model,[]) if isinstance(observer,DistributedStateObserver) else None
    x=torch.arange(8).reshape(2,4)[rank:rank+1]
    observer.before_microbatch(t,torch.tensor([rank]),x)
    loss=model(x);loss.backward()
    for p in model.parameters():dist.all_reduce(p.grad,op=dist.ReduceOp.SUM)
    return {n:p.grad.clone() for n,p in model.named_parameters()}


def verify_momentum(observer,rank):
    projected=observer.momentum_projections();local={}
    for n,p in observer.params.items():
        opt,_=observer.owners[id(p)];owner=observer.shard_rank[id(p)]
        if rank==owner:
            state=opt.state[p];local[n]=state.get('momentum',state.get('exp_avg')).clone()
        if isinstance(opt,pr323.Muon):
            assert ('momentum' in opt.state.get(p,{}))==(rank==owner)
    gathered=[None,None];dist.all_gather_object(gathered,local)
    direct={}
    for mapping in gathered:
        assert not direct.keys()&mapping.keys();direct.update(mapping)
    maxerr=0.
    for n,p in observer.params.items():
        m=observer.oriented(n,direct[n]).float()/observer.config['batch_tokens']
        expected=(m@observer.anchors[n].B,m.T@observer.anchors[n].A) if p.ndim==2 else (m,)
        values=projected[n] if p.ndim==2 else (projected[n],)
        for actual,wanted in zip(values,expected):
            torch.testing.assert_close(actual,wanted,atol=0,rtol=0)
            maxerr=max(maxerr,float((actual-wanted).abs().max()))
    return maxerr


def advance(model,opts,adapter,cursor,t,rank,rows):
    adapter.before_step(t,model,opts)
    raw=raw_backward(model,adapter,t,rank)
    old_consume=None;observed=[]
    if rank==0:
        old_consume=adapter.lifecycle.consume
        def consume(gradients,**kwargs):
            assert exact_state_equal(gradients,raw)
            observed.append(True)
            return old_consume(gradients,**kwargs)
        adapter.lifecycle.consume=consume
    adapter.after_backward(t,model)
    if rank==0:
        adapter.lifecycle.consume=old_consume;assert observed==[True]
    assert all(torch.equal(p.grad,raw[n]) for n,p in model.named_parameters())
    for opt in opts:opt.step()  # Actual Muon mutates owned gradient buffers here.
    changed=[]
    for n,p in model.named_parameters():
        opt,_=adapter.observer.owners[id(p)]
        if isinstance(opt,pr323.Muon):
            owned=adapter.observer.shard_rank[id(p)]==rank
            assert torch.equal(p.grad,raw[n])!=owned
            if owned:changed.append(n)
    assert changed
    maxerr=verify_momentum(adapter.observer,rank)
    adapter.after_step(t,model,opts)
    models=[None,None];dist.all_gather_object(models,model.state_dict());assert exact_state_equal(*models)
    for opt in opts:opt.zero_grad(set_to_none=True)
    cursor.batches+=1
    rows.append(dict(step=t,owned_mutated_gradients=changed,momentum_projection_max_abs=maxerr))


def restore(saved):
    model=Tiny();opts=make_optimizers(model);cursor=Cursor()
    adapter=AdaptiveChiselAdapter(model,opts,saved['config'],mode='learned')
    restore_checkpoint(copy.deepcopy(saved),model,opts,adapter,cursor,saved['config'])
    return model,opts,adapter,cursor


def worker(rank,directory,init):
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(817)
    directory=Path(directory)
    dist.init_process_group('gloo',init_method='file://'+init,rank=rank,world_size=2,timeout=datetime.timedelta(seconds=40))
    numerical=pr323.muon_update._torchdynamo_orig_callable
    actual_ns=pr323.zeropower_via_newtonschulz5;ns_calls=[]
    def counted_ns(value):ns_calls.append(tuple(value.shape));return actual_ns(value)
    try:
        with patch.object(pr323,'muon_update',numerical),patch.object(pr323,'zeropower_via_newtonschulz5',counted_ns):
            model=Tiny();opts=make_optimizers(model)
            config=dict(mode='baseline',world_size=2,gpt_precision='fp32',compile=False,batch_tokens=8,seq_len=4,max_steps=4,schedule_steps=4,
                fp32_prefix_transition=dict(completed_steps=1,rms_epsilon=RMS_EPSILON,tf32=False))
            sensors=dict(probe_rank=2,anchor_refresh_period=0,trajectory_positions=4,train_sequence_probes=0,audit_sequence_probes=0,radius_fraction=.01)
            observer=DistributedStateObserver(model,opts,config,sensor_config=sensors)
            raw_backward(model,observer,0,rank);observer.after_backward(0,model)
            for opt in opts:opt.step()
            verify_momentum(observer,rank)
            momentum=observer.momentum_projections()
            if rank==0:
                snapshot,measurements,_=observer.snapshot_from_momentum(0,momentum)
                statepath=directory/'state.pt';torch.save(dict(model=copy.deepcopy(model.state_dict()),completed_steps=1,
                    observation=pack_record(0,snapshot,measurements,train_probe_count=0)),statepath)
                actor=GroupedCoordinateChisel(rank=2,trajectory_dim=12,model_dim=128,num_layers=1,latent_tokens=2,
                    fresh_weight_core=True,factor_precision='fp64',residual_precision='fp32')
                private=actor.make_private(snapshot,state_key=('cpu-teaching',0,1,sha256(statepath)),seed=52)
                with torch.no_grad():
                    for bank in actor.output_heads:
                        for head in bank.values():head.weight.normal_(std=1e-4)
                torch.save(dict(actor=actor.state_dict(),private={1:private.state_dict()},metadata=dict(cases=[dict(state_id=1,
                    sha256={'state':sha256(statepath)},private_identity=private.get_extra_state())])),directory/'policy.pt')
            observer.release_step(0);observer.close()
            for opt in opts:opt.zero_grad(set_to_none=True)
            dist.barrier()
            baseline=copy.deepcopy(build_checkpoint(model,opts,None,Cursor(rank),config,1,{'tokens':8},.01))
            settings=dict(run_id='real-muon-tiny',parameter_view='unwrapped_stored_fp32',actor_execution={'backend':'eager'},online_auxiliary='none',
                bootstrap='teaching_snapshot_at_activation',head_lr=1e-9,body_lr=1e-7,private_lr=1e-4,shared_eps=1e-8,private_eps=1e-12,
                shared_clip=1.,private_clip=1.,radius_fraction=.01,betas=[.9,.99],sensor_config=sensors,distributed_qualification='cpu_gloo_two_rank_staging')
            transition=build_transition(baseline,policy_path=directory/'policy.pt',teaching_state_path=directory/'state.pt',case_id=1,settings=settings,mode='learned')
            model,opts,adapter,cursor=restore(transition);rows=[]
            advance(model,opts,adapter,cursor,1,rank,rows)
            saved=copy.deepcopy(build_checkpoint(model,opts,adapter,cursor,adapter.config,2,{'tokens':16},.02))
            other,other_opts,resumed,other_cursor=restore(saved)
            assert torch.equal(torch.get_rng_state(),saved['rng']['cpu'])
            advance(model,opts,adapter,cursor,2,rank,rows)
            advance(other,other_opts,resumed,other_cursor,2,rank,[])
            assert exact_state_equal(model.state_dict(),other.state_dict())
            assert all(exact_state_equal(a.state_dict(),b.state_dict()) for a,b in zip(opts,other_opts))
            if rank==0:
                for field in ('binding','shared_optimizer','private_optimizer'):
                    assert exact_state_equal(getattr(adapter.lifecycle,field).state_dict(),getattr(resumed.lifecycle,field).state_dict())
            advance(model,opts,adapter,cursor,3,rank,rows)
            if rank==0:
                assert adapter.lifecycle.terminal and adapter.lifecycle.pending is None and adapter.lifecycle.meta_updates==3
            owned=sum(len(o.state) for o in opts if isinstance(o,pr323.Muon))
            assert len(ns_calls)==owned*5  # prefix +3 ordinary suffix +1 resume replay
            (directory/f'rank{rank}.json').write_text(json.dumps(dict(rank=rank,rows=rows,newton_schulz_calls=len(ns_calls),
                owned_muon_momenta=owned,exact_pending_resume=True,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024),indent=2)+'\n')
            adapter.close();resumed.close()
    finally:dist.destroy_process_group()


def test_real_muon_global_gradient_timing_shards_and_pending_resume(tmp_path):
    torch.set_num_threads(1)
    ctx=mp.start_processes(worker,args=(str(tmp_path),str(tmp_path/'gloo-init')),nprocs=2,join=False,start_method='spawn')
    deadline=time.monotonic()+90
    try:
        while not ctx.join(timeout=1):
            if time.monotonic()>deadline:raise TimeoutError('Muon gloo staging exceeded90s')
    finally:
        for process in ctx.processes:
            if process.is_alive():process.terminate();process.join(timeout=5)
    rows=[json.loads((tmp_path/f'rank{rank}.json').read_text()) for rank in (0,1)]
    assert sum(row['owned_muon_momenta'] for row in rows)==6
