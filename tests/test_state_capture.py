import copy
import os
import tempfile
from unittest.mock import patch
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from rewriter.state_capture import StateCaptureAdapter
from rewriter.state_bank import StateBankReader
from tests.test_state_bank import provenance
from baseline.optim import Muon


def configuration(steps=2):
    return dict(batch_tokens=12,seq_len=3,max_steps=steps,schedule_steps=3250,mode='baseline',
                state_bank=dict(steps=steps,probe_rank=2,anchor_refresh_period=1,trajectory_positions=4))


def model():return torch.nn.Sequential(torch.nn.Linear(4,4),torch.nn.GELU(),torch.nn.Linear(4,4))


def recover_banks(features,rank=2):
    return [features[:,:4*rank][:,i*rank:(i+1)*rank]*features[0,5*rank+1+i].exp() for i in range(4)]


def test_capture_is_identity_raw_grad_and_actual_update(tmp_path):
    torch.manual_seed(9);live=model();control=copy.deepcopy(live)
    optimizer=torch.optim.AdamW(live.parameters(),lr=.01);base=torch.optim.AdamW(control.parameters(),lr=.01)
    config=configuration();rng=torch.random.get_rng_state().clone()
    capture=StateCaptureAdapter(live,[optimizer],config,directory=tmp_path,provenance=provenance())
    assert torch.equal(rng,torch.random.get_rng_state())
    inputs=torch.randn(4,3,4);raw={};before={}
    for step in range(2):
        capture.before_step(step,live,[optimizer]);capture.before_microbatch(step,torch.arange(4),inputs)
        before={n:p.detach().clone() for n,p in live.named_parameters()}
        live(inputs).square().sum().backward();control(inputs).square().sum().backward()
        raw={n:p.grad.clone()/12 for n,p in live.named_parameters()}
        capture.after_backward(step,live)
        optimizer.step();base.step();capture.after_step(step,live,[optimizer])
        assert all(torch.equal(p,q) for p,q in zip(live.parameters(),control.parameters()))
        live.zero_grad(set_to_none=True);control.zero_grad(set_to_none=True)
    reader=StateBankReader(tmp_path);record=reader.read(2)
    for n,obs in record.snapshot.matrices.items():
        p=dict(live.named_parameters())[n];a=obs.anchors;w,g,m,u=recover_banks(obs.rows)
        torch.testing.assert_close(w,p.detach()@a.B,rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(g,raw[n]@a.B,rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(m,(optimizer.state[p]['exp_avg']/12)@a.B,rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(u,(p.detach()-before[n])@a.B,rtol=1e-5,atol=1e-6)
    assert capture.metrics['capture_allones_relative_error']<1e-5
    assert capture.metrics['capture_actor_parameters']==0
    assert not hasattr(capture,'controller')


def test_compiled_gpt_path_preserves_sensor_values(tmp_path):
    torch.manual_seed(19);live=model();control=copy.deepcopy(live)
    optimizer=torch.optim.AdamW(live.parameters(),lr=.01);base=torch.optim.AdamW(control.parameters(),lr=.01)
    capture=StateCaptureAdapter(live,[optimizer],configuration(),directory=tmp_path,provenance=provenance())
    compiled=torch.compile(live,backend='eager',dynamic=False)
    inputs=torch.randn(4,3,4)
    for step in range(2):
        capture.before_step(step,live,[optimizer]);capture.before_microbatch(step,torch.arange(4),inputs)
        compiled(inputs).square().sum().backward();control(inputs).square().sum().backward()
        capture.after_backward(step,live);optimizer.step();base.step();capture.after_step(step,live,[optimizer])
        live.zero_grad(set_to_none=True);control.zero_grad(set_to_none=True)
        assert all(torch.equal(p,q) for p,q in zip(live.parameters(),control.parameters()))
    assert capture.metrics['capture_allones_relative_error']<1e-5


def fake_muon_update(grad,momentum,mu=.95):
    momentum.lerp_(grad,1-mu)
    return grad.lerp_(momentum,mu)  # deliberately mutates raw .grad, as PR323 does


def distributed_worker(rank,init_path,directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+init_path,rank=rank,world_size=2)
    try:
        torch.manual_seed(7);live=model();control=copy.deepcopy(live)
        matrix=[p for p in live.parameters() if p.ndim==2];vectors=[p for p in live.parameters() if p.ndim==1]
        opts=[torch.optim.AdamW(vectors,lr=.01),Muon(matrix,lr=.01)]
        base=[torch.optim.AdamW([p for p in control.parameters() if p.ndim==1],lr=.01),
              Muon([p for p in control.parameters() if p.ndim==2],lr=.01)]
        capture=StateCaptureAdapter(live,opts,configuration(1),directory=directory,provenance=provenance())
        inputs=torch.randn(4,3,4);ids=torch.arange(rank*2,rank*2+2)
        capture.before_step(0,live,opts);capture.before_microbatch(0,ids,inputs[ids])
        live(inputs[ids]).square().sum().backward();control(inputs[ids]).square().sum().backward()
        for network in (live,control):
            for p in network.parameters():dist.all_reduce(p.grad)
        raw={n:p.grad.clone()/12 for n,p in live.named_parameters()}
        capture.after_backward(0,live)
        with patch('baseline.optim.muon_update',fake_muon_update):
            for opt in opts+base:opt.step()
        capture.after_step(0,live,opts)
        assert all(torch.equal(p,q) for p,q in zip(live.parameters(),control.parameters()))
        dist.barrier()
        record=StateBankReader(directory).read(1)
        for n,obs in record.snapshot.matrices.items():
            _,g,m,_=recover_banks(obs.rows)
            torch.testing.assert_close(g,raw[n]@obs.anchors.B,rtol=1e-5,atol=1e-6)
            torch.testing.assert_close(m,.05*raw[n]@obs.anchors.B,rtol=1e-5,atol=1e-6)
        if rank==0:assert capture.metrics['capture_allones_relative_error']<1e-5
    finally:dist.destroy_process_group()


def test_two_rank_capture_sums_sensors_and_sharded_momentum(tmp_path):
    mp.spawn(distributed_worker,args=(str(tmp_path/'init'),str(tmp_path/'bank')),nprocs=2,join=True)


def test_tiny_gpt_capture_has_all_global_trajectory_boundaries(tmp_path):
    from baseline.model import GPT
    torch.manual_seed(41)
    live=GPT(vocab_size=16,num_layers=1,model_dim=128).float()
    control=copy.deepcopy(live);opt=torch.optim.AdamW(live.parameters(),lr=.001)
    base=torch.optim.AdamW(control.parameters(),lr=.001)
    config=configuration(1);capture=StateCaptureAdapter(live,[opt],config,directory=tmp_path,provenance=provenance())
    inputs=torch.arange(12).reshape(4,3)%16;targets=(inputs+1)%16
    capture.before_step(0,live,[opt]);capture.before_microbatch(0,torch.arange(4),inputs)
    live(inputs,targets).backward();control(inputs,targets).backward()
    capture.after_backward(0,live);opt.step();base.step();capture.after_step(0,live,[opt])
    assert all(torch.equal(p,q) for p,q in zip(live.parameters(),control.parameters()))
    record=StateBankReader(tmp_path).read(1)
    assert record.snapshot.trajectories.shape==(4,3*2*2)
    assert torch.isfinite(record.snapshot.trajectories).all()
    assert record.snapshot.trajectories.abs().sum()>0
    assert 'norm1.gains' in record.snapshot.vectors
    assert 'blocks.0.mlp.fc.bias' in record.train.vectors
    assert 'norm1.gains' not in capture.editable
    assert 'proj.weight' in capture.editable and 'embed.weight' in capture.editable
    assert capture.metrics['capture_allones_relative_error']<1e-5


def test_capture_config_preserves_original_schedule():
    import pytest
    from baseline.train import validate_config
    config=dict(configuration(),unscored=True,compile=True)
    config['state_bank'].update(baseline_commit='pinned',directory='/temporary/state-bank')
    validate_config(config)
    config['schedule_steps']=1000
    with pytest.raises(ValueError,match='3250'):validate_config(config)


def test_compiled_pure_kernel_capture_matches_twelve_layer_gpt(tmp_path):
    from baseline.model import GPT
    from baseline.capture_execution import enable_capture_execution
    torch.manual_seed(43)
    live=GPT(vocab_size=16,num_layers=12,model_dim=128).float();control=copy.deepcopy(live)
    opt=torch.optim.AdamW(live.parameters(),lr=.001);base=torch.optim.AdamW(control.parameters(),lr=.001)
    config=configuration(2);capture=StateCaptureAdapter(live,[opt],config,directory=tmp_path,provenance=provenance())
    enable_capture_execution(live,backend='eager')
    inputs=torch.arange(12).reshape(4,3)%16;targets=(inputs+1)%16
    graph_counts=[]
    for step in range(2):
        capture.before_step(step,live,[opt]);capture.before_microbatch(step,torch.arange(4),inputs)
        loss=live(inputs,targets);expected=control(inputs,targets)
        torch.testing.assert_close(loss,expected,rtol=0,atol=0)
        loss.backward();expected.backward();capture.after_backward(step,live)
        opt.step();base.step();capture.after_step(step,live,[opt])
        assert all(torch.equal(p,q) for p,q in zip(live.parameters(),control.parameters()))
        live.zero_grad(set_to_none=True);control.zero_grad(set_to_none=True)
        graph_counts.append(torch._dynamo.utils.counters['stats']['unique_graphs'])
    assert graph_counts[0]==graph_counts[1]
    assert capture.metrics['capture_allones_relative_error']<1e-5


def test_finite_kl_capture_omits_only_unused_signed_banks(tmp_path):
    torch.manual_seed(53);original=model();inputs=torch.randn(4,3,4)
    records=[]
    for probes in (4,0):
        live=copy.deepcopy(original);opt=torch.optim.AdamW(live.parameters(),lr=.01)
        config=configuration(1);config['state_bank'].update(train_sequence_probes=probes,audit_sequence_probes=probes)
        prov=provenance();prov['sensor_config']=config['state_bank']
        directory=tmp_path/str(probes)
        capture=StateCaptureAdapter(live,[opt],config,directory=directory,provenance=prov)
        capture.before_step(0,live,[opt]);capture.before_microbatch(0,torch.arange(4),inputs)
        live(inputs).square().sum().backward();capture.after_backward(0,live);opt.step();capture.after_step(0,live,[opt])
        records.append(StateBankReader(directory).read(1))
        assert capture.metrics['capture_allones_relative_error']<1e-5
        assert all(v[0].shape[0]==probes for v in records[-1].train.matrices.values())
    a,b=records
    assert all(torch.equal(a.model[n],b.model[n]) for n in a.model)
    for n in a.snapshot.matrices:
        assert torch.equal(a.snapshot.matrices[n].rows,b.snapshot.matrices[n].rows)
        assert torch.equal(a.snapshot.matrices[n].columns,b.snapshot.matrices[n].columns)
    assert torch.equal(a.snapshot.trajectories,b.snapshot.trajectories)


def test_suffix_observer_replays_prefix_anchor_stream_and_restores_rng():
    from rewriter.state_capture import DistributedStateObserver
    torch.manual_seed(80);live=model();opt=torch.optim.AdamW(live.parameters(),lr=.01)
    cfg=configuration(8);sensor=dict(cfg['state_bank'],anchor_refresh_period=2)
    full=DistributedStateObserver(live,[opt],cfg,sensor_config=sensor)
    # Advance only the anchor schedule: this test concerns private anchor RNG,
    # not measured snapshots or a synthetic GPT trajectory.
    for step in range(5):
        full.before_step(step,live,[opt]);full.release_step(step)
    saved=full.state_dict();rng=torch.get_rng_state().clone()
    suffix=DistributedStateObserver(live,[opt],cfg,sensor_config=sensor,start_step=5)
    assert torch.equal(rng,torch.get_rng_state())
    for n in full.anchors:
        torch.testing.assert_close(full.anchors[n].A,suffix.anchors[n].A,rtol=0,atol=0)
        torch.testing.assert_close(full.anchors[n].B,suffix.anchors[n].B,rtol=0,atol=0)
    assert torch.equal(full.generator.get_state(),suffix.generator.get_state())
    suffix.load_state_dict(saved)
    for step in (5,6):
        full.before_step(step,live,[opt]);suffix.before_step(step,live,[opt])
        for n in full.anchors:torch.testing.assert_close(full.anchors[n].A,suffix.anchors[n].A,rtol=0,atol=0)
        full.release_step(step);suffix.release_step(step)
    full.close();suffix.close()
