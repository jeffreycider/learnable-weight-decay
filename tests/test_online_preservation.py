"""Real tiny-GPT replay, final measurement and boundary restart contract."""
import copy
import hashlib
import json
from datetime import timedelta
from pathlib import Path
import numpy as np
import pytest
import torch
from baseline.model import GPT
from baseline.fp32_execution import enable_fp32_execution
from rewriter.transformer_online import TransformerChiselAdapter
from rewriter.online_preservation import validate_replay_maintenance
from rewriter.function_preservation import FunctionalPreservation
from tests.test_transformer_action import FactorActor
from tests.test_transformer_online import _ordinary_step
from tests.test_online_norm_lifecycle import assert_tree


def settings(directory,world=1):
    files=[];manifest=[]
    for i in range(2):
        p=Path(directory)/f'fineweb_train_{i:03d}.bin'
        header=np.zeros(256,dtype='<i4');header[:3]=[20240520,1,100]
        data=(np.arange(100,dtype=np.uint16)+i)%32
        if not p.exists():p.write_bytes(header.tobytes()+data.tobytes())
        files.append(str(p));manifest.append(dict(kind='train',path=str(p),bytes=p.stat().st_size,tokens=100,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    manifest_path=Path(directory)/'data_manifest.json'
    if not manifest_path.exists():manifest_path.write_text(json.dumps(manifest))
    return dict(mode='replay_primal_dual',steps=1,sequences=2,microbatch_sequences=1,seed=744,split_seed=42,
        train_files=files,manifest_path=str(manifest_path),manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        teacher_cache_device='cpu',learning_rate=1e-6,representation_lr_multiplier=1.,gradient_clip=1.,
        shrinkage_coefficient=1.,tolerance=1e-6,rho=.01,dual_lr=.1)


def make(directory,world=1,actual_actor=False):
    torch.set_num_threads(1);torch.manual_seed(100)
    model=enable_fp32_execution(GPT(vocab_size=32,num_layers=1,model_dim=128).float())
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    config=dict(model=dict(vocab_size=32,num_layers=1,model_dim=128),batch_tokens=8,seq_len=4,max_steps=6,schedule_steps=3250,unscored=True,
        gpt_precision='fp32',compile=False,world_size=world,
        transformer_chisel=dict(prefix_steps=2,task_lr=1e-5,representation_lr_multiplier=1.,
            radius_fraction=.001,gradient_clip=1.,actor_seed=5,architecture={},policy_provenance={},
            actor_execution={'backend':'eager'},sensor_config=dict(probe_rank=2,anchor_refresh_period=1,trajectory_positions=4),
            replica_check_every=1,group_norm_metrics_every=1,replay_maintenance=settings(directory,world)))
    if actual_actor:
        from types import SimpleNamespace
        from rewriter.chisel_factory import chisel_architecture,build_chisel
        architecture=chisel_architecture(SimpleNamespace(chisel_model_dim=128,chisel_layers=1,chisel_patch_size=4),rank=2,trajectory_dim=12)
        config['transformer_chisel']['architecture']=architecture
        actor=build_chisel(architecture)
        with torch.no_grad():
            for head in actor.output_heads:head.weight.normal_(std=.01)
    else:actor=FactorActor().float()
    from rewriter.meta_optimizer import build_meta_adam
    actor_opt=build_meta_adam(actor,lr=1e-5)
    adapter=TransformerChiselAdapter(model,[opt],config,mode='learned',actor=actor,actor_optimizer=actor_opt)
    model._fp32_sensors=adapter.collector
    return model,opt,adapter


def trace(directory,rank=0,world=1):
    model,opt,adapter=make(directory,world)
    _ordinary_step(model,opt,adapter,2,rank,world)
    assert adapter.replay.updates==1
    assert adapter.replay.last_panel['action_step']==2
    assert all(sid<4 for sid in adapter.replay.last_panel['ids'])
    if rank==0:
        metrics=adapter.metrics
        assert metrics['chisel_extra_gpt_forwards']==3*(2//world)
        assert metrics['chisel_extra_gpt_backwards']==2//world
        assert metrics['chisel_actor_optimizer_step_calls']==1
        assert adapter.lifecycle.action is not None
        # Independent functional reference for the actual final stored weights.
        panel=adapter.replay.sampler.sample(2,2)
        engine=FunctionalPreservation(model,parameter_state=adapter.lifecycle.action.parameters,
            gpt_precision='fp32',kl_dtype=torch.float64,microbatch_sequences=1)
        teachers=engine.cache_teacher(panel['inputs'],data_role='past_training')
        with torch.no_grad():
            ref=sum(float(engine.student_parameter_loss(dict(model.named_parameters()),t)) for t in teachers)/2
        assert abs(ref-metrics['chisel_replay_final_kl'])<1e-9,(ref,metrics['chisel_replay_final_kl'],world)
    saved=copy.deepcopy(dict(model=model.state_dict(),optimizer=opt.state_dict(),adapter=adapter.state_dict()))
    restored,opt2,adapter2=make(directory,world)
    restored.load_state_dict(saved['model']);opt2.load_state_dict(saved['optimizer'])
    rng=torch.get_rng_state().clone();adapter2.load_state_dict(saved['adapter']);assert torch.equal(rng,torch.get_rng_state())
    for t in range(3,6):
        _ordinary_step(model,opt,adapter,t,rank,world)
        _ordinary_step(restored,opt2,adapter2,t,rank,world)
        assert_tree(model.state_dict(),restored.state_dict());assert_tree(opt.state_dict(),opt2.state_dict())
        assert_tree(adapter.replay.state_dict(),adapter2.replay.state_dict())
        if rank==0:
            assert_tree(adapter.lifecycle.state_dict(),adapter2.lifecycle.state_dict())
            assert adapter.lifecycle.meta_updates==t-2
            assert adapter.replay.updates==min(t-1,3)
            assert int(adapter.lifecycle.optimizer.state[adapter.lifecycle.actor.x]['step'])==(t-2)+min(t-1,3)
        assert not adapter.collector.active and adapter.collector.ctx is None
    assert adapter.replay.blocks==3
    if rank==0:
        assert adapter.metrics['terminal_identity'];assert adapter.metrics['chisel_extra_gpt_forwards']==0
    if world==1:
        bad=copy.deepcopy(saved['adapter']);bad['replay']['updates']+=1
        m,o,a=make(directory,world);m.load_state_dict(saved['model'])
        with pytest.raises(RuntimeError,match='clock'):a.load_state_dict(bad)
        a.close()
    adapter.close();adapter2.close()


def test_actual_gpt_replay_final_kl_resume_and_terminal(tmp_path):trace(tmp_path)


def worker(rank,directory,rendezvous):
    torch.distributed.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2,timeout=timedelta(seconds=60))
    try:
        trace(directory,rank,2)
        from rewriter.online_preservation import _broadcast_action
        from rewriter.transformer_action import prepare_transformer_action
        from tests.test_transformer_action import fixture
        base,snapshot=fixture()
        prepared=prepare_transformer_action(FactorActor().float(),snapshot,base) if rank==0 else None
        import rewriter.online_preservation as replay_module
        original=replay_module._allocate_wire_payload
        def fail_allocation(*args,**kwargs):raise RuntimeError('injected receiver allocation failure')
        if rank==1:replay_module._allocate_wire_payload=fail_allocation
        try:
            with pytest.raises(RuntimeError,match='injected receiver allocation failure'):
                _broadcast_action(prepared,torch.device('cpu'))
        finally:replay_module._allocate_wire_payload=original
        with pytest.raises(RuntimeError,match='rank-zero'):
            _broadcast_action(None,torch.device('cpu'))
    finally:torch.distributed.destroy_process_group()


def test_two_rank_replay_resume(tmp_path):
    settings(tmp_path)
    # Corrupt-clock test constructs collectives; only use it in single-process trace.
    torch.multiprocessing.spawn(worker,args=(str(tmp_path),str(tmp_path/'replay-rendezvous')),nprocs=2,join=True)


def test_settings_fail_closed(tmp_path):
    s=settings(tmp_path);assert validate_replay_maintenance(None) is None
    for k,v in [('steps',0),('teacher_cache_device','auto'),('tolerance',float('nan'))]:
        bad=dict(s);bad[k]=v
        with pytest.raises(ValueError):validate_replay_maintenance(bad)


def test_actual_thin_replay_gradient_includes_radius_scale_vjp(tmp_path):
    from rewriter.online_preservation import _broadcast_action
    from rewriter.transformer_action import build_action_reference_cache
    model,opt,adapter=make(tmp_path)
    _ordinary_step(model,opt,adapter,2)
    lifecycle=adapter.lifecycle;prepared=lifecycle.action;replay=adapter.replay
    assert float(prepared.scale.detach())<1
    panel=replay.sampler.sample(2,2)
    teachers=replay.engine.cache_teacher(panel['inputs'],data_role='past_training')
    proposal=prepared.materialize()
    kl=sum(replay.engine.student_parameter_loss(proposal,t)*t.tokens/8 for t in teachers)
    cache=build_action_reference_cache(prepared.parameters)
    norm=replay.settings['shrinkage_coefficient']*100*cache.grouped.evaluate(proposal).mean_relative_change
    multiplier=replay.dual.multipliers[0];rho=replay.dual.rho;tau=replay.dual.tolerance
    violation=kl.clamp_min(0)/tau-1
    loss=norm+multiplier*violation+.5*rho*violation.clamp_min(0).square()
    expected=torch.autograd.grad(loss,lifecycle.actor.x,retain_graph=True)[0]
    wire=_broadcast_action(prepared,torch.device('cpu'))
    value,gradients=replay._student(wire,teachers,prepared.editable,backward=True)
    assert abs(float(value)-float(kl.detach()))<1e-9
    scale_index=wire[3].index(('','scale'))
    assert float(gradients[scale_index].abs())>0
    outputs=[norm];cots=[torch.ones_like(norm)]
    coefficient=replay.dual.terms(float(value),0)['kl_coefficient']
    for (name,role),g in zip(wire[3],gradients):
        v=prepared.scale if role=='scale' else prepared.action.vectors[name] if role=='v' else getattr(prepared.action.matrices[name],role)
        if v.requires_grad:outputs.append(v);cots.append(coefficient*g)
    got=torch.autograd.grad(outputs,lifecycle.actor.x,grad_outputs=cots,retain_graph=True)[0]
    relative=float((got-expected).norm()/expected.norm())
    assert relative<3e-4,relative
    # Dropping dscale must be observably wrong for this saturated-cap fixture.
    reduced=[(o,c) for o,c in zip(outputs,cots) if o is not prepared.scale]
    wrong=torch.autograd.grad([o for o,c in reduced],lifecycle.actor.x,grad_outputs=[c for o,c in reduced])[0]
    assert float((wrong-expected).norm()/expected.norm())>1e-3
    adapter.close()


def test_retained_snapshot_qualification_command_cpu(tmp_path):
    import subprocess,sys
    model,opt,adapter=make(tmp_path,actual_actor=True)
    _ordinary_step(model,opt,adapter,2)
    state=dict(model=model.state_dict(),controller=adapter.state_dict(),config=adapter.config,step=3,
        data_cursor=dict(file_names=[Path(p).name for p in adapter.replay.settings['train_files']]))
    checkpoint=tmp_path/'retained.pt';torch.save(state,checkpoint)
    setting=tmp_path/'settings.json';setting.write_text(json.dumps(adapter.replay.settings))
    output=tmp_path/'qualification.json'
    command=[sys.executable,'runs/chisel-astra/qualify_replay_maintenance.py','--checkpoint',str(checkpoint),
        '--expected-sha256',hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'--settings',str(setting),
        '--output',str(output),'--device','cpu']
    result=subprocess.run(command,text=True,capture_output=True)
    assert result.returncode==0,result.stdout+result.stderr
    report=json.loads(output.read_text())
    assert report['qualification']['numerical_checks_within_limits']
    assert report['qualification']['regenerated_actor_proposal_and_scale_exact']
    assert report['oracle_extra_actor_forward_calls']==1
    assert report['oracle_actor_backward_calls']==2
    preliminary=json.loads(output.with_suffix('.maintenance.json').read_text())
    assert preliminary['status']=='maintenance_complete_oracle_pending'
    assert preliminary['metrics']==report['metrics']
    assert preliminary['adam_steps_after']==report['adam_steps_after']
    assert report['metrics']['chisel_replay_updates']==1
    # Main block: 2 teachers, 4 student forwards, 2 backwards. Independent
    # oracle: 2 extra teachers, 4 extra students, 4 extra backwards.
    assert report['metrics']['chisel_extra_gpt_forwards']==6
    assert report['rank_records'][0]['oracle_extra_gpt_compute']['teacher_forwards']==2
    assert report['rank_records'][0]['oracle_extra_gpt_compute']['student_backwards']==4
    curve=tmp_path/'curve.json'
    command[command.index('--output')+1]=str(curve)
    command+=['--mode','learning-curve','--qualification-result',str(output),'--curve-updates','4']
    result=subprocess.run(command,text=True,capture_output=True)
    assert result.returncode==0,result.stdout+result.stderr
    curve_result=json.loads(curve.read_text())
    assert [r['update'] for r in curve_result['milestones']]==[1,4]
    counts=curve_result['rank_records'][0]['gpt_compute']
    assert counts['teacher_forwards']==2  # reused across all four updates
    assert counts['student_forwards']==12 and counts['student_backwards']==8
    assert curve_result['ordinary_gpt_steps']==0
    last=torch.load(tmp_path/'curve'/'checkpoints'/'update0004.pt',weights_only=False)
    assert last['dual']['updates']==4 and last['update']==4
    adapter.close()
