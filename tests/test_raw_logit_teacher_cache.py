import json
import os
import pytest
import torch
from scripts.experimental_teacher_logits_cache import RawLogitTeacherCache
from scripts.diagnose_cpu_norm_preservation import DiskTeacherCache,load_bound_state,LocalCPUContexts
from rewriter.tokenwise_reference import TokenwiseConfig,TokenwiseDual
from rewriter.grouped_norm import GroupedNormObjective
from rewriter.edit_scope import editable_names,editable_weight_norm_sq
from tests.test_adaptive_preservation_actor import actual_actor_case


def fixture(tmp_path):
    torch.set_num_threads(1)
    engine,actor,private,snapshot,_,inputs=actual_actor_case();inputs=inputs[:,:4];ids=[21,22]
    load_bound_state(engine,{**engine.parameters,**engine.buffers},'tiny-logits')
    grouped=GroupedNormObjective(engine.parameters,zero_reference_policy='freeze')
    DiskTeacherCache.create(engine,inputs,ids,'tiny-logits',tmp_path/'fp64')
    before=engine.compute.teacher_forwards
    meta=RawLogitTeacherCache.create(engine,inputs,ids,'tiny-logits',tmp_path/'logits')
    assert engine.compute.teacher_forwards-before==2 and meta['teacher_compute']['teacher_forwards']==2
    raw=RawLogitTeacherCache(tmp_path/'logits',engine,inputs,ids,'tiny-logits')
    reference=DiskTeacherCache(tmp_path/'fp64',engine,inputs,ids,'tiny-logits')
    return engine,actor,private,snapshot,grouped,inputs,ids,raw,reference


def test_actual_tiny_gpt_teacher_bits_loss_gradients_and_rank_equivalence(tmp_path):
    engine,actor,private,snapshot,grouped,inputs,ids,raw,reference=fixture(tmp_path)
    for i in range(2):assert torch.equal(raw(i,i+1).log_probs,reference(i,i+1).log_probs)
    results=[];gradients=[];editable=editable_names(engine.parameters,'grouped')-grouped.frozen_parameter_names
    config=TokenwiseConfig(mean_tolerance=1e-10,tail_tolerance=1e-10,scale=1e-5,rho=.01,ranking_weight=.01,ranking_k=10,ranking_tie_policy='exact_teacher_ties',ranking_loss_mode='relative_gap')
    dual=TokenwiseDual([6],config=config,dual_lr=.01);dual.multipliers[6]=.1
    for provider in (reference,raw):
        actor.zero_grad(set_to_none=True);private.zero_grad(set_to_none=True)
        context=dict(engine=engine,domain='gpt',local_id=6,snapshot=snapshot,grouped=grouped,inputs=inputs,ids=ids,teacher=provider,norm_squared=editable_weight_norm_sq(engine.parameters,editable))
        results.append(LocalCPUContexts().measure(context,actor,private,config,dual,learning=True,weight=.5))
        gradients.append([None if p.grad is None else p.grad.clone() for m in (actor,private) for p in m.parameters()])
    assert results[0]==results[1]
    for a,b in zip(*gradients):
        assert (a is None)==(b is None)
        if a is not None:assert torch.equal(a,b)
    assert engine.compute.teacher_forwards==4 and engine.compute.student_forwards==4 and engine.compute.student_backwards==4


def test_runtime_representation_input_state_and_mutation_guards(tmp_path,monkeypatch):
    engine,_,_,_,_,inputs,ids,raw,_=fixture(tmp_path)
    with pytest.raises(ValueError,match='input'):RawLogitTeacherCache(tmp_path/'logits',engine,inputs.flip(0),ids,'tiny-logits')
    old_chunk=engine.chunk_tokens;engine.chunk_tokens+=1
    with pytest.raises(ValueError,match='runtime'):raw(0,1)
    with pytest.raises(ValueError,match='runtime'):RawLogitTeacherCache(tmp_path/'logits',engine,inputs,ids,'tiny-logits')
    engine.chunk_tokens=old_chunk
    with monkeypatch.context() as patch:
        patch.setattr(torch,'get_num_threads',lambda:2)
        with pytest.raises(ValueError,match='runtime'):raw(0,1)
    engine._local_cpu_binding=('other-state',engine.teacher_id)
    with pytest.raises(ValueError,match='state/input'):RawLogitTeacherCache(tmp_path/'logits',engine,inputs,ids,'other-state')
    engine._local_cpu_binding=('tiny-logits',engine.teacher_id)
    manifest=tmp_path/'logits'/'manifest.json';meta=json.loads(manifest.read_text());meta['representation']='other';manifest.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='representation'):RawLogitTeacherCache(tmp_path/'logits',engine,inputs,ids,'tiny-logits')
    path=tmp_path/'logits'/'0000.pt';before=path.stat();data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data);os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
    with pytest.raises(ValueError,match='file changed'):raw(0,1)
    engine.teacher_id='new-state';engine._local_cpu_binding=('other','new-state')
    with pytest.raises(ValueError,match='stale'):raw(1,2)


def test_reconstruction_explicitly_cpu_despite_global_default_device():
    from scripts.experimental_teacher_logits_cache import reconstruct_log_probs
    torch.set_num_threads(1);logits=torch.randn(1,5,7,device='cpu')
    expected=torch.log_softmax(logits.double(),dim=-1)
    with torch.device('meta'):actual=reconstruct_log_probs(logits,2)
    assert actual.device.type=='cpu' and torch.equal(actual,expected)


def test_two_thread_streamed_full_pool_gradients_adam_and_subset_alignment(tmp_path):
    import copy
    from scripts.diagnose_cpu_norm_preservation import teacher_cache_class,SubsetTeacherProvider,panel_positions
    engine,actor,private,snapshot,_,inputs=actual_actor_case()
    from rewriter.adaptive_coordinate_chisel import AdaptiveCoordinateChisel
    actor=AdaptiveCoordinateChisel(rank=2,trajectory_dim=12,model_dim=128,num_layers=1,latent_tokens=2,fresh_weight_core=True,factor_precision='fp64',residual_precision='fp32')
    private=actor.make_private(snapshot,state_key=('test',0,6),seed=77)
    with torch.no_grad():
        for head in actor.output_heads:head.weight.normal_(std=.001)
        actor.blocks[0].attn.proj.weight.normal_(std=.001)
    torch.set_num_threads(2)  # Fixture sets one; qualify the actual diagnostic runtime.
    inputs=torch.cat([inputs[:,:4],inputs.flip(0)[:,:4]],dim=0);ids=[21,22,23,24]
    load_bound_state(engine,{**engine.parameters,**engine.buffers},'two-thread')
    grouped=GroupedNormObjective(engine.parameters,zero_reference_policy='freeze')
    editable=editable_names(engine.parameters,'grouped')-grouped.frozen_parameter_names
    config=TokenwiseConfig(mean_tolerance=1e-10,tail_tolerance=1e-10,scale=1e-5,rho=.01,ranking_weight=.01,ranking_k=10,ranking_tie_policy='exact_teacher_ties',ranking_loss_mode='relative_gap')
    initial=[copy.deepcopy(m.state_dict()) for m in (actor,private)];results=[];grads=[];states=[];providers=[]
    for format_name in ('fp64_log_probs','fp32_logits'):
        cls=teacher_cache_class(format_name);path=tmp_path/format_name
        cls.create(engine,inputs,ids,'two-thread',path)
        provider=cls(path,engine,inputs,ids,'two-thread');providers.append(provider)
        for step in (0,1,2):
            positions=panel_positions(4,2,step);subset=SubsetTeacherProvider(provider,positions)
            assert subset.ids==[ids[i] for i in positions]
            for i,pos in enumerate(positions):assert torch.equal(subset(i,i+1).inputs,inputs[pos:pos+1])
        for m,state in zip((actor,private),initial):m.load_state_dict(state);m.zero_grad(set_to_none=True)
        optimizers=[torch.optim.Adam(m.parameters(),lr=1e-5,eps=1e-8) for m in (actor,private)]
        dual=TokenwiseDual([6],config=config,dual_lr=.01);dual.multipliers[6]=.1
        context=dict(engine=engine,domain='gpt',local_id=6,snapshot=snapshot,grouped=grouped,inputs=inputs,ids=ids,teacher=provider,norm_squared=editable_weight_norm_sq(engine.parameters,editable))
        results.append(LocalCPUContexts('direct_streamed').measure(context,actor,private,config,dual,learning=True,weight=.5))
        grads.append([None if p.grad is None else p.grad.clone() for m in (actor,private) for p in m.parameters()])
        for optimizer in optimizers:optimizer.step()
        states.append([p.detach().clone() for m in (actor,private) for p in m.parameters()])
    assert results[0]==results[1]
    for a,b in zip(*grads):
        assert (a is None)==(b is None)
        if a is not None:assert torch.equal(a,b)
    for a,b in zip(*states):assert torch.equal(a,b)
    for i in range(4):assert torch.equal(providers[0](i,i+1).log_probs,providers[1](i,i+1).log_probs)
    assert engine.compute.teacher_forwards==8 and engine.compute.student_forwards==8 and engine.compute.student_backwards==8
    with pytest.raises(ValueError,match='representation'):DiskTeacherCache(tmp_path/'fp32_logits',engine,inputs,ids,'two-thread')
    with pytest.raises(ValueError,match='representation'):RawLogitTeacherCache(tmp_path/'fp64_log_probs',engine,inputs,ids,'two-thread')
    engine.teacher_id='reloaded'
    for provider in providers:
        with pytest.raises(ValueError,match='stale'):SubsetTeacherProvider(provider,[3,1])(0,1)
    torch.set_num_threads(1)


def test_format_resume_identity_and_explicit_source_pin(tmp_path):
    import copy
    from scripts.diagnose_cpu_norm_preservation import resume_contract,restore_training_state,pin_sources,teacher_cache_class
    meta=dict(args={},objective={},dual_lr=.01,cases=[],state_weights=[.5,.5],private_gradient_normalization='undo',pool_schedule={'indices_by_state':{'6':[0,1,3,4],'7':[0,1,3,4]}})
    explicit=copy.deepcopy(meta);explicit['args']['teacher_cache_format']='fp64_log_probs'
    assert resume_contract(meta)==resume_contract(explicit)
    raw=copy.deepcopy(meta);raw['args']['teacher_cache_format']='fp32_logits'
    with pytest.raises(ValueError,match='identity'):restore_training_state({'metadata':meta},None,[],None,{},None,raw,2)
    assert 'scripts/experimental_teacher_logits_cache.py' in pin_sources(tmp_path)
    with pytest.raises(ValueError,match='format'):teacher_cache_class('unknown')


@pytest.mark.parametrize('selected',[None,'fp64_log_probs','fp32_logits'])
def test_diagnostic_cli_selects_cache_before_source_pin(monkeypatch,tmp_path,selected):
    import sys
    from scripts import diagnose_cpu_norm_preservation as d
    original=d.teacher_cache_class;seen=[]
    def select(name):
        seen.append(name)
        return original(name)
    class StopBeforeModel(Exception):pass
    def pin(output):
        assert seen==[selected or 'fp64_log_probs']
        if selected=='fp32_logits':assert 'scripts.experimental_teacher_logits_cache' in sys.modules
        raise StopBeforeModel
    monkeypatch.setattr(d,'teacher_cache_class',select);monkeypatch.setattr(d,'pin_sources',pin)
    argv=['diagnose','--cases','missing.json','--output',str(tmp_path/'out'),'--shared-lr','1e-9','--private-lr','1e-4']
    if selected:argv+=['--teacher-cache-format',selected]
    monkeypatch.setattr(sys,'argv',argv)
    with pytest.raises(StopBeforeModel):d.main()


def test_created_raw_cache_constructor_reuses_verified_producer_checksums(tmp_path,monkeypatch):
    from scripts import diagnose_cpu_norm_preservation as d
    from rewriter.function_preservation import PreservationCompute
    from types import SimpleNamespace
    torch.set_num_threads(1);d._TEACHER_CHECKSUM_CACHE.clear()
    inputs=torch.tensor([[1,2,3,4],[5,6,7,8]]);ids=[10,11]
    engine=SimpleNamespace(teacher_id='synthetic',_local_cpu_binding=('state','synthetic'),
        device=torch.device('cpu'),microbatch_sequences=1,gpt_precision='fp32',kl_dtype=torch.float64,
        chunk_tokens=2,compute=PreservationCompute(),parameters={})
    engine._forward=lambda parameters,x:x.float().unsqueeze(-1).expand(-1,-1,32).contiguous()
    original=d.sha;calls=[]
    def counted(path):calls.append(Path(path));return original(path)
    from pathlib import Path
    monkeypatch.setattr(d,'sha',counted)
    meta=RawLogitTeacherCache.create(engine,inputs,ids,'state',tmp_path/'cache')
    payload_bytes=sum(p.stat().st_size for p in calls)
    assert len(calls)==2 and payload_bytes>0
    provider=RawLogitTeacherCache(tmp_path/'cache',engine,inputs,ids,'state')
    assert len(calls)==2
    assert provider.checksum_stats['hits']==2 and provider.checksum_stats['bytes_hashed']==provider.checksum_stats['files_hashed']==0
    assert engine.compute.teacher_forwards==2
    # Same-size corruption with restored mtime still invalidates producer memo.
    path=tmp_path/'cache'/meta['files'][0]['file'];before=path.stat()
    data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data)
    os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
    with pytest.raises(ValueError,match='file changed'):RawLogitTeacherCache(tmp_path/'cache',engine,inputs,ids,'state')
    assert len(calls)==3
    with pytest.raises(ValueError,match='file changed'):provider(0,1)
    d._TEACHER_CHECKSUM_CACHE.clear()
