"""CPU ownership/dispatch tests; no claim of CUDA capture qualification."""
import copy
from types import SimpleNamespace
import pytest
import torch

from rewriter.coordinate_actor_graph import (CoordinateActorGraph,PrivateActorDispatch,
                                              actor_trainability_signature)
from rewriter.adaptive_actor_graph import EditableAdaptiveOutputs
from rewriter.adaptive_sensing import query_weight_cores
from rewriter.private_probe_slot import PrivateProbeSlot
from scripts.train_coordinate_chisel import (build_actor,build_shared_adam,
    configure_shared_trainability,validate_recipe,ActorDispatch)
from scripts.qualify_heads_only_adaptive_graph import qualification_recipe,frozen_owners
from tests.test_controller import make_snapshot
from tests.test_coordinate_training_driver import recipe


def config():
    r=recipe();r.update(action_family='adaptive_ucv_v1',student_backend='dense',private_inner_steps=8)
    r['architecture'].update(fresh_weight_core=True,frame_mode='tied_orthonormal')
    return r


def fixture():
    torch.set_num_threads(1);torch.manual_seed(404)
    r=config();a=build_actor(r);s=make_snapshot();p=a.make_private(s,state_key=('gpt',0,6),seed=5)
    with torch.no_grad():
        for head in a.output_heads:
            for value in head.parameters():value.normal_(std=.02)
    return r,a,p,s


def test_explicit_qualification_guard_preserves_eager_default_and_precision_scope():
    r=config();r['shared_update_policy']='heads_only';validate_recipe(r)
    r['actor_backend']=r['private_actor_backend']='adaptive_actor_graph'
    with pytest.raises(ValueError,match='heads_only requires eager'):validate_recipe(r)
    r['experimental_heads_only_graph']=True
    with pytest.raises(ValueError,match='qualification-only'):validate_recipe(r)
    r['qualification_only']=True;validate_recipe(r)
    for key,value in [('qualification_only',False),('shared_update_policy','full'),('actor_backend','eager'),('private_actor_backend','eager')]:
        changed=copy.deepcopy(r);changed[key]=value
        with pytest.raises(ValueError,match='qualification-only'):validate_recipe(changed)
    r['architecture']['factor_precision']='fp64'
    with pytest.raises(ValueError,match='qualification-only'):validate_recipe(r)
    r['experimental_heads_only_graph']=1
    with pytest.raises(ValueError,match='explicit boolean'):validate_recipe(r)


def test_prepared_recipe_changes_no_objective_optimizer_or_data_fields():
    old=config();old['shared_optimizer_step']=False
    for graph in (False,True):
        new=qualification_recipe(old,start=160,graph=graph)
        for key in ('objectives','shared_optimizer','private_optimizer','panels','architecture','actor_seed','sampler_seed'):
            assert new[key]==old[key]
        assert new['shared_update_policy']=='heads_only'
        assert new['stopping']['max_updates']==164
        assert new.get('experimental_heads_only_graph',False)==graph
    assert old['shared_optimizer_step'] is False and 'shared_update_policy' not in old


def test_named_capture_mask_distinguishes_heads_frozen_and_rejects_rebind_mutation():
    r,actor,private,snapshot=fixture();slot=PrivateProbeSlot(actor,private)
    configure_shared_trainability(actor,'heads_only')
    signature=(actor_trainability_signature(actor),actor_trainability_signature(private))
    graph=object.__new__(CoordinateActorGraph);graph.slot=slot;graph.capture_mask_signature=signature
    graph._check_trainability()
    with torch.no_grad():next(actor.parameters()).add_(.01)
    graph._check_trainability()  # values change; no recapture per optimizer step
    actor.requires_grad_(False)
    with pytest.raises(RuntimeError,match='trainability mask'):graph._check_trainability()
    configure_shared_trainability(actor,'heads_only');graph._check_trainability()
    next(private.parameters()).requires_grad_(False)
    with pytest.raises(RuntimeError,match='trainability mask'):graph._check_trainability()


def test_heads_only_live_query_vjp_and_frozen_inner_preserve_prior_head_gradients():
    r,actor,private,snapshot=fixture();configure_shared_trainability(actor,'heads_only')
    heads=tuple(p for p in actor.parameters() if p.requires_grad)
    body=tuple(p for p in actor.parameters() if not p.requires_grad)
    weights={n:torch.randn(o.rows.shape[0],o.columns.shape[0]) for n,o in snapshot.matrices.items()}
    editable={next(iter(snapshot.matrices))};name=next(iter(editable))
    wrapper=EditableAdaptiveOutputs(actor,private,snapshot,editable,weights,())
    expected=actor(snapshot,private,weight_cores=query_weight_cores(weights,private))
    actual=wrapper.action(wrapper(torch.zeros(())))
    params=heads+tuple(private.parameters())
    e=torch.autograd.grad(expected.matrices[name].norm_sq(),params,allow_unused=True)
    a=torch.autograd.grad(actual.matrices[name].norm_sq(),params,allow_unused=True)
    for x,y in zip(a,e):
        assert (x is None)==(y is None)
        if x is not None:torch.testing.assert_close(x,y,atol=0,rtol=0)
    assert any(x is not None and x.count_nonzero() for x in a[len(heads):])
    assert all(p.grad is None for p in body)
    for p in heads:p.grad=torch.randn_like(p)
    prior=[p.grad.clone() for p in heads]
    flags=actor_trainability_signature(actor)
    actor.requires_grad_(False)
    # Private-only live graph remains differentiable through frozen shared body.
    wrapper.action(wrapper(torch.zeros(()))).matrices[name].norm_sq().backward()
    assert any(p.grad is not None and p.grad.count_nonzero() for p in private.parameters())
    for p,value in zip(heads,prior):torch.testing.assert_close(p.grad,value,atol=0,rtol=0)
    assert all(p.grad is None for p in body)
    for (_,flag),p in zip(flags,actor.parameters()):p.requires_grad_(flag)
    assert actor_trainability_signature(actor)==flags


def test_private_capture_restores_partial_mask_and_uses_distinct_frozen_program(monkeypatch):
    r,actor,private,snapshot=fixture();configure_shared_trainability(actor,'heads_only')
    original=actor_trainability_signature(actor);captured=[]
    class FakeGraph:
        editable=frozenset(snapshot.matrices)
        def bind_context(self,context):pass
        def bind_scope(self,private,editable):pass
    def capture(actor,slot,context,editable,**kwargs):
        captured.append(actor_trainability_signature(actor))
        assert not any(flag for _,flag in captured[-1])
        return FakeGraph()
    monkeypatch.setattr('rewriter.adaptive_actor_graph.capture_adaptive_actor',capture)
    monkeypatch.setattr('rewriter.edit_scope.editable_names',lambda *args:set(snapshot.matrices))
    context=dict(snapshot=snapshot,engine=SimpleNamespace(parameters={}),grouped=SimpleNamespace(frozen_parameter_names=set()))
    dispatch=PrivateActorDispatch(actor,{'gpt':PrivateProbeSlot(actor,private)},{'gpt':context},backend='adaptive_actor_graph')
    assert actor_trainability_signature(actor)==original and len(captured)==1
    with pytest.raises(RuntimeError,match='freeze shared'):dispatch.for_visit('gpt',context,private)
    actor.requires_grad_(False)
    for _ in range(3):dispatch.for_visit('gpt',context,private)
    assert len(captured)==1 and dispatch.replays==3


def test_joint_dispatch_cannot_fallback_under_wrong_trainability_mask():
    r,actor,private,snapshot=fixture();configure_shared_trainability(actor,'heads_only')
    dispatch=object.__new__(ActorDispatch);dispatch.actor=actor
    dispatch.shared_trainability=actor_trainability_signature(actor)
    dispatch.graphs={'gpt':object()}
    actor.requires_grad_(False)
    with pytest.raises(RuntimeError,match='joint dispatch trainability'):
        dispatch.for_visit('gpt',{},private)


def test_frozen_owner_comparison_includes_adam_moments_not_only_weights():
    r,actor,private,snapshot=fixture();optimizer=build_shared_adam(actor,r['shared_optimizer'])
    for p in actor.parameters():p.grad=torch.ones_like(p)
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    shared=copy.deepcopy(dict(model=actor.state_dict(),optimizer=optimizer.state_dict()))
    view=frozen_owners(shared,actor)
    assert view['optimizer'] and all(v['exp_avg'].count_nonzero() for v in view['optimizer'].values())
    assert not any(n.startswith(('matrix_output.','vector_output.')) for n in view['parameters'])


def test_pending_partial_mask_mutation_is_rejected_before_backward():
    r,actor,private,snapshot=fixture();configure_shared_trainability(actor,'heads_only')
    slot=PrivateProbeSlot(actor,private)
    graph=object.__new__(CoordinateActorGraph);graph.slot=slot;graph.pending=True;graph._generation=1
    graph._pending_versions=tuple((p._version,p.data_ptr(),p.requires_grad) for p in (*slot.shared_parameters,*slot.private_parameters))
    actor.requires_grad_(False)
    with pytest.raises(RuntimeError,match='changed before pending VJP'):
        graph._backward_started(torch.ones(1),1)


def test_submodule_training_flip_rejects_capture_even_with_wrapper_unchanged():
    r,actor,private,snapshot=fixture()
    weights={n:torch.zeros(o.rows.shape[0],o.columns.shape[0]) for n,o in snapshot.matrices.items()}
    wrapper=EditableAdaptiveOutputs(actor,private,snapshot,set(snapshot.matrices),weights,())
    graph=object.__new__(CoordinateActorGraph);graph.slot=PrivateProbeSlot(actor,private)
    graph.wrapper=wrapper;graph._wrapper_training=wrapper.training
    graph._module_training=tuple((n,m.training) for n,m in wrapper.named_modules())
    graph._actor_configuration=actor.configuration();graph._actor_tokens=actor.token_counts(snapshot)
    actor.eval()
    assert wrapper.training==graph._wrapper_training
    with pytest.raises(RuntimeError,match='submodule training mode'):
        graph._check_architecture(snapshot)


def test_new_fp32_residual_is_eager_only_even_with_heads_qualification_flag():
    r=config();r['architecture']['residual_precision']='fp32';validate_recipe(r)
    r=qualification_recipe(config(),start=160,graph=True)
    r['architecture']['residual_precision']='fp32'
    with pytest.raises(ValueError,match='FP32 residual actor graphs'):validate_recipe(r)
    r['architecture']['residual_precision']='bf16';validate_recipe(r)


def test_abba_rejects_wrong_device_overlap_failure_and_order():
    from scripts.qualify_heads_only_adaptive_graph import validate_execution_order
    order=['a','b','b2','a2']
    rows={name:dict(device_uuid='GPU-physical',status='completed',exit_code=0,started=i*3.,ended=i*3.+2.) for i,name in enumerate(order)}
    validate_execution_order(rows,order)
    for field,value in [('device_uuid','GPU-other'),('started',0),('status','failed'),('exit_code',1)]:
        bad=copy.deepcopy(rows);bad['b2'][field]=value
        with pytest.raises(ValueError):validate_execution_order(bad,order)


def test_raw_gradient_trace_rejects_failed_uncommitted_retry(tmp_path):
    import json
    from scripts.qualify_heads_only_adaptive_graph import validate_trace_restart
    path=tmp_path/'trace.jsonl'
    row=dict(update=162,ordinal=0)
    path.write_text(json.dumps(row)+'\n')
    validate_trace_restart(path,162)
    with pytest.raises(ValueError,match='uncommitted'):validate_trace_restart(path,161)
    path.write_text((json.dumps(row)+'\n')*2)
    with pytest.raises(ValueError,match='duplicate'):validate_trace_restart(path,162)
