import pytest
import torch
from rewriter.coordinate_actor_graph import EditableCoordinateOutputs,actor_observation_tensors,CoordinateActorGraph
from rewriter.private_probe_slot import PrivateProbeSlot
from tests.test_private_probe_slot import fixture


def test_filtered_output_vjp_preserves_unused_decoder_none_and_values():
    snapshot,actor,private,_=fixture();name=next(iter(snapshot.matrices));editable={name}
    wrapper=EditableCoordinateOutputs(actor,private,snapshot,editable)
    expected=actor(snapshot,private);actual=wrapper.action(wrapper(torch.zeros(())))
    torch.testing.assert_close(actual.matrices[name].P,expected.matrices[name].P,atol=0,rtol=0)
    torch.testing.assert_close(actual.matrices[name].Q,expected.matrices[name].Q,atol=0,rtol=0)
    parameters=tuple(actor.parameters())+tuple(private.parameters())
    a=torch.autograd.grad(actual.matrices[name].norm_sq(),parameters,allow_unused=True)
    e=torch.autograd.grad(expected.matrices[name].norm_sq(),parameters,allow_unused=True)
    for x,y in zip(a,e):
        assert (x is None)==(y is None)
        if x is not None:torch.testing.assert_close(x,y,atol=0,rtol=0)
    for n,value in actual.vectors.items():assert not value.requires_grad and torch.count_nonzero(value)==0


def test_observation_mask_is_explicit_and_invalid_scope_rejected():
    snapshot,actor,private,_=fixture()
    values=actor_observation_tensors(snapshot)
    assert ('trajectory_mask' in values)==(snapshot.trajectory_mask is not None)
    with pytest.raises(ValueError,match='editable'):EditableCoordinateOutputs(actor,private,snapshot,{'absent'})
    with pytest.raises(ValueError,match='CUDA'):CoordinateActorGraph(actor,PrivateProbeSlot(actor,private),snapshot,set(snapshot.matrices))


def test_opt_in_static_indices_preserve_outputs_vjp_and_checkpoint_schema():
    from dataclasses import replace
    snapshot,actor,private,_=fixture();state_keys=set(actor.state_dict())
    original=actor(snapshot,private);parameters=tuple(actor.parameters())+tuple(private.parameters())
    expected=torch.autograd.grad(original.norm_sq(),parameters,allow_unused=True)
    actor.prepare_static_indices(snapshot);actual=actor(snapshot,private)
    obtained=torch.autograd.grad(actual.norm_sq(),parameters,allow_unused=True)
    assert set(actor.state_dict())==state_keys
    for name in original.matrices:
        torch.testing.assert_close(actual.matrices[name].P,original.matrices[name].P,atol=0,rtol=0)
        torch.testing.assert_close(actual.matrices[name].Q,original.matrices[name].Q,atol=0,rtol=0)
    for x,y in zip(obtained,expected):
        assert (x is None)==(y is None)
        if x is not None:torch.testing.assert_close(x,y,atol=0,rtol=0)
    with pytest.raises(ValueError,match='layout'):
        actor._static_indices(replace(snapshot,trajectories=snapshot.trajectories[:,:-1]),torch.device('cpu'))
    values=next(iter(actor._static_index_cache.values()));values[0].add_(1)
    with pytest.raises(ValueError,match='changed'):actor._static_indices(snapshot,torch.device('cpu'))


def test_completed_vjp_owns_shared_accumulation_before_next_replay():
    from types import SimpleNamespace
    p=torch.nn.Parameter(torch.ones(3));replay_buffer=torch.tensor([1.,2.,3.]);p.grad=replay_buffer
    graph=object.__new__(CoordinateActorGraph);graph.slot=SimpleNamespace(shared_parameters=(p,));graph.pending=True
    graph._backward_complete();replay_buffer.fill_(99.)
    assert not graph.pending
    torch.testing.assert_close(p.grad,torch.tensor([1.,2.,3.]),atol=0,rtol=0)


def test_each_capture_retains_constants_when_same_layout_cache_is_replaced():
    import weakref,gc
    snapshot,actor,private,_=fixture();actor.prepare_static_indices(snapshot)
    graph=object.__new__(CoordinateActorGraph)
    graph._index_owners=actor._static_indices(snapshot,torch.device('cpu'))
    old=weakref.ref(graph._index_owners[0]);expected=graph._index_owners[0].clone()
    actor.prepare_static_indices(snapshot);gc.collect()
    assert old() is not None and old() is not actor._static_indices(snapshot,torch.device('cpu'))[0]
    torch.testing.assert_close(old(),expected,atol=0,rtol=0)


def test_capture_rejects_frame_change_even_when_parameter_addresses_match():
    from types import SimpleNamespace
    snapshot,actor,private,_=fixture()
    graph=object.__new__(CoordinateActorGraph);graph.slot=SimpleNamespace(shared=actor)
    graph._actor_configuration=actor.configuration();graph._actor_tokens=actor.token_counts(snapshot)
    addresses=[p.data_ptr() for p in actor.parameters()]
    graph._check_architecture(snapshot)
    actor.write_observation_frame='decoder'
    assert addresses==[p.data_ptr() for p in actor.parameters()]
    with pytest.raises(RuntimeError,match='new capture'):graph._check_architecture(snapshot)


def test_decoder_observation_static_indices_preserve_exact_filtered_vjp():
    from rewriter.coordinate_chisel import CoordinateChisel
    snapshot,original,private,_=fixture()
    actor=CoordinateChisel(rank=original.rank,trajectory_dim=original.trajectory_dim,
        model_dim=original.model_dim,num_layers=original.num_layers,latent_tokens=original.latent_tokens,
        write_observation_frame='decoder')
    actor.load_state_dict(original.state_dict())
    # Nonzero projections expose cross-token D-read gradients, not just zero
    # output initialization. Read-only actions remain excluded, but their D
    # frames may legitimately affect another token's action through attention.
    with torch.no_grad():
        for name,parameter in actor.named_parameters():
            if 'output' in name or 'blocks.' in name:parameter.add_(torch.randn_like(parameter)*.01)
    name=next(iter(snapshot.matrices));parameters=tuple(actor.parameters())+tuple(private.parameters())
    expected=actor(snapshot,private)
    gradients=torch.autograd.grad(expected.matrices[name].norm_sq(),parameters,allow_unused=True)
    actor.prepare_static_indices(snapshot)
    wrapper=EditableCoordinateOutputs(actor,private,snapshot,{name})
    actual=wrapper.action(wrapper(torch.zeros(())))
    obtained=torch.autograd.grad(actual.matrices[name].norm_sq(),parameters,allow_unused=True)
    torch.testing.assert_close(actual.matrices[name].P,expected.matrices[name].P,atol=0,rtol=0)
    torch.testing.assert_close(actual.matrices[name].Q,expected.matrices[name].Q,atol=0,rtol=0)
    for a,b in zip(obtained,gradients):
        assert (a is None)==(b is None)
        if a is not None:torch.testing.assert_close(a,b,atol=0,rtol=0)
    counts=actor.token_counts(snapshot);old=original.token_counts(snapshot)
    assert counts['total_tokens']==old['total_tokens']+old['write_tokens']


def test_private_capture_failure_restores_shared_gradient_ownership(monkeypatch):
    from types import SimpleNamespace
    from rewriter.coordinate_actor_graph import PrivateActorDispatch
    snapshot,actor,private,_=fixture();parameters=tuple(actor.parameters());parameters[0].requires_grad_(False)
    before=tuple(p.requires_grad for p in parameters)
    def fail(*args,**kwargs):
        assert not any(p.requires_grad for p in parameters)
        raise RuntimeError('capture failed')
    monkeypatch.setattr('rewriter.coordinate_actor_graph.CoordinateActorGraph',fail)
    monkeypatch.setattr('rewriter.edit_scope.editable_names',lambda *args:set(snapshot.matrices))
    context=dict(snapshot=snapshot,engine=SimpleNamespace(parameters={}),grouped=SimpleNamespace(frozen_parameter_names=set()))
    with pytest.raises(RuntimeError,match='capture failed'):
        PrivateActorDispatch(actor,{'gpt':PrivateProbeSlot(actor,private)},{'gpt':context})
    assert tuple(p.requires_grad for p in parameters)==before
    parameters[1].grad=torch.ones_like(parameters[1])
    with pytest.raises(ValueError,match='clean gradient'):
        PrivateActorDispatch(actor,{'gpt':PrivateProbeSlot(actor,private)},{'gpt':context})
    assert torch.equal(parameters[1].grad,torch.ones_like(parameters[1]))


def test_stale_output_and_pending_parameter_mutation_fail_before_vjp():
    from types import SimpleNamespace
    graph=object.__new__(CoordinateActorGraph);p=torch.nn.Parameter(torch.ones(2))
    graph.slot=SimpleNamespace(shared_parameters=(p,),private_parameters=())
    graph.pending=True;graph._generation=2;graph._pending_versions=((p._version,p.data_ptr(),p.requires_grad),)
    with pytest.raises(RuntimeError,match='stale'):graph._backward_started(torch.ones(2),1)
    with torch.no_grad():p.add_(1)
    with pytest.raises(RuntimeError,match='changed before'):graph._backward_started(torch.ones(2),2)
