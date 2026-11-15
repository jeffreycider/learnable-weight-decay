import copy
import pytest
import torch
from rewriter.coordinate_chisel import CoordinateChisel
from rewriter.private_probe_slot import PrivateProbeSlot
from tests.test_controller import make_snapshot


def fixture():
    torch.set_num_threads(1);torch.manual_seed(106)
    snapshot=make_snapshot();actor=CoordinateChisel(rank=2,trajectory_dim=12,model_dim=128,num_layers=1,latent_tokens=2)
    with torch.no_grad():
        for head in actor.output_heads:head.weight.normal_(std=.01)
    a=actor.make_private(snapshot,state_key=('gpt',0,6),seed=81)
    b=actor.make_private(snapshot,state_key=('gpt',1,6),seed=82)
    return snapshot,actor,a,b


def loss(actor,private,snapshot):
    action=actor(snapshot,private)
    return action.contraction(snapshot.weights,snapshot.vector_weights)+.1*action.norm_sq()


def test_rebind_preserves_addresses_shared_accumulation_and_source_identity():
    snapshot,actor,a,b=fixture();slot=PrivateProbeSlot(actor,a)
    signature=slot.storage_signature();source=copy.deepcopy(b.state_dict())
    for p in actor.parameters():p.grad=torch.ones_like(p)
    shared_before=[p.grad.clone() for p in actor.parameters()]
    slot.stage(source,state_key=b.state_key)
    assert slot.require_bound(b.state_key) is a and slot.storage_signature()==signature
    assert {id(p) for p in slot.shared_parameters}.isdisjoint(map(id,slot.private_parameters))
    torch.testing.assert_close(loss(actor,a,snapshot),loss(actor,b,snapshot),atol=0,rtol=0)
    for before,p in zip(shared_before,actor.parameters()):torch.testing.assert_close(p.grad,before,atol=0,rtol=0)
    exported=slot.export_state()
    assert exported['_extra_state']==source['_extra_state']
    for name,p in a.named_parameters():
        assert exported[name].data_ptr()!=p.data_ptr()
        torch.testing.assert_close(exported[name],source[name],atol=0,rtol=0)


@pytest.mark.parametrize('failure',['identity','schema','shape','dtype','nan','pending_gradient'])
def test_invalid_stage_is_rejected_before_any_mutation(failure):
    snapshot,actor,a,b=fixture();slot=PrivateProbeSlot(actor,a);source=copy.deepcopy(b.state_dict());before=slot.export_state();name=next(iter(dict(a.named_parameters())))
    if failure=='identity':source['_extra_state']['state_key']=('gpt',99,6)
    if failure=='schema':source['_extra_state']['latent_tokens']+=1
    if failure=='shape':source[name]=source[name][:-1]
    if failure=='dtype':source[name]=source[name].double()
    if failure=='nan':source[name].flatten()[0]=float('nan')
    if failure=='pending_gradient':next(a.parameters()).grad=torch.zeros_like(next(a.parameters()))
    with pytest.raises((ValueError,RuntimeError)):slot.stage(source,state_key=b.state_key)
    assert slot.state_key==before['_extra_state']['state_key']
    for n,p in a.named_parameters():torch.testing.assert_close(p,before[n],atol=0,rtol=0)


def test_private_unweighting_is_once_per_visit_and_never_changes_shared_gradients():
    snapshot,actor,a,b=fixture();slot=PrivateProbeSlot(actor,a);weight=1/16
    (weight*loss(actor,a,snapshot)).backward()
    shared=[p.grad.clone() if p.grad is not None else None for p in actor.parameters()]
    private=[p.grad.clone() if p.grad is not None else None for p in a.parameters()]
    slot.unweight_gradients(weight)
    for before,p in zip(private,a.parameters()):
        if before is not None:torch.testing.assert_close(p.grad,before/weight,atol=0,rtol=0)
    for before,p in zip(shared,actor.parameters()):
        if before is not None:torch.testing.assert_close(p.grad,before,atol=0,rtol=0)
    with pytest.raises(RuntimeError,match='already'):slot.unweight_gradients(weight)
    a.zero_grad(set_to_none=True);slot.stage(b.state_dict(),state_key=b.state_key)
    (weight*loss(actor,a,snapshot)).backward();slot.unweight_gradients(weight)
    assert not any(p.grad is not None for p in b.parameters())


def test_one_compiled_graph_across_two_checkpoint_values_and_identity_keys():
    snapshot,actor,a,b=fixture();slot=PrivateProbeSlot(actor,a);states=[copy.deepcopy(a.state_dict()),copy.deepcopy(b.state_dict())];graphs=[]
    def backend(gm,inputs):graphs.append(gm);return gm.forward
    compiled=torch.compile(lambda snap:loss(actor,a,snap),backend=backend,fullgraph=True,dynamic=False)
    for source in [states[0],states[1],states[0]]:
        actor.zero_grad(set_to_none=True);a.zero_grad(set_to_none=True);slot.stage(source,state_key=source['_extra_state']['state_key'])
        expected=loss(actor,a,snapshot);ep=torch.autograd.grad(expected,[*actor.parameters(),*a.parameters()],allow_unused=True)
        actual=compiled(snapshot);ap=torch.autograd.grad(actual,[*actor.parameters(),*a.parameters()],allow_unused=True)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        for x,y in zip(ap,ep):
            assert (x is None)==(y is None)
            if x is not None:torch.testing.assert_close(x,y,atol=0,rtol=0)
    assert len(graphs)==1


def test_replacing_a_container_parameter_is_not_a_valid_static_rebind():
    _,actor,a,_=fixture();slot=PrivateProbeSlot(actor,a)
    a.matrix_rows[0].encoder=torch.nn.Parameter(a.matrix_rows[0].encoder.detach().clone())
    with pytest.raises(RuntimeError,match='addresses'):slot.require_bound(a.state_key)
