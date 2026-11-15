import copy
from dataclasses import replace
import pytest
import torch
from rewriter.adaptive_coordinate_chisel import AdaptiveCoordinateChisel
from rewriter.grouped_coordinate_chisel import GroupedCoordinateChisel, MATRIX_GROUPS, VECTOR_GROUPS, functional_head_group, head_key
from rewriter.adaptive_sensing import query_weight_cores
from rewriter.grouped_norm import functional_groups
from rewriter.cifar_adapter import observe_cifar_terminal
from tests.test_adaptive_preservation_actor import actual_actor_case
from tests.test_cifar_adapter import fixture as cnn_fixture


def gpt_case(precision='fp32'):
    engine,_,_,snapshot,grouped,_=actual_actor_case()
    kwargs=dict(rank=2,trajectory_dim=12,model_dim=128,num_layers=1,latent_tokens=2,
                fresh_weight_core=True,factor_precision=precision,residual_precision='fp32')
    torch.manual_seed(181);actor=GroupedCoordinateChisel(**kwargs)
    private=actor.make_private(snapshot,state_key=('group-test',6),seed=19)
    return snapshot,actor,private,engine.parameters,kwargs


def action(actor,s,p,w):
    return actor(s,p,weight_cores=query_weight_cores(w,p))


def test_routing_matches_gpt_and_airbench_objectives_and_rejects_unknowns():
    s,a,p,w,_=gpt_case()
    for group,names in functional_groups(w).items():
        for name in names:
            t=w[name]
            assert functional_head_group(name,'matrix' if t.ndim==2 else 'vector',t.numel())==group
    _,engine,x,y=cnn_fixture()
    for group,names in engine.groups.items():
        for name in names:
            t=engine.parameters[name]
            assert functional_head_group(name,'matrix' if t.ndim==2 else 'vector',t.numel())==group
    for name,kind,size in [('whiten.weight','matrix',48),('layers.1.norm1.bias','vector',4),('rms_norm.weight','vector',8),('gain','vector',8),('scale','vector',1)]:
        assert functional_head_group(name,kind,size) is None
    for name,kind in [('unknown.weight','matrix'),('unknown.bias','vector'),('bias','vector'),('layers.1.conv3.weight','matrix')]:
        with pytest.raises(ValueError,match='unknown editable'):functional_head_group(name,kind,10)
    assert a.output_head('head.weight') is a.matrix_output[head_key('head.weight')]


@pytest.mark.parametrize('precision',['fp32','fp64'])
def test_exact_trunk_rng_zero_heads_gradients_dtype_and_strict_roundtrip(precision):
    s,a,p,w,kwargs=gpt_case(precision)
    torch.manual_seed(181);legacy=AdaptiveCoordinateChisel(**kwargs);rng=torch.get_rng_state().clone()
    torch.manual_seed(181);fresh=GroupedCoordinateChisel(**kwargs)
    assert torch.equal(rng,torch.get_rng_state())
    for n,v in legacy.state_dict().items():
        if n!='_extra_state' and not n.startswith(('matrix_output.','vector_output.')):
            torch.testing.assert_close(v,a.state_dict()[n],rtol=0,atol=0)
    out=action(a,s,p,w);assert out.norm_sq()==0
    dtype=torch.float64 if precision=='fp64' else torch.float32
    assert all(v.dtype==dtype for bank in a.output_heads for v in bank.parameters())
    assert all(v.dtype==torch.float32 for v in a.blocks.parameters())
    assert all(f.C.dtype==dtype for f in out.matrices.values())
    out.contraction({n:torch.randn_like(f.dense()) for n,f in out.matrices.items()},
                    {n:torch.randn_like(v) for n,v in out.vectors.items()}).backward()
    used=set()
    for n,o in s.matrices.items():used.add(functional_head_group(n,'matrix',o.rows.shape[0]*o.columns.shape[0]))
    for n,o in s.vectors.items():used.add(functional_head_group(n,'vector',o.weight.numel()))
    for group in used-{None}:assert a.output_head(group).weight.grad.norm()>0
    fresh.load_state_dict(a.state_dict())
    again=action(fresh,s,p,w)
    for n,t in out.dense().items():torch.testing.assert_close(t,again.dense()[n],rtol=0,atol=0)
    for target,source in [(a,legacy),(legacy,a)]:
        with pytest.raises(ValueError,match='architecture'):target.load_state_dict(source.state_dict(),strict=False)
    bad=copy.deepcopy(a.state_dict());key=next(n for n in bad if n.startswith('matrix_output.') and n.endswith('.weight'));bad[key]=bad[key].to(torch.float16)
    with pytest.raises(ValueError,match='precision'):a.load_state_dict(bad)


@pytest.mark.parametrize('precision',['fp32','fp64'])
def test_explicit_shared_head_copy_matches_writable_outputs_and_all_global_features(precision):
    s,a,p,w,kwargs=gpt_case(precision)
    torch.manual_seed(181);legacy=AdaptiveCoordinateChisel(**kwargs)
    with torch.no_grad():
        legacy.matrix_output.weight.normal_(std=.02);legacy.matrix_output.bias.normal_(std=.01)
        legacy.vector_output.weight.normal_(std=.02);legacy.vector_output.bias.normal_(std=.01)
        for group in MATRIX_GROUPS:a.output_head(group).load_state_dict(legacy.matrix_output.state_dict())
        for group in VECTOR_GROUPS:a.output_head(group).load_state_dict(legacy.vector_output.state_dict())
    x=action(legacy,s,p,w);y=action(a,s,p,w)
    restored=GroupedCoordinateChisel(**kwargs)
    restored.load_state_dict(a.state_dict())
    z=action(restored,s,p,w)
    for n,t in y.dense().items():torch.testing.assert_close(t,z.dense()[n],rtol=0,atol=0)
    for n,o in s.matrices.items():
        torch.testing.assert_close(x.matrices[n].C,y.matrices[n].C,rtol=2e-6,atol=2e-7)
    for n,o in s.vectors.items():
        if functional_head_group(n,'vector',o.weight.numel()) is not None:
            torch.testing.assert_close(x.vectors[n],y.vectors[n],rtol=2e-6,atol=2e-7)
        else:assert y.vectors[n].count_nonzero()==0


def test_global_influence_across_different_heads_and_readonly_cnn_observations():
    _,engine,images,labels=cnn_fixture()
    snapshot,_=observe_cifar_terminal(engine,images,labels,momentum={},anchor_seed=87,rank=2,trajectory_images=4)
    actor=GroupedCoordinateChisel(rank=2,trajectory_dim=100,model_dim=128,num_layers=1,latent_tokens=2,fresh_weight_core=True,factor_precision='fp64')
    private=actor.make_private(snapshot,state_key=('cnn',1),seed=19)
    weights={n:p.flatten(1) if p.ndim==4 else p for n,p in engine._all_parameters.items()}
    with torch.no_grad():
        for bank in actor.output_heads:
            for head in bank.values():head.weight.normal_(std=.02)
        actor.blocks[0].attn.proj.weight.normal_(std=.03)
    first=action(actor,snapshot,private,weights)
    for name in set(engine._all_parameters)-set(engine.parameters):
        t=first.matrices[name].dense() if name in first.matrices else first.vectors[name]
        assert t.count_nonzero()==0
    obs=snapshot.matrices['whiten.weight']
    changed=replace(snapshot,matrices={**snapshot.matrices,'whiten.weight':replace(obs,rows=obs.rows+10.,columns=obs.columns+10.)})
    second=action(actor,changed,private,weights)
    assert (first.matrices['head.weight'].C-second.matrices['head.weight'].C).norm()>1e-8
    # Editable conv observation globally influences a different functional head.
    name=next(n for n in snapshot.matrices if n.endswith('.conv1.weight'));obs=snapshot.matrices[name]
    changed=replace(snapshot,matrices={**snapshot.matrices,name:replace(obs,rows=obs.rows+10.,columns=obs.columns+10.)})
    third=action(actor,changed,private,weights)
    assert (first.matrices['head.weight'].C-third.matrices['head.weight'].C).norm()>1e-8
    third.contraction({n:torch.randn_like(f.dense()) for n,f in third.matrices.items()},
                      {n:torch.randn_like(v) for n,v in third.vectors.items()}).backward()
    assert private.matrix_rows[0].raw.grad.norm()>0
    assert actor.weight_core_input.weight.grad.norm()>0
    # One exact actor also consumes GPT observations with the same padded
    # trajectory width; GPT proj.weight and CNN head.weight share one Linear.
    gpt_engine,_,_,gpt_snapshot,_,_=actual_actor_case()
    gpt_snapshot=replace(gpt_snapshot,trajectories=torch.zeros(2,100),trajectory_mask=None)
    gpt_private=actor.make_private(gpt_snapshot,state_key=('gpt',6),seed=31)
    shared_head=actor.output_head('head.weight')
    gpt_action=action(actor,gpt_snapshot,gpt_private,gpt_engine.parameters)
    assert 'proj.weight' in gpt_action.matrices
    assert actor.output_head('head.weight') is shared_head
