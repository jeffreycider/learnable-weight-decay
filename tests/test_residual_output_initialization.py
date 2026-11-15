import dataclasses
import math
import pytest
import torch
from rewriter.grouped_coordinate_chisel import GroupedCoordinateChisel
from scripts.prepare_random_grouped_cpu_seeds import fresh_actor,initialize_residual_outputs,assert_zero_heads
from scripts.fit_cpu_edit_teacher import exact_state_equal
from tests.test_controller import make_snapshot


def tiny_case(mode):
    snapshot=make_snapshot()
    snapshot=dataclasses.replace(snapshot,matrices={'blocks.0.attn.q.weight':snapshot.matrices['first'],'blocks.0.mlp.fc.weight':snapshot.matrices['second']},
        vectors={'blocks.0.attn.q.bias':snapshot.vectors['gain']})
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(671)
        actor=GroupedCoordinateChisel(rank=2,trajectory_dim=12,model_dim=128,num_layers=2,latent_tokens=3,
            factor_precision='fp64',residual_precision='fp32')
    spec=initialize_residual_outputs(actor,mode,672)
    private=actor.make_private(snapshot,state_key=('test',0,6),seed=673)
    return snapshot,actor,private,spec


def test_initializer_changes_only_declared_weights_preserves_rng_private_and_zero_edit():
    torch.set_num_threads(1);torch.manual_seed(670);before=torch.get_rng_state().clone()
    snapshot,zero,private0,_=tiny_case('zero')
    _,random,private1,spec=tiny_case('scaled_random')
    assert torch.equal(before,torch.get_rng_state())
    assert zero.configuration()==random.configuration()
    assert exact_state_equal(private0.state_dict(),private1.state_dict())
    changed={n for n,p in random.named_parameters() if not torch.equal(p,dict(zero.named_parameters())[n])}
    assert changed==set(spec['changed_parameters']) and len(changed)==4
    generator=torch.Generator().manual_seed(672)
    for name,std in spec['std_by_parameter'].items():
        weight=dict(random.named_parameters())[name]
        expected=torch.empty_like(weight).normal_(std=std,generator=generator)
        assert torch.equal(weight,expected)
        assert std==math.sqrt(.33/(weight.shape[1]*2*len(random.blocks)))
        assert torch.count_nonzero(dict(random.named_parameters())[name[:-6]+'bias'])==0
    for actor,private in [(zero,private0),(random,private1)]:
        assert_zero_heads(actor)
        action=actor(snapshot,private)
        assert all(torch.count_nonzero(f.C)==0 for f in action.matrices.values())
        assert all(torch.count_nonzero(v)==0 for v in action.vectors.values())
        for f in action.matrices.values():
            base=torch.arange(f.U.shape[0]*f.V.shape[0],dtype=torch.float32).reshape(f.U.shape[0],f.V.shape[0])
            assert torch.equal((base.double()+f.dense()).float(),base)


def test_same_nonzero_heads_unlock_inner_block_gradients_only_with_random_outputs():
    torch.set_num_threads(1)
    gradients={}
    for mode in ('zero','scaled_random'):
        snapshot,actor,private,_=tiny_case(mode)
        generator=torch.Generator().manual_seed(674)
        with torch.no_grad():
            for bank in actor.output_heads:
                for head in bank.values():head.weight.normal_(std=.001,generator=generator)
        action=actor(snapshot,private)
        terms=[(f.C*torch.linspace(-.7,1.1,f.C.numel(),dtype=f.C.dtype).reshape_as(f.C)).sum() for f in action.matrices.values()]
        terms += [(v*torch.linspace(-.2,.9,v.numel(),dtype=v.dtype)).sum() for v in action.vectors.values()]
        sum(terms).backward()
        gradients[mode]={n:p.grad for n,p in actor.named_parameters() if n.startswith('blocks.') and any(part in n for part in ('.attn.q.weight','.attn.k.weight','.attn.v.weight','.mlp.fc.weight'))}
    assert len(gradients['zero'])==8
    for name,zero_grad in gradients['zero'].items():
        random_grad=gradients['scaled_random'][name]
        assert zero_grad is not None and torch.count_nonzero(zero_grad)==0,name
        assert random_grad is not None and torch.isfinite(random_grad).all() and random_grad.norm()>0,name


def test_fresh_actor_optional_rng_stream_preserves_base_draw_state():
    torch.set_num_threads(1);torch.manual_seed(675);before=torch.get_rng_state().clone()
    zero,rng0=fresh_actor(676)
    random,rng1=fresh_actor(676,residual_output_init='scaled_random',residual_seed=677)
    assert torch.equal(before,torch.get_rng_state()) and torch.equal(rng0,rng1)
    for name,p in zero.named_parameters():
        if not (name.startswith('blocks.') and name.endswith(('.attn.proj.weight','.mlp.proj.weight'))):
            assert torch.equal(p,dict(random.named_parameters())[name]),name
    assert_zero_heads(random)
    with pytest.raises(ValueError,match='fresh zero'):initialize_residual_outputs(random,'scaled_random',677)
