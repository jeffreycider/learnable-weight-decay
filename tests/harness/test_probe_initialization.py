import torch
from chisel.network import PrivateProbes
from chisel.probe_initialization import attention_pairs


def test_paired_learned_maps_represent_compensated_value_output_edit():
    torch.manual_seed(9)
    names=['blocks.0.attn.v.weight','blocks.0.attn.proj.weight']
    snapshot={name:dict(raw=torch.randn(8,8,dtype=torch.float64),consolidated=torch.zeros(8,8,dtype=torch.float64)) for name in names}
    probes=PrivateProbes({name:(8,8) for name in names},2).double()
    bias=torch.randn(8,dtype=torch.float64)
    attention_pairs(probes,snapshot,head_dim=4,seed=13,reference={'blocks.0.attn.v.bias':bias})
    u=probes.write_left[0][:,0].detach()
    value,output=(snapshot[name]['raw'] for name in names)
    scale=1.2
    codes=torch.zeros(2,2,2,2,dtype=torch.float64)
    codes[0,0,0,0]=(scale-1)*(value.T@u).norm()/value.square().mean().sqrt()
    codes[1,0,0,0]=(1/scale-1)*(output@u).norm()/output.square().mean().sqrt()
    edits=probes.decode(codes,snapshot)
    changed_value=value+edits[names[0]]['raw']
    changed_output=output+edits[names[1]]['raw']
    torch.testing.assert_close(changed_output@changed_value,output@value,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(changed_output@bias,output@bias,rtol=1e-12,atol=1e-12)
    sum(item['raw'].square().sum() for item in edits.values()).backward()
    assert all(p.requires_grad and p.grad is not None for bank in (probes.write_left,probes.write_right) for p in bank)


def test_initialization_preserves_rng_and_nonzero_dormant_maps():
    names=['blocks.0.attn.v.weight','blocks.0.attn.proj.weight']
    snapshot={name:dict(raw=torch.zeros(8,8),consolidated=torch.zeros(8,8)) for name in names}
    probes=PrivateProbes({name:(8,8) for name in names},2)
    rng=torch.get_rng_state().clone()
    attention_pairs(probes,snapshot,4,13)
    assert torch.equal(rng,torch.get_rng_state())
    assert all(p.norm()>0 for bank in (probes.write_left,probes.write_right) for p in bank)
