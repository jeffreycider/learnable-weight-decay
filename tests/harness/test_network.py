import torch
from chisel.network import Chisel, PrivateProbes, propose, BANKS

def example():
    shapes={'blocks.0.attn.q.weight':(8,8),'blocks.1.mlp.fc.weight':(12,8)}
    snapshots={n:{bank:torch.randn(*shape) for bank in BANKS} for n,shape in shapes.items()}
    actor=Chisel(2,16,2,head_dim=4,causal=False,expansion=2)
    return actor,PrivateProbes(shapes,2),snapshots

def test_zero_head_is_identity_and_trainable():
    actor,probes,snapshot=example()
    edits=propose(actor,probes,snapshot)
    assert all(torch.count_nonzero(x)==0 for x in edits.values())
    sum((edit*snapshot[name]['raw']).sum() for name,edit in edits.items()).backward()
    assert actor.proj.weight.grad.norm()>0

def test_global_context_and_learnable_private_maps():
    actor,probes,snapshot=example()
    with torch.no_grad():actor.proj.weight.normal_(std=.1)
    edits=propose(actor,probes,snapshot)
    first=next(iter(edits));last=list(edits)[-1]
    loss=sum(edit.square().sum() for edit in edits.values());loss.backward()
    assert all(p.grad is not None and p.grad.norm()>0 for p in probes.parameters())
    snapshot[last]['raw']=torch.randn_like(snapshot[last]['raw'])*5
    changed=propose(actor,probes,snapshot)[first]
    assert not torch.allclose(changed,edits[first])
