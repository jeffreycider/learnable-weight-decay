import pytest
import torch
from chisel.network import Chisel, PrivateProbes, propose, BANKS
from chisel.transfer import apply_edits
from chisel.preservation import group_norm_change


def example(device='cpu'):
    shapes={'blocks.0.attn.q.weight':(8,8),'blocks.1.mlp.fc.weight':(12,8)}
    snapshots={n:{bank:torch.randn(*shape,device=device) for bank in BANKS} for n,shape in shapes.items()}
    actor=Chisel(2,16,2,full_rank_gains=True,head_dim=4,causal=False,expansion=2).to(device)
    return actor,PrivateProbes(shapes,2).to(device),snapshots


def test_zero_added_branch_preserves_existing_learned_proposal():
    actor,probes,snapshot=example()
    old=Chisel(2,16,2,head_dim=4,causal=False,expansion=2)
    with torch.no_grad():old.proj.weight.normal_(std=.1)
    missing=actor.load_state_dict(old.state_dict(),strict=False)
    assert set(missing.missing_keys)=={'gain_proj.weight','gain_proj.bias'}
    assert not missing.unexpected_keys
    expected=propose(old,probes,snapshot);actual=propose(actor,probes,snapshot)
    for name in actual:
        for side in actual[name]:torch.testing.assert_close(actual[name][side],expected[name][side],rtol=0,atol=0)
    sum(edit['raw'].square().sum() for edit in actual.values()).backward()
    assert actor.gain_proj.weight.grad.norm()>0


def test_independent_full_rank_effective_scalings_and_group_norm():
    actor,probes,snapshot=example()
    with torch.no_grad():
        actor.gain_proj.bias.view(2,-1)[:,0].copy_(torch.tensor([-.25,.10]))
    edits=propose(actor,probes,snapshot)
    for name,item in edits.items():
        effective=snapshot[name]['raw']+snapshot[name]['consolidated']
        torch.testing.assert_close(item['raw'],-.25*effective)
        torch.testing.assert_close(item['consolidated'],.10*effective)
        assert torch.linalg.matrix_rank(item['raw'])>probes.rank
    effective=apply_edits(snapshot,edits)[2]
    assert group_norm_change(snapshot,effective).item()==pytest.approx(-.15,abs=1e-6)


def test_row_column_codes_and_global_context_are_trainable():
    actor,probes,snapshot=example()
    with torch.no_grad():actor.gain_proj.weight.normal_(std=.03)
    edits=propose(actor,probes,snapshot)
    sum(x.square().sum() for e in edits.values() for x in e.values()).backward()
    assert actor.gain_proj.weight.grad.norm()>0
    for bank in [probes.read_left,probes.read_right,probes.write_left,probes.write_right]:
        assert all(p.grad is not None and p.grad.norm()>0 for p in bank)
    first,last=list(snapshot)
    snapshot[last]['raw']=snapshot[last]['raw']*5
    changed=propose(actor,probes,snapshot)
    assert not torch.allclose(edits[first]['raw'],changed[first]['raw'])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_full_rank_gain_compilation_gradient_parity_and_reuse():
    from torch._dynamo.utils import counters
    torch.set_float32_matmul_precision('highest')
    actor,probes,snapshot=example('cuda')
    with torch.no_grad():actor.gain_proj.weight.normal_(std=.01)
    compiled=torch.compile(propose,fullgraph=True,dynamic=False)
    def compute(fn):
        edits=fn(actor,probes,snapshot)
        loss=sum(x.square().sum() for e in edits.values() for x in e.values())
        return edits,torch.autograd.grad(loss,list(actor.parameters())+list(probes.parameters()))
    expected,eg=compute(propose);actual,ag=compute(compiled)
    for name in expected:
        for side in expected[name]:torch.testing.assert_close(actual[name][side],expected[name][side],rtol=3e-4,atol=3e-6)
    for a,b in zip(ag,eg):torch.testing.assert_close(a,b,rtol=5e-4,atol=2e-4)
    count=counters['stats']['unique_graphs']
    for _ in range(3):
        with torch.no_grad():actor.gain_proj.bias.add_(.0001)
        compute(compiled)
    assert counters['stats']['unique_graphs']==count
