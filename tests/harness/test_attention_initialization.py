import torch
from baseline.model import Block
from chisel.harness import RunState
from chisel.network import Chisel, PrivateProbes, propose
from chisel.attention_initialization import identity_codes, probes as initialize_probes
from chisel.preservation import group_norm_change
from chisel.transfer import apply_edits


def test_analytic_initialization_preserves_function_and_is_trainable():
    torch.manual_seed(39)
    model = torch.nn.Sequential(Block(16, head_dim=8)).double()
    reference = {n: p.detach().clone() for n, p in model.named_parameters()}
    snapshot = {n: dict(raw=p, consolidated=torch.zeros_like(p),
                       gradient=torch.zeros_like(p), momentum=torch.zeros_like(p),
                       proposal=torch.zeros_like(p)) for n,p in reference.items() if p.ndim == 2}
    actor = Chisel(4, 16, 1, head_dim=8).double()
    private = PrivateProbes({n:p['raw'].shape for n,p in snapshot.items()}, 4).double()
    identity_codes()({}, RunState(model=actor))
    initialize_probes(private, snapshot, 3, reference, actor, head_dim=8)
    effective = apply_edits(snapshot, propose(actor, private, snapshot))[2]
    inputs = torch.randn(2,7,16,dtype=torch.float64)
    before = torch.func.functional_call(model, reference, (inputs,))
    after = torch.func.functional_call(model, reference | effective, (inputs,))
    torch.testing.assert_close(before, after, atol=1e-12, rtol=1e-12)
    loss = group_norm_change(snapshot, effective)
    assert loss < 0
    loss.backward()
    assert actor.proj.weight.grad.norm() > 0
    assert sum(p.grad.norm() for p in private.write_left) > 0
