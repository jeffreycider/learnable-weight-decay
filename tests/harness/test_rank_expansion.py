import pytest
import torch
from chisel.network import Chisel, PrivateProbes, propose
from chisel.rank_expansion import actor, probes, block_indices


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('gains', [False, True])
def test_preserves_edits_and_new_directions_can_learn(device, gains):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    torch.manual_seed(4)
    kwargs = dict(width=16, depth=1, head_dim=4, expansion=2, causal=False,
                  full_rank_gains=gains, normalize_gains=gains)
    small = Chisel(rank=2, **kwargs).to(device).double()
    large = Chisel(rank=4, **kwargs).to(device).double()
    with torch.no_grad():
        small.proj.weight.normal_(std=.1)
        small.proj.bias.normal_(std=.1)
        if gains:
            small.gain_proj.weight.normal_(std=.1)
            small.gain_proj.bias.normal_(std=.1)
    shapes = {'blocks.0.attn.q.weight': (8, 12), 'blocks.1.mlp.fc.weight': (16, 8)}
    left = PrivateProbes(shapes, 2).to(device).double()
    right = PrivateProbes(shapes, 4).to(device).double()
    snapshot = {name: {bank: torch.randn(*shape, device=device, dtype=torch.float64)
                       for bank in ['raw', 'consolidated', 'gradient', 'momentum', 'proposal']}
                for name, shape in shapes.items()}
    for name, (rows, cols) in shapes.items():
        snapshot[name]['activity'] = dict(input=torch.rand(cols, device=device, dtype=torch.float64),
                                         adjoint=torch.rand(rows, device=device, dtype=torch.float64))
    actor(large, small.state_dict())
    probes(right, left.state_dict())
    expected, actual = propose(small, left, snapshot), propose(large, right, snapshot)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    sum(value.square().sum() for edit in actual.values() for value in edit.values()).backward()
    old_rows = set(block_indices(2, 4, 2))
    new_rows = [index for index in range(32) if index not in old_rows]
    assert large.proj.weight.grad[new_rows].norm() > 0
    assert large.embed.weight.grad.norm() > 0
    if gains:
        new_rows = [3, 4, 7, 8, 12, 13, 16, 17]
        assert large.gain_proj.weight.grad[new_rows].norm() > 0


def test_rejects_rank_reduction():
    small = Chisel(rank=2, width=16, depth=1, head_dim=4)
    large = Chisel(rank=4, width=16, depth=1, head_dim=4)
    with pytest.raises(ValueError, match='nondecreasing'):
        actor(small, large.state_dict())
