import pytest
import torch
from chisel.network import PrivateProbes
from chisel.probe_initialization import orthonormal


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA required'))])
def test_full_rank_maps_represent_arbitrary_dense_edits_and_remain_trainable(device):
    torch.manual_seed(21)
    probes = PrivateProbes({'matrix': (7, 5)}, rank=5).to(device).double()
    rng = torch.random.get_rng_state().clone()
    orthonormal(probes, {}, 0)
    torch.testing.assert_close(rng, torch.random.get_rng_state(), rtol=0, atol=0)
    for p in probes.parameters():
        torch.testing.assert_close(p.T @ p, torch.eye(5, device=device, dtype=torch.double))
        assert p.requires_grad
    # The trainable left map supplies the missing rows of a rectangular edit.
    desired = torch.randn(7, 5, device=device, dtype=torch.double)
    right = probes.write_right[0]
    with torch.no_grad():
        probes.write_left[0].copy_(desired @ right)
    decoded = probes.write_left[0] @ right.T
    torch.testing.assert_close(decoded, desired)
    decoded.square().sum().backward()
    assert probes.write_left[0].grad.norm() > 0
    assert probes.write_right[0].grad.norm() > 0


def test_orthogonal_initialization_does_not_overwrite_saved_learned_maps(tmp_path):
    from chisel.offline import ProbeStore
    cfg = dict(rank=3, seed=99, initialize=dict(name='chisel.probe_initialization.orthonormal', kwargs={}),
               optimizer=dict(_target_='torch.optim.Adam', lr=.01), cache_bytes=0)
    snapshot = {'matrix': {'raw': torch.randn(5, 4)}}
    store = ProbeStore(tmp_path, cfg, 'cpu')
    probes, opt = store.load('abcdef1234567890', snapshot)
    sum(p.sum() for p in probes.parameters()).backward()
    opt.step()
    store.save('abcdef1234567890', 1, probes, opt)
    loaded, other = store.load('abcdef1234567890', snapshot)
    torch.testing.assert_close(loaded.state_dict(), probes.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(other.state_dict(), opt.state_dict(), rtol=0, atol=0)
