import pytest
import torch
from chisel.network import Chisel, PrivateProbes, propose
from chisel.probe_initialization import zero_write_left
from chisel.preservation import group_norm_change
from chisel.transfer import apply_edits


def fixture():
    torch.manual_seed(31)
    actor = Chisel(2, 8, 1, head_dim=4, causal=False)
    probes = PrivateProbes({'weight': (4, 6)}, 2)
    raw = torch.randn(4, 6)
    snapshot = {'weight': {key: raw.clone() if key == 'raw' else torch.zeros_like(raw)
        for key in ('raw', 'consolidated', 'gradient', 'momentum', 'proposal')}}
    return actor, probes, snapshot


def test_zero_initial_edit_has_nonzero_learnable_shrinkage_derivative():
    actor, probes, snapshot = fixture()
    torch.nn.init.normal_(actor.proj.weight, std=.02)
    before_rng = torch.get_rng_state().clone()
    zero_write_left(probes, snapshot, seed=13, actor=actor)
    assert torch.equal(before_rng, torch.get_rng_state())
    edits = propose(actor, probes, snapshot)
    assert all(value.count_nonzero() == 0 for item in edits.values() for value in item.values())
    _, _, effective = apply_edits(snapshot, edits)
    group_norm_change(snapshot, effective).backward()
    assert probes.write_left[0].grad.norm() > 0
    with torch.no_grad():
        probes.write_left[0].sub_(.01 * probes.write_left[0].grad)
    _, _, effective = apply_edits(snapshot, propose(actor, probes, snapshot))
    assert group_norm_change(snapshot, effective) < 0


def test_zero_output_backbone_cannot_silently_dead_initialize_private_maps():
    actor, probes, snapshot = fixture()
    with pytest.raises(ValueError, match='prevent learning'):
        zero_write_left(probes, snapshot, seed=13, actor=actor)


def test_fitted_zero_initialized_probes_are_not_reset_on_reload(tmp_path):
    from chisel.offline import ProbeStore
    actor, _, snapshot = fixture()
    torch.nn.init.normal_(actor.proj.weight, std=.02)
    config = dict(seed=4, rank=2, optimizer=dict(_target_='torch.optim.Adam', lr=.001),
                  initialize=dict(name='chisel.probe_initialization.zero_write_left', kwargs={}))
    store = ProbeStore(tmp_path, config, 'cpu')
    probes, opt = store.load('b'*64, snapshot, actor=actor)
    with torch.no_grad():
        probes.write_left[0].add_(.01)
    store.save('b'*64, 0, probes, opt)
    restored, _ = store.load('b'*64, snapshot, actor=actor)
    torch.testing.assert_close(restored.write_left[0], probes.write_left[0], rtol=0, atol=0)
