import math
import pytest
import torch
from chisel.preservation import MultiplicativeDual
from chisel.offline import ProbeStore, initialize_fitted_probes
from chisel.harness import RunState
from chisel.storage import sha256


def test_relative_constraint_updates_are_bounded_in_both_directions():
    dual = MultiplicativeDual(100., .1, 1e-12, 1000., minimum=1.)
    dual.update(1e-12)
    assert dual.value == 100.
    dual.update(1e-3)
    assert dual.value == pytest.approx(100. * math.exp(.1))
    dual.update(0.)
    assert dual.value == pytest.approx(100.)
    for _ in range(100):
        dual.update(1e-3)
    assert dual.value == 1000.
    for _ in range(200):
        dual.update(0.)
    assert dual.value == 1.
    with pytest.raises(ValueError):
        MultiplicativeDual(0., .1, 1e-12, 1000.)


def test_stage_initialization_preserves_private_adam_and_pins_source(tmp_path):
    cfg = dict(seed=3, rank=2, optimizer=dict(_target_='torch.optim.Adam', lr=.001))
    old = ProbeStore(tmp_path/'old', cfg, 'cpu')
    snapshot = dict(w=dict(raw=torch.randn(4, 4)))
    probes, optimizer = old.load('a'*64, snapshot)
    sum(p.square().sum() for p in probes.parameters()).backward()
    optimizer.step()
    old.save('a'*64, 5, probes, optimizer)
    source = tmp_path/'source.pt'
    torch.save(dict(extras=dict(private_probes=old.state_dict())), source)
    new = ProbeStore(tmp_path/'new', cfg, 'cpu')
    state = RunState(extras=dict(private_probes=new))
    initialize_fitted_probes(str(source), sha256(source))({}, state)
    fitted, adam = new.load('a'*64, snapshot)
    for actual, expected in zip(fitted.parameters(), probes.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(item['step'] == 1 for item in adam.state.values())
    with pytest.raises(ValueError, match='SHA256'):
        initialize_fitted_probes(str(source), 'bad')({}, state)
    before = new.state_dict()
    initialize_fitted_probes('unused-on-resume', 'unused')({'resume': str(source)}, state)
    assert new.state_dict() == before
