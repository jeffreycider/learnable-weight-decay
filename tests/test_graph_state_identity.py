from types import SimpleNamespace
import pytest
import torch

from rewriter.graph_state_identity import training_state_id, validate_graph_panel_identity
from rewriter.graph_offline import NativeThinGradientGraph


def record(**changes):
    values = dict(completed_steps=1, assert_verified=lambda: 'verified')
    values.update(changes)
    return SimpleNamespace(**values)


def panel(**changes):
    values = dict(state_id=1, state_sha256='verified')
    values.update(changes)
    return SimpleNamespace(metadata=values)


def test_legacy_clock_remains_compatible():
    assert training_state_id(record()) == 1
    assert validate_graph_panel_identity(record(), panel()) == (1, 'verified')


def test_global_dual_id_does_not_reinterpret_local_teacher_clock():
    r = record(state_id=1001, bank_seed=1)
    p = panel(training_state_id=1001, bank_seed=1)
    assert validate_graph_panel_identity(r, p) == (1001, 'verified')
    assert r.completed_steps == p.metadata['state_id'] == 1


@pytest.mark.parametrize('change', [
    {'state_id': 1001}, {'state_sha256': 'wrong'},
    {'training_state_id': 1}, {'bank_seed': 2},
])
def test_mismatched_local_or_global_binding_is_rejected(change):
    metadata = dict(training_state_id=1001, bank_seed=1); metadata.update(change)
    with pytest.raises(ValueError): validate_graph_panel_identity(record(state_id=1001, bank_seed=1), panel(**metadata))


def test_failed_namespace_bind_revokes_previous_graph_before_any_copy():
    graph = object.__new__(NativeThinGradientGraph)
    graph.binding_valid = True
    graph.teacher_features = torch.empty(1)
    with pytest.raises(ValueError, match='namespace'):
        graph.load_checkpoint(None, record(state_id=1001, bank_seed=1),
                              panel(training_state_id=1, bank_seed=1), None)
    assert not graph.binding_valid
