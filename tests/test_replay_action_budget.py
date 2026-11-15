"""Replay candidates, final commit and delayed credit share one base-step cap."""
import copy

import pytest
import torch

from rewriter.transformer_online import TransformerChiselAdapter
from tests.test_online_preservation import make
from tests.test_transformer_online import _ordinary_step
from tests.test_online_norm_lifecycle import assert_tree


def make_budgeted(directory, base_lr):
    model, optimizer, original = make(directory)
    config = copy.deepcopy(original.config)
    config['transformer_chisel']['action_budget'] = dict(mode='base_motion', fraction=.0001)
    config['transformer_chisel']['replay_maintenance']['steps'] = 2
    actor, actor_optimizer = original.lifecycle.actor, original.lifecycle.optimizer
    original.close()
    optimizer.param_groups[0]['lr'] = base_lr
    adapter = TransformerChiselAdapter(model, [optimizer], config, mode='learned',
                                      actor=actor, actor_optimizer=actor_optimizer)
    model._fp32_sensors = adapter.collector
    return model, optimizer, adapter


@pytest.mark.parametrize('base_lr', [0., .001])
def test_replay_candidates_and_pending_restart_share_budget(tmp_path, monkeypatch, base_lr):
    import rewriter.online_preservation as replay
    original_prepare = replay.prepare_transformer_action
    proposals = []

    def inspect(*args, **kwargs):
        action = original_prepare(*args, **kwargs)
        assert action.action_budget is not None
        budget = action.action_budget
        norm = float((action.scale * action.action.norm_sq().clamp_min(0).sqrt()).detach())
        assert norm <= float(budget['radius']) + 1e-10
        proposals.append((budget['action_step'], float(budget['radius'])))
        return action

    monkeypatch.setattr(replay, 'prepare_transformer_action', inspect)
    model, optimizer, adapter = make_budgeted(tmp_path, base_lr)
    other = None
    try:
        _ordinary_step(model, optimizer, adapter, 2)
        assert len(proposals) == 3  # Two optimized candidates and the regenerated final edit.
        assert len(set(proposals)) == 1
        if base_lr == 0:
            assert proposals[0][1] == 0
            assert adapter.metrics['chisel_realized_edit_norm_grouped'] == 0
        saved = copy.deepcopy(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                   adapter=adapter.state_dict()))
        restored, optimizer2, other = make_budgeted(tmp_path, base_lr)
        restored.load_state_dict(saved['model'])
        optimizer2.load_state_dict(saved['optimizer'])
        other.load_state_dict(saved['adapter'])
        # The next real backward consumes the exact capped action-time graph.
        _ordinary_step(model, optimizer, adapter, 3)
        _ordinary_step(restored, optimizer2, other, 3)
        assert_tree(model.state_dict(), restored.state_dict())
        assert_tree(adapter.lifecycle.state_dict(), other.lifecycle.state_dict())
        assert_tree(adapter.replay.state_dict(), other.replay.state_dict())
    finally:
        adapter.close()
        if other is not None:
            other.close()
