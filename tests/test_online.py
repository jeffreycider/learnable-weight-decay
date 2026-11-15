from types import SimpleNamespace
import pytest
import torch
from torch import nn
from rewriter.algebra import Anchors, Factors, RawAction, Measurements
from rewriter.online import OnlineRewriter


class TinyController(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(4, 1, dtype=torch.float64))
        self.q = nn.Parameter(torch.zeros(3, 1, dtype=torch.float64))
        self.v = nn.Parameter(torch.zeros(2, dtype=torch.float64))
        self.calls = 0

    def forward(self, snapshot):
        self.calls += 1
        return RawAction({'w': Factors.canonical(self.p, self.q, snapshot.anchors)}, {'v': self.v})


def setup(config=None):
    a = Anchors.random(4, 3, 1, generator=torch.Generator().manual_seed(52), dtype=torch.float64)
    w = torch.arange(12, dtype=torch.float64).view(4, 3) / 12
    v = torch.tensor([1., 2.], dtype=torch.float64)
    snapshot = SimpleNamespace(anchors=a, weights={'w': (w @ a.B, w.T @ a.A)},
                               vector_weights={'v': v}, weight_norm=(w.square().sum() + v.square().sum()).sqrt(),
                               radius=torch.tensor(.1, dtype=w.dtype))
    train = Measurements({'w': (torch.ones(4, 4, 1, dtype=w.dtype), torch.ones(4, 3, 1, dtype=w.dtype))},
                         {'v': torch.ones(4, 2, dtype=w.dtype)})
    opts = {'auxiliary_steps': 2, 'warmup_steps': 1, 'ramp_steps': 1, 'loss_budget': 1.,
            'auxiliary_lr': 1e-3, 'task_lr': 2e-3}
    opts.update(config or {})
    loop = OnlineRewriter(TinyController(), opts)
    return loop, snapshot, train


def test_warmup_consumed_task_credit_and_terminal_adam_counts():
    loop, s, m = setup()
    assert loop.propose(s, m, m, 0, 4) is None
    assert loop.updates == 2 and loop.aux_evaluations == 2
    loop.consume_credit(s.weights, s.vector_weights)
    assert loop.updates == 2
    assert loop.propose(s, m, m, 1, 4) is not None
    assert loop.updates == 4
    loop.consume_credit(s.weights, s.vector_weights)
    assert loop.updates == 5
    assert loop.optimizer.param_groups[0]['lr'] == 2e-3
    assert loop.propose(s, m, m, 2, 4) is not None
    loop.consume_credit(s.weights, s.vector_weights)
    assert loop.updates == 8
    calls = loop.controller.calls
    assert loop.propose(s, m, m, 3, 4) is None
    assert loop.controller.calls == calls
    assert loop.pending is None and loop.metrics['terminal_identity']
    assert loop.aux_evaluations == 6
    assert {int(state['step']) for state in loop.optimizer.state.values()} == {8}


def test_pending_blocks_updates_and_parameter_mutation_is_detected():
    loop, s, m = setup({'warmup_steps': 0})
    pending = loop.propose(s, m, m, 0, 3)
    versions = loop._versions()
    assert pending.controller_versions == versions
    with pytest.raises(RuntimeError, match='consume'):
        loop.propose(s, m, m, 1, 3)
    with torch.no_grad():
        loop.controller.q.add_(.01)
    with pytest.raises(RuntimeError, match='mutated'):
        loop.consume_credit(s.weights, s.vector_weights)


def test_pending_uses_old_anchors_when_next_observation_refreshes():
    loop, s, m = setup({'warmup_steps': 0})
    pending = loop.propose(s, m, m, 0, 3)
    old = pending.action.matrices['w'].anchors
    new = Anchors.random(4, 3, 1, generator=torch.Generator().manual_seed(111), dtype=torch.float64)
    gradient = torch.arange(12, dtype=torch.float64).view(4, 3)
    old_projection = (gradient @ old.B, gradient.T @ old.A)
    new_projection = (gradient @ new.B, gradient.T @ new.A)
    good = pending.action.contraction({'w': old_projection}, s.vector_weights)
    bad = pending.action.contraction({'w': new_projection}, s.vector_weights)
    assert not torch.allclose(good, bad)
    expected = (gradient * pending.action.matrices['w'].dense()).sum() + (s.vector_weights['v'] * pending.action.vectors['v']).sum()
    torch.testing.assert_close(good, expected)
    s.anchors = new
    assert pending.action.matrices['w'].anchors is old
    loop.consume_credit({'w': old_projection}, s.vector_weights)
    assert loop.pending is None


def test_shadow_and_disabled_task_credit_do_not_advance_task_adam():
    loop, s, m = setup({'warmup_steps': 0, 'task_credit': False})
    assert loop.propose(s, m, m, 0, 3, shadow=True) is None
    assert loop.updates == 2
    loop.consume_credit(s.weights, s.vector_weights)
    assert loop.updates == 2
    assert loop.propose(s, m, m, 1, 3) is not None
    loop.consume_credit(s.weights, s.vector_weights)
    assert loop.updates == 4 and loop.pending is None


def test_auxiliary_updates_do_not_access_audit_bank():
    loop, s, m = setup()
    class RejectAudit:
        def penalty(self, action):
            raise AssertionError('warmup auxiliary training must not use audit')
    assert loop.propose(s, m, RejectAudit(), 0, 4) is None


def test_predeclared_warmup_ramp():
    loop, s, m = setup({'auxiliary_steps': 0, 'warmup_steps': 2, 'ramp_steps': 4})
    assert loop.propose(s, m, m, 0, 8) is None
    assert loop.propose(s, m, m, 1, 8) is None
    pending = loop.propose(s, m, m, 2, 8)
    assert pending.warmup == .25
    torch.testing.assert_close(pending.kappa, torch.tensor(.25, dtype=torch.float64))


def test_pr323_3250_schedule_supported_and_invalid_clocks_rejected():
    # Later user correction explicitly supersedes the original 3000-step cap.
    loop, s, m = setup()
    assert loop.propose(s, m, m, 3249, 3250) is None
    for t, total in [(-1, 3), (3, 3), (0, 0)]:
        with pytest.raises(ValueError):
            loop.propose(s, m, m, t, total)
