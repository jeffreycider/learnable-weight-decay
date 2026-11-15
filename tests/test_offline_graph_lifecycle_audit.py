"""Independent systems review of accumulator/optimizer/audit ordering.

The CPU graph double evaluates a real differentiable objective using current
actor weights. It tests lifecycle semantics, not CUDA numerics or throughput.
"""
from types import SimpleNamespace

import torch

from rewriter.offline_graph_backend import OfflineGraphAccumulator


def objective(actor, state):
    x = torch.tensor([[state / 3., -1.], [.5, state / 7.]])
    target = torch.tensor([[.1], [-.2]])
    return (actor(x)-target).square().mean()/200


class GraphDouble:
    def __init__(self, actor, state, inputs):
        self.actor = actor
        self.state = state
        self.inputs = inputs
        self.private = [torch.zeros_like(p) for p in actor.parameters()]
        self.closed = False

    def storage_signature(self):
        return tuple(p.data_ptr() for p in self.actor.parameters())

    def load_checkpoint(self, cache, record, panel, dual, **kwargs):
        assert not self.closed
        self.state = record.completed_steps

    def accumulate_into(self, buffers):
        assert not self.closed
        parameters = tuple(self.actor.parameters())
        gradients = torch.autograd.grad(objective(self.actor, self.state), parameters)
        with torch.no_grad():
            for p, private, total, gradient in zip(parameters, self.private, buffers, gradients):
                private.copy_(gradient)  # replay overwrites private gradients
                p.grad = private
                total.add_(private)
        return {}

    def check_finite(self):
        assert all(torch.isfinite(g).all() for g in self.private)

    def close(self):
        self.actor.zero_grad(set_to_none=True)
        self.closed = True


def test_saved_adam_steps_match_dense_across_reuse_and_audit_close(monkeypatch):
    torch.manual_seed(123)
    actor = torch.nn.Linear(2,1)
    reference = torch.nn.Linear(2,1)
    reference.load_state_dict(actor.state_dict())
    engine = SimpleNamespace(device=torch.device('cuda:0'), gpt_precision='fp32',
                             kl_dtype=torch.float64, microbatch_sequences=16)
    adapter = OfflineGraphAccumulator(actor, actor, engine, None,
        SimpleNamespace(engine=engine), radius_fraction=.01)
    captures = []

    def capture(record, panel, dual):
        adapter._release_graph()
        adapter.graph = GraphDouble(actor, record.completed_steps, panel.inputs)
        adapter._signature = adapter.graph.storage_signature()
        captures.append(adapter.graph)

    monkeypatch.setattr(adapter, '_capture', capture)
    monkeypatch.setattr(adapter, '_metrics', lambda *args: ({}, {}))
    opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    refopt = torch.optim.Adam(reference.parameters(), lr=1e-3)
    panel = SimpleNamespace(inputs=torch.ones(64,2))
    dual = SimpleNamespace(multipliers={1:0., 3:0.})

    # Prime nonempty Adam moments before exercising the adapter lifecycle.
    for model, optimizer in ((actor,opt),(reference,refopt)):
        objective(model,2).backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
    for update in range(3):
        if update == 2:
            adapter.close()  # eager audit may clear grad aliases and use the model
            assert captures[-1].closed
            with torch.no_grad():
                torch.testing.assert_close(actor(torch.ones(1,2)),reference(torch.ones(1,2)),rtol=0,atol=0)
        opt.zero_grad(set_to_none=True);refopt.zero_grad(set_to_none=True)
        adapter.begin_update([1,3],global_checkpoint_count=200)
        for state in (1,3):
            adapter.accumulate(SimpleNamespace(completed_steps=state),panel,dual)
            objective(reference,state).backward()
        published = adapter.finish_update()
        saved = [g.clone() for g in published]
        for p, q, g in zip(actor.parameters(),reference.parameters(),published):
            assert p.grad is g
            torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
        # Metrics/readout work must not erase gradients before the optimizer.
        assert all(torch.equal(g,s) for g,s in zip(published,saved))
        for model, optimizer in ((actor,opt),(reference,refopt)):
            torch.nn.utils.clip_grad_norm_(model.parameters(),.01)
            optimizer.step()
        for p,q in zip(actor.parameters(),reference.parameters()):
            torch.testing.assert_close(p,q,rtol=0,atol=0)
            for key in ('step','exp_avg','exp_avg_sq'):
                torch.testing.assert_close(opt.state[p][key],refopt.state[q][key],rtol=0,atol=0)
        assert len(captures) == (1 if update < 2 else 2)
    assert adapter.compute['graph_training_student_forwards'] == 24
    assert adapter.compute['graph_training_student_backwards'] == 24
    adapter.close()
