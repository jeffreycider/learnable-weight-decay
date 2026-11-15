"""CPU preparation only; finite CUDA equivalence is a separate gate."""
import copy
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from rewriter.offline_tokenwise_graph import TokenwiseGraphAccumulator, validate_training_options
from rewriter.offline_graph_backend import validate_training_options as validate_legacy
from scripts.train_multistate_chisel import aggregate_controller_gradients, sharded_epoch_batches
from tests.test_offline_graph_backend import fake_adapter


def options(**changes):
    values = dict(gpt_precision='fp32', norm_backend='fused', objective='tokenwise-v1',
        actor_backend='transformer', device='cuda', shrinkage_coefficient=1.,
        sequences_per_checkpoint=64, gpt_microbatch=16, checkpoint_batch=40,
        teacher_cache='cache', offline_systems_cache=True, eager_chisel=False,
        compile_gpt=False, gradient_diagnostics=False, no_component_timing=True)
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize('count', [40, 200])
def test_only_tokenwise_batch_allowlist_expands_without_mutating_options(count):
    args = options(checkpoint_batch=count)
    before = vars(args).copy()
    validate_training_options(args)
    assert vars(args) == before
    if count == 40:
        with pytest.raises(ValueError, match='checkpoint_batch'):
            validate_legacy(options(objective='primal-dual', checkpoint_batch=count))


@pytest.mark.parametrize('key,value', [
    ('checkpoint_batch', 8), ('checkpoint_batch', 80), ('checkpoint_batch', 400),
    ('gpt_microbatch', 32), ('sequences_per_checkpoint', 16), ('gpt_precision', 'baseline'),
    ('norm_backend', 'reference'), ('shrinkage_coefficient', 0.), ('objective', 'primal-dual'),
    ('teacher_cache', None), ('offline_systems_cache', False), ('eager_chisel', True),
    ('compile_gpt', True), ('gradient_diagnostics', True), ('no_component_timing', False),
])
def test_all_other_scope_guards_remain_in_force(key, value):
    with pytest.raises(ValueError): validate_training_options(options(**{key: value}))


def test_multibank_index_supplies_cache_without_disabling_systems_cache_guard():
    args = options(teacher_cache=None, multibank_index='verified-index.json')
    validate_training_options(args)
    assert args.teacher_cache is None and args.multibank_index == 'verified-index.json'
    with pytest.raises(ValueError):
        validate_training_options(options(teacher_cache=None, multibank_index='verified-index.json', offline_systems_cache=False))


@pytest.mark.parametrize('state_id', [6, 1006])
def test_actual_capture_receives_one_over40_and_recaptures_when_count_changes(monkeypatch, state_id):
    import rewriter.offline_tokenwise_graph as module
    base = fake_adapter()
    base.engine.microbatch_sequences = 16
    base.constants.load = lambda *args: None
    base.constants.grouped = lambda: object()
    base.constants.radius_squared = lambda grouped: torch.tensor(1.)
    captured = []
    class FakeGraph:
        def __init__(self, *args, **kwargs):
            self.weight = kwargs['state_weight']; captured.append((self.weight, args[8]))
            self.inputs = args[5]
        def capture(self, **kwargs): pass
        def storage_signature(self): return ('fixed',)
        def accumulate_into(self, buffers):
            for buffer in buffers: buffer.add_(self.weight)
            return {}
        def check_finite(self): pass
        def close(self): pass
    monkeypatch.setattr(TokenwiseGraphAccumulator, 'graph_type', FakeGraph)
    monkeypatch.setattr(module, 'scoped_snapshot', lambda snapshot, *args, **kwargs: snapshot)
    monkeypatch.setattr(module, 'ResidentTeacherFeatures', lambda panel, engine: panel)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *args: None)
    adapter = TokenwiseGraphAccumulator(base.actor, base.forward, base.engine, None, base.constants,
                                       radius_fraction=.01, shrinkage_coefficient=1.)
    adapter._metrics = lambda *args: ({}, {})
    panel = SimpleNamespace(inputs=torch.ones(64, 2, dtype=torch.long), features=torch.zeros(64, 2),
                            metadata={'state_id': 6, 'state_sha256': 'verified'})
    record = SimpleNamespace(completed_steps=6, state_id=state_id, snapshot=object(), assert_verified=lambda: 'verified')
    if state_id != 6:
        record.bank_seed = 1
        panel.metadata.update(training_state_id=state_id, bank_seed=1)
    dual = SimpleNamespace(config=object(), multipliers={state_id: 0.})
    for count in (200, 40):
        adapter.begin_update([state_id], global_checkpoint_count=count)
        adapter.accumulate(record, panel, dual)
        for gradient in adapter.finish_update():
            torch.testing.assert_close(gradient, torch.full_like(gradient, 1/count), rtol=0, atol=0)
    assert captured == [(1/200, state_id), (1/40, state_id)]
    assert adapter.capture_count == 2 and adapter.global_count == 40


def test_fixed800_shards_are_five_per_rank_and_twenty_updates():
    states = list(range(800))
    owned = [set(states[rank::8]) for rank in range(8)]
    for epoch in (0, 1):
        batches = sharded_epoch_batches(states, epoch, 400001, 40, 8)
        assert len(batches) == 20 and all(len(batch) == 40 for batch in batches)
        assert sorted(state for batch in batches for state in batch) == states
        for rank in range(8):
            assert all(len(batch[rank::8]) == 5 for batch in batches)
            assert set(state for batch in batches for state in batch[rank::8]) == owned[rank]


def distributed_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://'+rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=45))
    try:
        x = torch.arange(80, dtype=torch.float64).reshape(40, 2)/17-2
        y = torch.sin(torch.arange(40, dtype=torch.float64))*20+5
        for learning_rate in (1e-6, 2e-7):
            torch.manual_seed(741)
            actor = torch.nn.Linear(2, 1).double()
            optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
            # Preserve nonzero existing moments and step; no fresh-Adam shortcut.
            actor(torch.ones(1, 2, dtype=torch.float64)).square().sum().backward()
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            reference = copy.deepcopy(actor)
            reference_optimizer = torch.optim.Adam(reference.parameters(), lr=learning_rate)
            reference_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
            for state in range(rank, 40, 2):
                ((actor(x[state]).squeeze()-y[state]).square()/40).backward()
            assert all(int(state['step']) == 1 for state in optimizer.state.values())
            aggregate_controller_gradients(actor, 2)
            (reference(x).squeeze()-y).square().mean().backward()
            for actual, expected in zip(actor.parameters(), reference.parameters()):
                torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-13, atol=1e-13)
            norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1., error_if_nonfinite=True)
            ref_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 1., error_if_nonfinite=True)
            assert float(norm) > 1
            torch.testing.assert_close(norm, ref_norm, rtol=1e-13, atol=1e-13)
            optimizer.step(); reference_optimizer.step()
            for actual, expected in zip(actor.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=1e-15)
            for actual, expected in zip(optimizer.state.values(), reference_optimizer.state.values()):
                assert int(actual['step']) == int(expected['step']) == 2
                for key in ('exp_avg', 'exp_avg_sq'):
                    torch.testing.assert_close(actual[key], expected[key], rtol=1e-12, atol=1e-15)
    finally:
        dist.destroy_process_group()


def test40_weighted_contributions_sum_before_one_clip_and_saved_adam_step(tmp_path):
    torch.multiprocessing.spawn(distributed_worker, args=(str(tmp_path/'gloo'),), nprocs=2, join=True)
