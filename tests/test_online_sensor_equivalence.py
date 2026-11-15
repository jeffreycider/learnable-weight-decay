"""Exact actor-feature and live-trajectory equivalence with unused banks omitted."""
import copy
from dataclasses import fields, is_dataclass

import pytest
import torch

from tests.test_transformer_online import _ordinary_step


def same(a, b):
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif is_dataclass(a):
        assert type(a) is type(b)
        for f in fields(a):same(getattr(a, f.name), getattr(b, f.name))
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:same(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):same(x, y)
    else:assert a == b


def fixture(probes, world=1):
    from baseline.model import GPT
    from baseline.fp32_execution import enable_fp32_execution
    from rewriter.transformer_chisel import TransformerChisel
    from rewriter.transformer_online import TransformerChiselAdapter
    torch.set_num_threads(1)
    torch.manual_seed(123)
    model = enable_fp32_execution(GPT(vocab_size=32, num_layers=1, model_dim=128).float())
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    actor = TransformerChisel(rank=2, trajectory_dim=12, model_dim=128, num_layers=1, patch_size=32)
    with torch.no_grad():
        for head in actor.output_heads:head.weight.normal_(std=.002)
    prefix = dict(probe_rank=2, anchor_refresh_period=1, trajectory_positions=4,
                  train_sequence_probes=probes, audit_sequence_probes=probes)
    settings = dict(prefix_steps=0, task_lr=1e-5, representation_lr_multiplier=1.,
                    radius_fraction=.001, gradient_clip=1., actor_seed=5, architecture={}, policy_provenance={},
                    actor_execution={'backend':'eager'}, sensor_config=prefix)
    config = dict(batch_tokens=8, seq_len=4, max_steps=4, schedule_steps=3250, unscored=True,
                  gpt_precision='fp32', compile=False, world_size=world, transformer_chisel=settings)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-5, betas=(.9,.99))
    adapter = TransformerChiselAdapter(model, [opt], config, mode='learned', actor=actor, actor_optimizer=actor_opt)
    model._fp32_sensors = adapter.collector
    captured = []
    original = adapter.observer.snapshot_from_momentum
    def snapshot(*args):
        result = original(*args)
        captured.append(copy.deepcopy(result[0]))
        measurements = result[1]
        assert all(value[0].shape[0] == adapter.collector.nt for value in measurements.matrices.values())
        return result
    adapter.observer.snapshot_from_momentum = snapshot
    return model, opt, adapter, captured


def check_pair(rank=0, world=1):
    full = fixture(4, world)
    task = fixture(0, world)
    assert full[2].collector.K == 9 and task[2].collector.K == 1
    try:
        for step in range(4):
            before = torch.get_rng_state().clone()
            a = _ordinary_step(*full[:3], step, rank, world)
            b = _ordinary_step(*task[:3], step, rank, world)
            same(before, torch.get_rng_state())
            same(a, b)
            same(full[0].state_dict(), task[0].state_dict())
            same(full[1].state_dict(), task[1].state_dict())
            if rank == 0:
                same(full[3][-1], task[3][-1])
                same(full[2].lifecycle.state_dict(), task[2].lifecycle.state_dict())
                assert task[2].metrics['chisel_allones_relative_error'] < 1e-5
                assert task[2].metrics['chisel_group_norm_metric_measured'] == (step == 0)
                assert task[2].metrics['chisel_sequence_gradient_banks'] == 1
                if step == 0:assert task[2].metrics['committed_edit_norm'] > 0
            if step == 0:
                # Resume the omitted-bank observer and pending real actor action.
                saved = copy.deepcopy(dict(model=task[0].state_dict(), optimizer=task[1].state_dict(),
                                           adapter=task[2].state_dict()))
                task[2].close()
                task = fixture(0, world)
                task[0].load_state_dict(saved['model']);task[1].load_state_dict(saved['optimizer'])
                task[2].load_state_dict(saved['adapter'])
        bad = copy.deepcopy(task[2].state_dict())
        bad['settings']['sensor_config']['train_sequence_probes'] = 4
        with pytest.raises(ValueError, match='configuration mismatch'):task[2].load_state_dict(bad)
    finally:
        full[2].close();task[2].close()


def test_real_transformer_features_actions_lag_one_trajectory_and_resume_are_exact():
    check_pair()


def worker(rank, rendezvous):
    import torch.distributed as dist
    dist.init_process_group('gloo', init_method='file://'+rendezvous, rank=rank, world_size=2)
    try:check_pair(rank, 2)
    finally:dist.destroy_process_group()


def test_two_rank_zero_probe_packed_reduction_keeps_exact_actor_features(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(worker, args=(str(tmp_path/'sensor-rendezvous'),), nprocs=2, join=True)
