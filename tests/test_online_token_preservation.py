"""Experimental tokenwise online replay: real GPT VJP, DDP, and pending restart."""
import copy
from dataclasses import asdict
from datetime import timedelta

import pytest
import torch

from rewriter.online_preservation import validate_replay_maintenance, _broadcast_action
from rewriter.tokenwise_reference import TokenwiseConfig
from rewriter.token_preservation import token_preservation_terms
from rewriter.ranking_loss import topk_margin_loss
from rewriter.transformer_online import TransformerChiselAdapter
from rewriter.transformer_action import build_action_reference_cache
from tests.test_online_preservation import make, settings
from tests.test_transformer_online import _ordinary_step
from tests.test_online_norm_lifecycle import assert_tree


def make_tokenwise(directory, world=1, dual_mode='unclipped_scaled'):
    model, optimizer, old = make(directory, world)
    config = copy.deepcopy(old.config)
    replay = config['transformer_chisel']['replay_maintenance']
    replay['tokenwise_objective'] = asdict(TokenwiseConfig(
        mean_tolerance=replay['tolerance'], tail_tolerance=replay['tolerance'],
        scale=1e-5, rho=replay['rho'], ranking_weight=300., margin_fraction=.5,
        dual_update_mode=dual_mode))
    actor, meta = (old.lifecycle.actor, old.lifecycle.optimizer) if old.lifecycle is not None else (None, None)
    old.close()
    adapter = TransformerChiselAdapter(model, [optimizer], config, mode='learned',
                                      actor=actor, actor_optimizer=meta)
    model._fp32_sensors = adapter.collector
    return model, optimizer, adapter


def test_settings_reject_unsupported_backend_and_conflicting_tolerances(tmp_path):
    s = settings(tmp_path)
    s['tokenwise_objective'] = asdict(TokenwiseConfig(s['tolerance'], s['tolerance'], 1e-5, s['rho']))
    assert validate_replay_maintenance(s) == s
    with pytest.raises(ValueError, match='tolerance/rho'):
        validate_replay_maintenance(dict(s, tolerance=1e-4))
    with pytest.raises(ValueError, match='eager backend'):
        validate_replay_maintenance(dict(s, teacher_cache_device='device',
                                       student_backend='compiled_student_v1_experimental'))
    unqualified = copy.deepcopy(s)
    unqualified['tokenwise_objective']['loss_backend'] = 'tiled_fp64_v1'
    with pytest.raises(ValueError):
        validate_replay_maintenance(unqualified)


def test_one_backward_thin_vjp_matches_full_gpt_objective(tmp_path):
    model, optimizer, adapter = make_tokenwise(tmp_path)
    try:
        _ordinary_step(model, optimizer, adapter, 2)
        lifecycle, replay = adapter.lifecycle, adapter.replay
        prepared = lifecycle.action
        teachers = replay.engine.cache_teacher(replay.sampler.sample(2, 2)['inputs'], data_role='past_training')
        proposal = prepared.materialize()
        cache = build_action_reference_cache(prepared.parameters)
        norm = 100 * cache.grouped.evaluate(proposal).mean_relative_change
        config = replay.tokenwise_config
        replay.dual.multipliers[0] = .4
        loss = norm
        for teacher in teachers:
            logits = replay.engine._forward(proposal, teacher.inputs).double()
            lp = logits.log_softmax(-1)
            target = teacher.log_probs
            kl = (target.exp() * (target - lp)).sum(-1)
            rank = topk_margin_loss(target, logits, k=10, margin_fraction=config.margin_fraction)
            terms = token_preservation_terms(kl, multiplier=.4, mean_tolerance=config.mean_tolerance,
                tail_tolerance=config.tail_tolerance, scale=config.scale, rho=config.rho,
                ranking_loss=rank, ranking_weight=config.ranking_weight)
            loss = loss + terms['total'].sum() / 8
        expected = torch.autograd.grad(loss, lifecycle.actor.x, retain_graph=True)[0]
        wire = _broadcast_action(prepared, torch.device('cpu'))
        replay._sync_tokenwise_multiplier()
        _, grads = replay._student(wire, teachers, prepared.editable, backward=True)
        replay._reduce_tokenwise_metrics()
        outputs, cots = [norm], [torch.ones_like(norm)]
        for (name, role), grad in zip(wire[3], grads):
            value = prepared.scale if role == 'scale' else prepared.action.vectors[name] if role == 'v' else getattr(prepared.action.matrices[name], role)
            if value.requires_grad:
                outputs.append(value)
                cots.append(grad)  # No second multiplier applied to tokenwise gradients.
        actual = torch.autograd.grad(outputs, lifecycle.actor.x, grad_outputs=cots)[0]
        assert float((actual - expected).norm() / expected.norm()) < 4e-4
        assert replay._last_token_metrics['token_count'] == 8
        assert replay._last_token_metrics['ranking_penalty'] >= 0
    finally:
        adapter.close()


def trace(directory, rank=0, world=1, dual_mode='unclipped_scaled'):
    model, optimizer, adapter = make_tokenwise(directory, world, dual_mode)
    restored_adapter = None
    try:
        _ordinary_step(model, optimizer, adapter, 2, rank, world)
        if rank == 0:
            metrics = adapter.metrics
            assert metrics['chisel_extra_gpt_forwards'] == 3 * (2 // world)
            assert metrics['chisel_extra_gpt_backwards'] == 2 // world
            token = metrics['chisel_replay_final_token_metrics']
            assert token['token_count'] == 8
            assert token['maximum_token_kl'] >= metrics['chisel_replay_final_kl']
            assert 0 <= token['argmax_changes'] <= token['top10_order_changes'] <= 8
        saved = copy.deepcopy(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), adapter=adapter.state_dict()))
        restored, other_opt, restored_adapter = make_tokenwise(directory, world, dual_mode)
        restored.load_state_dict(saved['model']); other_opt.load_state_dict(saved['optimizer'])
        restored_adapter.load_state_dict(saved['adapter'])
        for step in range(3, 6):
            _ordinary_step(model, optimizer, adapter, step, rank, world)
            _ordinary_step(restored, other_opt, restored_adapter, step, rank, world)
            assert_tree(model.state_dict(), restored.state_dict())
            assert_tree(adapter.replay.state_dict(), restored_adapter.replay.state_dict())
            if rank == 0:
                assert_tree(adapter.lifecycle.state_dict(), restored_adapter.lifecycle.state_dict())
        if rank == 0:
            assert adapter.metrics['terminal_identity']
            assert adapter.metrics['chisel_extra_gpt_forwards'] == 0
    finally:
        adapter.close()
        if restored_adapter is not None: restored_adapter.close()


@pytest.mark.parametrize('dual_mode', ['unclipped_scaled', 'clipped_relative'])
def test_pending_credit_restart_and_terminal_identity(tmp_path, dual_mode):
    trace(tmp_path, dual_mode=dual_mode)


def worker(rank, directory, rendezvous):
    torch.distributed.init_process_group('gloo', init_method='file://' + rendezvous,
        rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        trace(directory, rank, 2)
    finally:
        torch.distributed.destroy_process_group()


def test_two_rank_tokenwise_replay_and_resume(tmp_path):
    settings(tmp_path)
    torch.multiprocessing.spawn(worker, args=(str(tmp_path), str(tmp_path / 'rendezvous')), nprocs=2, join=True)


@pytest.mark.parametrize('candidate_backend', ['reference', 'tiled_fp64_v1'])
def test_retained_tokenwise_qualifier_checks_finite_adam_step(tmp_path, candidate_backend):
    import hashlib
    import json
    from pathlib import Path
    import subprocess
    import sys
    model, optimizer, adapter = make(tmp_path, actual_actor=True)
    try:
        _ordinary_step(model, optimizer, adapter, 2)
        checkpoint = tmp_path / 'retained.pt'
        torch.save(dict(model=model.state_dict(), controller=adapter.state_dict(), config=adapter.config, step=3,
            data_cursor=dict(file_names=[Path(p).name for p in adapter.replay.settings['train_files']])), checkpoint)
        replay = copy.deepcopy(adapter.replay.settings)
        replay['tokenwise_objective'] = asdict(TokenwiseConfig(replay['tolerance'], replay['tolerance'],
            1e-5, replay['rho'], ranking_weight=300., dual_update_mode='clipped_relative'))
        settings_file = tmp_path / 'settings.json'
        settings_file.write_text(json.dumps(replay))
        output = tmp_path / 'qualified.json'
        command = [sys.executable, 'runs/chisel-astra/qualify_replay_maintenance.py',
            '--checkpoint', str(checkpoint), '--expected-sha256', hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            '--settings', str(settings_file), '--output', str(output), '--device', 'cpu', '--finite-adam-check',
            '--compare-finite-outputs', '--finite-fidelity-metric', 'mutual-kl',
            '--candidate-token-loss-backend', candidate_backend]
        result = subprocess.run(command, text=True, capture_output=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
        report = json.loads(output.read_text())
        implementations = report['loss_implementation_diagnostic']
        assert implementations['candidate_loss_backend'] == candidate_backend
        assert implementations['oracle_loss_backend'] == 'reference'
        assert implementations['candidate_tokenwise_objective']['loss_backend'] == candidate_backend
        assert not implementations['production_backend_enabled']
        finite = report['finite_adam_check']
        assert finite['actor_and_optimizer_restored'] and finite['dual_updates'] == 0
        assert len(finite['steps']) == 2
        assert finite['steps'][0]['adam_steps'] == finite['steps'][1]['adam_steps']
        assert finite['adam_delta_relative_l2'] <= finite['adam_delta_relative_limit']
        assert report['qualification']['numerical_checks_within_limits']
        direct = finite['direct_output_comparison']
        assert finite['fidelity_metric'] == 'mutual-kl'
        assert direct['dense_to_thin_mean_kl'] <= finite['mutual_mean_kl_limit']
        assert direct['tokens'] == 8
        assert 0 <= direct['argmax_changes'] <= direct['top10_order_changes'] <= 8
        # Two scratch evaluations and their direct comparison add eight forwards.
        oracle = report['rank_records'][0]['oracle_extra_gpt_compute']
        assert oracle['student_forwards'] == 12 and oracle['student_backwards'] == 4
        assert report['metrics']['chisel_replay_updates'] == 1
    finally:
        adapter.close()


def test_candidate_loss_cannot_bypass_production_or_weaken_diagnostic(tmp_path):
    import runpy
    from types import SimpleNamespace
    script = runpy.run_path('runs/chisel-astra/qualify_replay_maintenance.py')
    configure = script['candidate_loss_configuration']
    with pytest.raises(ValueError, match='tokenwise objective'):
        configure(SimpleNamespace(tokenwise_config=None), 'tiled_fp64_v1')
    config = TokenwiseConfig(1e-10, 1e-10, 1e-8, .01)
    original = asdict(config)
    replay = SimpleNamespace(tokenwise_config=config, dual=SimpleNamespace(config=config))
    configure(replay, 'tiled_fp64_v1')
    assert asdict(replay.dual.config) == original  # No changed objective or dual schedule.
    with pytest.raises(ValueError, match='qualified reference backend'):
        s = settings(tmp_path)
        s['tokenwise_objective'] = asdict(TokenwiseConfig(s['tolerance'], s['tolerance'], 1e-5,
                                                        s['rho'], loss_backend='tiled_fp64_v1'))
        validate_replay_maintenance(s)
    import subprocess
    import sys
    # Rejected before any checkpoint/settings file is opened, including a
    # learning-curve request that could otherwise train the experimental path.
    for extra in ([], ['--mode', 'learning-curve', '--finite-adam-check',
                      '--compare-finite-outputs', '--finite-fidelity-metric', 'mutual-kl']):
        command = [sys.executable, 'runs/chisel-astra/qualify_replay_maintenance.py',
            '--checkpoint', 'unused.pt', '--expected-sha256', '0' * 64,
            '--settings', 'unused.json', '--output', str(tmp_path / 'unused.json'),
            '--device', 'cpu', '--candidate-token-loss-backend', 'tiled_fp64_v1'] + extra
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        assert result.returncode != 0
        assert 'candidate loss requires correctness mode' in result.stderr
