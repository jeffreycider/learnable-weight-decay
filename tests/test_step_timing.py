import os
os.environ.setdefault('PROFILE_CHISEL_SKIP_SOURCE_GUARD', '1')  # shared tree is edited concurrently by other agents
import json, struct
from pathlib import Path
import numpy as np
import pytest
import torch
from baseline.model import GPT
from rewriter.function_preservation import FunctionalPreservation
from rewriter.step_timing import EventPhaseTimer, install, synchronous_timed, untimed
import scripts.train_multistate_chisel as train
from rewriter.deferred_kl import mean_kl_and_backward_deferred


def test_event_timer_cpu_fallback_matches_sync_semantics_and_install_restores():
    device = torch.device('cpu')
    timings = {}
    timer = EventPhaseTimer()
    assert timer.timed(timings, 'a_seconds', lambda: 3, device) == 3
    assert timer.timed(None, 'a_seconds', lambda: 4, device) == 4
    assert timings['a_seconds'] >= 0 and timer.resolve() == 0
    sync = {}
    assert synchronous_timed(sync, 'x_seconds', lambda: 5, device) == 5 and 'x_seconds' in sync
    assert untimed({}, 'x', lambda: 6, device) == 6
    original = train.timed
    for mode in ('event', 'none', 'sync'):
        timer, restore = install(train, mode)
        assert train.timed is not original
        assert (timer is not None) == (mode == 'event')
        restore()
        assert train.timed is original
    with pytest.raises(ValueError):
        install(train, 'bogus')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA events')
def test_event_timer_accounts_device_time_once_per_resolve():
    device = torch.device('cuda')
    timer = EventPhaseTimer()
    timings = {}
    x = torch.randn(4096, 4096, device=device)
    timer.timed(timings, 'mm_seconds', lambda: x @ x, device)
    timer.timed(timings, 'mm_seconds', lambda: x @ x, device)
    assert 'mm_seconds' not in timings
    assert timer.resolve(device) == 2
    assert timings['mm_seconds'] > 0 and timings['mm_host_seconds'] >= 0


# Fixtures live in tests/chisel_fixtures.py (production imports only) so component tests of
# other modules can import them without this file's rewriter.step_timing dependency.
from tests.chisel_fixtures import tiny_engine_and_inputs as _tiny_engine_and_inputs  # noqa: E402
from tests.chisel_fixtures import tiny_bank_and_cache as _tiny_bank_and_cache  # noqa: E402


@pytest.mark.parametrize('use_dual', [False, True])
def test_deferred_readback_is_bit_identical_to_reference(use_dual):
    from rewriter.primal_dual import PrimalDual
    engine, inputs = _tiny_engine_and_inputs()
    direction = torch.linspace(-1., 1., 16)
    results = []
    for function in (train.mean_kl_and_backward, mean_kl_and_backward_deferred):
        theta = torch.tensor(.02, requires_grad=True)
        params = dict(engine.parameters)
        params['proj.bias'] = params['proj.bias'] + theta * direction
        dual = PrimalDual([7], tolerance=1e-6, rho=.01, dual_lr=.1) if use_dual else None
        if dual is not None:
            dual.multipliers[7] = .3
        timings = {}
        measured = function(engine, params, theta.square(), inputs, kl_budget=1e-5, state_weight=.25,
                            timings=timings, primal_dual=dual, state_id=7)
        results.append((measured, theta.grad.clone(), timings))
    (reference, reference_grad, reference_timings), (deferred, deferred_grad, deferred_timings) = results
    assert reference['mean_kl'] == deferred['mean_kl']
    assert reference['kl_penalty'] == deferred['kl_penalty']
    assert reference['objective'] == deferred['objective']
    if use_dual:
        assert reference['kl_coefficient'] == deferred['kl_coefficient']
    assert torch.equal(reference_grad, deferred_grad)
    assert set(reference_timings) == set(deferred_timings)


def test_deferred_readback_rejects_component_diagnostics_and_nonfinite():
    engine, inputs = _tiny_engine_and_inputs()
    theta = torch.tensor(.02, requires_grad=True)
    params = dict(engine.parameters)
    params['proj.bias'] = params['proj.bias'] + theta * torch.linspace(-1., 1., 16)
    with pytest.raises(ValueError, match='diagnostics'):
        mean_kl_and_backward_deferred(engine, params, theta.square(), inputs, kl_budget=1e-5, diagnostic_parameters=[])
    with pytest.raises(FloatingPointError, match='norm'):
        mean_kl_and_backward_deferred(engine, params, theta.square() * float('nan'), inputs, kl_budget=1e-5)


@pytest.mark.parametrize('arm', [dict(timing='sync'), dict(timing='event', deferred=True), dict(timing='none', deferred=True)])
def test_profiler_harness_replays_a_rank_share_on_cpu_and_arms_are_bit_identical(tmp_path, arm):
    from scripts.profile_chisel_update import main
    bank, cache = _tiny_bank_and_cache(tmp_path)
    common = ['--state-bank', str(bank), '--teacher-cache', str(cache), '--device', 'cpu', '--gpt-precision', 'baseline',
              '--state-ids', '1', '3', '--epoch', '0', '--sequences', '2', '--microbatch', '1', '--checkpoint-batch', '4',
              '--repeats', '1', '--chisel-model-dim', '128', '--chisel-layers', '1', '--chisel-patch-size', '8']
    reference = main([*common, '--output', str(tmp_path / 'reference'), '--timing', 'sync'])
    flags = ['--timing', arm['timing']] + (['--deferred-kl'] if arm.get('deferred') else [])
    other = main([*common, '--output', str(tmp_path / 'arm'), *flags])
    assert reference['gradient_sha256'] == other['gradient_sha256']
    assert other['all_measured_passes_bit_identical']
    assert other['states'] == [1, 3]
    for row in other['passes']:
        assert row['states'] == 2 and row['wall_seconds'] > 0
        if arm['timing'] == 'none':
            assert row['component_totals'] == {}
        else:
            assert row['component_totals']['gpt_kl_backward_seconds'] > 0
            assert row['component_totals']['state_load_seconds'] > 0
    assert (tmp_path / 'arm' / 'summary.json').exists()
    with pytest.raises(SystemExit):
        main([*common, '--output', str(tmp_path / 'arm'), *flags])


def test_block_selection_mirrors_trainer(tmp_path):
    from scripts.profile_chisel_update import select_training_blocks
    from rewriter.teacher_cache import TeacherPanel
    panel = TeacherPanel(torch.arange(128 * 2).reshape(128, 2), torch.zeros(128, 2, 3), tuple(range(128)), {'microbatch_sequences': 16})
    seen = set()
    for epoch in range(4):
        chosen = select_training_blocks(panel, 64, epoch, 400001, 17, train.stable_seed)
        assert len(chosen.inputs) == 64 and len(set(chosen.sequence_ids)) == 64
        seen |= set(chosen.sequence_ids)
        if epoch % 2 == 1:
            assert len(seen) == 128  # every two epochs cover the whole 128-sequence pool exactly
            seen = set()
    assert select_training_blocks(panel, 128, 0, 400001, 17, train.stable_seed) is panel


def test_staged_leaves_restore_exact_values_and_strides():
    from rewriter.state_prefetch import stage_leaf, restore_leaf
    from rewriter.prefix_cache import _restore_layout
    torch.manual_seed(0)
    cases = [torch.randn(5, 3), torch.randn(5, 3).T, torch.randn(4, 1).expand(4, 6), torch.randn(0, 3), torch.randn(7)]
    for tensor in cases:
        reference = _restore_layout(tensor, tensor.stride(), 'cpu')
        restored = restore_leaf(stage_leaf(tensor, pin=False), torch.device('cpu'), False)
        assert torch.equal(reference, restored) and reference.stride() == restored.stride()
        assert restored.storage_offset() == 0


def test_prefetcher_matches_synchronous_reader_bytes_and_layouts(tmp_path):
    from rewriter.state_prefetch import StatePrefetcher
    from rewriter.state_bank import StateBankReader
    from rewriter.teacher_cache import TeacherCacheReader
    from scripts.benchmark_observation_transfer import leaves
    bank, cache_dir = _tiny_bank_and_cache(tmp_path)
    reader = StateBankReader(bank, verify_once=True, mmap_model=True)
    cache = TeacherCacheReader(cache_dir, verify_once=True, mmap=True)
    direct = {s: (reader.read(s, device='cpu', model_device='cpu'), cache.read(s, 'train')) for s in (1, 2, 3)}
    prefetcher = StatePrefetcher(reader, cache, device='cpu')
    prefetcher.schedule([1, 2, 3])
    try:
        for state_id in (1, 2, 3):
            record, panel, worker = prefetcher.take(state_id)
            expected_record, expected_panel = direct[state_id]
            assert record.assert_verified() == expected_record.assert_verified()
            for name, value in expected_record.model.items():
                assert torch.equal(record.model[name], value)
            for left, right in zip(leaves(record.snapshot), leaves(expected_record.snapshot), strict=True):
                assert torch.equal(left, right) and left.stride() == right.stride()
            for left, right in zip(leaves(record.train), leaves(expected_record.train), strict=True):
                assert torch.equal(left, right) and left.stride() == right.stride()
            assert torch.equal(panel.inputs, expected_panel.inputs) and torch.equal(panel.features, expected_panel.features)
            assert panel.sequence_ids == expected_panel.sequence_ids and panel.metadata == expected_panel.metadata
            assert worker['bank_read_seconds'] >= 0 and 'panel_stage_seconds' in worker
        with pytest.raises(RuntimeError, match='order'):
            prefetcher.schedule([2]); prefetcher.take(3)
    finally:
        prefetcher.close()


def test_prefetcher_surfaces_worker_errors(tmp_path):
    from rewriter.state_prefetch import StatePrefetcher
    from rewriter.state_bank import StateBankReader
    bank, cache_dir = _tiny_bank_and_cache(tmp_path)
    reader = StateBankReader(bank, verify_once=True, mmap_model=True)
    prefetcher = StatePrefetcher(reader, None, device='cpu')
    prefetcher.schedule([99])
    with pytest.raises(IndexError):
        prefetcher.take(99)
    prefetcher.close()


def test_profiler_harness_prefetch_arm_is_bit_identical(tmp_path):
    from scripts.profile_chisel_update import main
    bank, cache = _tiny_bank_and_cache(tmp_path)
    common = ['--state-bank', str(bank), '--teacher-cache', str(cache), '--device', 'cpu', '--gpt-precision', 'baseline',
              '--state-ids', '1', '2', '3', '--epoch', '1', '--sequences', '2', '--microbatch', '1', '--checkpoint-batch', '4',
              '--repeats', '1', '--chisel-model-dim', '128', '--chisel-layers', '1', '--chisel-patch-size', '8']
    reference = main([*common, '--output', str(tmp_path / 'reference'), '--timing', 'sync'])
    prefetch = main([*common, '--output', str(tmp_path / 'prefetch'), '--timing', 'event', '--deferred-kl', '--prefetch'])
    assert reference['gradient_sha256'] == prefetch['gradient_sha256']
    assert prefetch['all_measured_passes_bit_identical'] and prefetch['arm']['prefetch']
    for row in prefetch['passes']:
        assert row['prefetch_wait_seconds'] >= 0
        assert 'prefetch_worker_bank_read_seconds' in row['component_totals']
        assert 'teacher_panel_load_seconds' not in row['component_totals']


def test_incremental_metrics_matches_full_parse_and_handles_partial_lines_and_rewrites(tmp_path):
    from rewriter.progress_metrics import IncrementalMetrics
    path = tmp_path / 'metrics.jsonl'
    path.write_text(json.dumps(dict(event='start', update=0)) + '\n')
    reader = IncrementalMetrics(path)
    assert reader.read() == [dict(event='start', update=0)]
    with path.open('a') as stream:
        stream.write(json.dumps(dict(event='update', update=1)) + '\n')
        stream.write('{"event": "partial"')
    rows = reader.read()
    assert [r['event'] for r in rows] == ['start', 'update']
    with path.open('a') as stream:
        stream.write(', "update": 2}\n')
    assert [r.get('update') for r in reader.read()] == [0, 1, 2]
    assert reader.read() is rows and len(rows) == 3
    full = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows == full
    path.write_text(json.dumps(dict(event='start', update=0)) + '\n')  # truncation/rewrite resets
    assert reader.read() == [dict(event='start', update=0)]


def test_profiler_harness_two_rank_gloo_arms_match_at_equal_world_size(tmp_path):
    """Full 4-checkpoint batch on 2 gloo ranks: the optimized arm must reproduce the
    baseline arm's post-step parameters/Adam/dual bit for bit. (A single process
    sums the four contributions in a different order, so cross-world-size
    equality is not expected, exactly as in the 8-rank production update.)"""
    import subprocess, sys, os
    bank, cache = _tiny_bank_and_cache(tmp_path)
    common = ['--state-bank', str(bank), '--teacher-cache', str(cache), '--device', 'cpu', '--gpt-precision', 'baseline',
              '--state-ids', '1', '2', '3', '4', '--epoch', '0', '--sequences', '2', '--microbatch', '1', '--checkpoint-batch', '4',
              '--repeats', '1', '--chisel-model-dim', '128', '--chisel-layers', '1', '--chisel-patch-size', '8']
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    script = str(Path(__file__).resolve().parents[1] / 'scripts' / 'profile_chisel_update.py')
    results = {}
    for name, flags in (('baseline', ['--timing', 'sync']), ('optimized', ['--timing', 'event', '--deferred-kl', '--prefetch'])):
        subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2', script,
                        *common, '--output', str(tmp_path / name), *flags], check=True, env=env, timeout=600, capture_output=True)
        results[name] = json.loads((tmp_path / name / 'summary.json').read_text())
    baseline, optimized = results['baseline'], results['optimized']
    assert baseline['world_size'] == 2 and baseline['batch_states'] == [1, 2, 3, 4] and baseline['states'] == [1, 3]
    assert (tmp_path / 'optimized' / 'summary_rank1.json').exists()
    for summary in (baseline, optimized):
        assert summary['all_passes_same_update_all_ranks']
    for key in ('gradient_sha256', 'parameters_after_step_sha256', 'optimizer_state_sha256', 'dual_state_sha256'):
        assert optimized[key] == baseline[key], key
    assert optimized['passes'][0]['local_gradient_sha256'] != optimized['gradient_sha256']  # rank-local partial differs from the SUM
    assert optimized['passes'][0]['wall_seconds_max_over_ranks'] >= optimized['passes'][0]['wall_seconds']


def test_profiler_harness_constants_cache_arm_is_bit_identical_across_two_epochs(tmp_path):
    """Astra's SHA-keyed scalar caches imported into the harness: the second and
    third passes revisit every state through the cache and must reproduce bits."""
    from scripts.profile_chisel_update import main
    bank, cache = _tiny_bank_and_cache(tmp_path)
    common = ['--state-bank', str(bank), '--teacher-cache', str(cache), '--device', 'cpu', '--gpt-precision', 'baseline',
              '--state-ids', '1', '2', '3', '4', '--epoch', '0', '--sequences', '2', '--microbatch', '1', '--checkpoint-batch', '4',
              '--repeats', '2', '--chisel-model-dim', '128', '--chisel-layers', '1', '--chisel-patch-size', '8']
    reference = main([*common, '--output', str(tmp_path / 'reference'), '--timing', 'sync'])
    combined = main([*common, '--output', str(tmp_path / 'combined'), '--timing', 'event', '--deferred-kl', '--prefetch', '--constants-cache'])
    assert len(combined['passes']) == 3
    for key in ('gradient_sha256', 'parameters_after_step_sha256', 'optimizer_state_sha256', 'dual_state_sha256'):
        assert combined[key] == reference[key], key
    assert combined['arm']['constants_cache'] and combined['all_passes_same_update_all_ranks']


def test_trainer_plot_progress_uses_incremental_rows(tmp_path):
    pytest.importorskip('matplotlib')
    from scripts.train_multistate_chisel import plot_progress, _PLOT_METRICS
    output = tmp_path / 'run'; output.mkdir()
    def row(update, role='train', event='update'):
        return dict(event=event, role=role, update=update, objective=1., mean_kl=1e-7, kl_penalty=0., mean_group_l2_percentage_reduction=.1,
                    group_l2_percentage_reduction={'embedding.weight': .1}, completed_epochs=update)
    (output / 'metrics.jsonl').write_text('\n'.join(json.dumps(row(u)) for u in range(3)) + '\n')
    plot_progress(output)
    assert len(_PLOT_METRICS[output].rows) == 3 and (output / 'learning_curves.png').exists()
    with (output / 'metrics.jsonl').open('a') as stream:
        stream.write(json.dumps(row(3, 'heldout_states', 'audit_summary')) + '\n')
    plot_progress(output)
    assert len(_PLOT_METRICS[output].rows) == 4 and (output / 'group_curves.png').exists()


def test_profiler_harness_panel_only_prefetch_is_bit_identical(tmp_path):
    from scripts.profile_chisel_update import main
    bank, cache = _tiny_bank_and_cache(tmp_path)
    common = ['--state-bank', str(bank), '--teacher-cache', str(cache), '--device', 'cpu', '--gpt-precision', 'baseline',
              '--state-ids', '1', '2', '3', '--epoch', '1', '--sequences', '2', '--microbatch', '1', '--checkpoint-batch', '4',
              '--repeats', '1', '--chisel-model-dim', '128', '--chisel-layers', '1', '--chisel-patch-size', '8']
    reference = main([*common, '--output', str(tmp_path / 'reference'), '--timing', 'sync'])
    panel_only = main([*common, '--output', str(tmp_path / 'panel_only'), '--timing', 'sync', '--prefetch-panel-only'])
    for key in ('gradient_sha256', 'parameters_after_step_sha256', 'optimizer_state_sha256', 'dual_state_sha256'):
        assert panel_only[key] == reference[key], key
    totals = panel_only['passes'][0]['component_totals']
    assert 'prefetch_worker_panel_stage_seconds' in totals and 'prefetch_worker_bank_read_seconds' not in totals
    assert totals['state_load_seconds'] > 0 and panel_only['arm']['prefetch_panel_only']


def test_summarizer_labels_scopes_and_ratios(tmp_path, capsys):
    from scripts.profile_chisel_update import main
    from scripts.summarize_systems_benchmarks import main as summarize
    bank, cache = _tiny_bank_and_cache(tmp_path)
    common = ['--state-bank', str(bank), '--teacher-cache', str(cache), '--device', 'cpu', '--gpt-precision', 'baseline',
              '--state-ids', '1', '2', '--epoch', '0', '--sequences', '2', '--microbatch', '1', '--checkpoint-batch', '4',
              '--repeats', '1', '--chisel-model-dim', '128', '--chisel-layers', '1', '--chisel-patch-size', '8']
    main([*common, '--output', str(tmp_path / 'a'), '--timing', 'sync'])
    main([*common, '--output', str(tmp_path / 'b'), '--timing', 'none', '--deferred-kl'])
    entries = summarize(['--profile', str(tmp_path / 'a' / 'summary.json'), str(tmp_path / 'b' / 'summary.json'), '--json', str(tmp_path / 'table.json')])
    out = capsys.readouterr().out
    assert 'LOCAL 2-state contribution' in out and 'ratio' in out
    assert entries[1]['same_as_baseline_arm'] is True and (tmp_path / 'table.json').exists()


def test_coalesced_reader_matches_reference_reader_and_harness_arm(tmp_path):
    from rewriter.coalesced_state_reader import CoalescedStateBankReader
    from rewriter.state_bank import StateBankReader
    from scripts.benchmark_observation_transfer import leaves
    from scripts.profile_chisel_update import main
    bank, cache = _tiny_bank_and_cache(tmp_path)
    reference = StateBankReader(bank, verify_once=True, mmap_model=True)
    coalesced = CoalescedStateBankReader(bank, verify_once=True, mmap_model=True)
    for state_id in (1, 2):
        a, b = reference.read(state_id, device='cpu', model_device='cpu'), coalesced.read(state_id, device='cpu', model_device='cpu')
        assert a.assert_verified() == b.assert_verified()
        for left, right in zip(leaves(a.snapshot) + leaves(a.train), leaves(b.snapshot) + leaves(b.train), strict=True):
            assert torch.equal(left, right) and left.stride() == right.stride()
    with pytest.raises(IndexError):
        coalesced.read(99)
    common = ['--state-bank', str(bank), '--teacher-cache', str(cache), '--device', 'cpu', '--gpt-precision', 'baseline',
              '--state-ids', '1', '2', '--epoch', '0', '--sequences', '2', '--microbatch', '1', '--checkpoint-batch', '4',
              '--repeats', '1', '--chisel-model-dim', '128', '--chisel-layers', '1', '--chisel-patch-size', '8']
    ref = main([*common, '--output', str(tmp_path / 'ref'), '--timing', 'sync'])
    arm = main([*common, '--output', str(tmp_path / 'arm'), '--timing', 'sync', '--coalesced-observations'])
    assert arm['gradient_sha256'] == ref['gradient_sha256'] and arm['arm']['coalesced_observations']


def test_harness_shrinkage_coefficient_zero_removes_norm_term_and_changes_gradient(tmp_path):
    from scripts.profile_chisel_update import main
    bank, cache = _tiny_bank_and_cache(tmp_path)
    common = ['--state-bank', str(bank), '--teacher-cache', str(cache), '--device', 'cpu', '--gpt-precision', 'baseline',
              '--state-ids', '1', '2', '--epoch', '0', '--sequences', '2', '--microbatch', '1', '--checkpoint-batch', '4',
              '--repeats', '1', '--chisel-model-dim', '128', '--chisel-layers', '1', '--chisel-patch-size', '8']
    full = main([*common, '--output', str(tmp_path / 'full'), '--timing', 'sync'])
    zero = main([*common, '--output', str(tmp_path / 'zero'), '--timing', 'sync', '--shrinkage-coefficient', '0'])
    assert full['arm']['shrinkage_coefficient'] == 1. and zero['arm']['shrinkage_coefficient'] == 0.
    rows = [json.loads(l) for l in (tmp_path / 'zero' / 'states_rank0.jsonl').read_text().splitlines()]
    assert all(r['norm_term'] == 0. for r in rows)
    assert zero['gradient_sha256'] != full['gradient_sha256']
    # constants-cache path must accept the coefficient too
    zero_cached = main([*common, '--output', str(tmp_path / 'zero_cached'), '--timing', 'sync', '--shrinkage-coefficient', '0', '--constants-cache'])
    assert zero_cached['gradient_sha256'] == zero['gradient_sha256']
