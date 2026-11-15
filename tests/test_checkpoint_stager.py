import os
os.environ.setdefault('PROFILE_CHISEL_SKIP_SOURCE_GUARD', '1')
import pytest
import torch
from rewriter.checkpoint_stager import CheckpointStager, frozen_signature


def test_stager_binds_states_in_order_with_engine_and_store_equal_to_eager(tmp_path):
    from tests.test_step_timing import _tiny_bank_and_cache
    from rewriter.state_bank import StateBankReader
    from rewriter.teacher_cache import TeacherCacheReader
    from rewriter.function_preservation import FunctionalPreservation
    from rewriter.grouped_norm import GroupedNormObjective
    from rewriter.static_snapshot_store import _leaves
    from baseline.model import GPT
    bank, cache_dir = _tiny_bank_and_cache(tmp_path)
    reader = StateBankReader(bank, verify_once=True, mmap_model=True)
    cache = TeacherCacheReader(cache_dir, verify_once=True, mmap=True)
    cfg = reader.manifest['provenance']['model_config']
    engine = FunctionalPreservation(GPT(**cfg).requires_grad_(False), microbatch_sequences=1, kl_dtype=torch.float64)
    reference = FunctionalPreservation(GPT(**cfg).requires_grad_(False), microbatch_sequences=1, kl_dtype=torch.float64)
    eager_reader = StateBankReader(bank, verify_once=True, mmap_model=True)
    stager = CheckpointStager(reader, cache, engine, device='cpu', norm_class=GroupedNormObjective)
    states = [1, 2, 3, 1]
    stager.schedule(states)
    try:
        expected = None
        for state_id in states:
            record, panel, grouped, signature = stager.bind(state_id, expected_signature=expected)
            expected = expected or signature
            assert stager.bound and stager.bound_state == state_id
            eager = eager_reader.read(state_id, device='cpu', model_device='cpu')
            cache.load_engine(reference, eager)
            for name in reference.parameters:
                assert torch.equal(engine.parameters[name], reference.parameters[name]), name
            assert engine._teacher_cache_binding[1] == eager.assert_verified()
            static_snapshot, static_train = stager.static_views(record)
            for (name, held), (_, fresh) in zip(_leaves(static_snapshot, static_train), _leaves(eager.snapshot, eager.train)):
                assert torch.equal(held, fresh) and held.stride() == fresh.stride(), name
            assert torch.equal(panel.inputs, cache.read(state_id, 'train').inputs)
            assert set(stager.last_times) >= {'wait_for_worker', 'engine_load', 'grouped_setup', 'observation_refill', 'worker'}
        # a different frozen signature is rejected and leaves the store unbound
        stager.schedule([2])
        with pytest.raises(ValueError, match='signature'):
            stager.bind(2, expected_signature=(('made.up',), ('head.bias',)))
        assert not stager.bound
    finally:
        stager.close()


def test_stager_failure_after_engine_copy_leaves_it_unbound(tmp_path):
    from tests.test_step_timing import _tiny_bank_and_cache
    from rewriter.state_bank import StateBankReader
    from rewriter.teacher_cache import TeacherCacheReader
    from rewriter.function_preservation import FunctionalPreservation
    from rewriter.grouped_norm import GroupedNormObjective
    from baseline.model import GPT
    bank, cache_dir = _tiny_bank_and_cache(tmp_path)
    reader = StateBankReader(bank, verify_once=True, mmap_model=True)
    cache = TeacherCacheReader(cache_dir, verify_once=True, mmap=True)
    cfg = reader.manifest['provenance']['model_config']
    engine = FunctionalPreservation(GPT(**cfg).requires_grad_(False), microbatch_sequences=1, kl_dtype=torch.float64)
    calls = {'n': 0}
    class Exploding(GroupedNormObjective):
        def __init__(self, *a, **k):
            calls['n'] += 1
            if calls['n'] == 2:
                raise RuntimeError('injected after engine copy')
            super().__init__(*a, **k)
    stager = CheckpointStager(reader, cache, engine, device='cpu', norm_class=Exploding)
    stager.schedule([1, 2, 3])
    try:
        record, *_ = stager.bind(1)
        assert stager.bound
        with pytest.raises(RuntimeError, match='injected'):
            stager.bind(2)
        assert not stager.bound and stager.bound_state is None
        with pytest.raises(RuntimeError, match='not bound'):
            stager.static_views(record)
        record3, *_ = stager.bind(3)
        assert stager.bound and stager.bound_state == 3
    finally:
        stager.close()


def test_stager_with_constants_cache_matches_plain_path(tmp_path):
    from tests.test_step_timing import _tiny_bank_and_cache
    from rewriter.state_bank import StateBankReader
    from rewriter.teacher_cache import TeacherCacheReader
    from rewriter.offline_systems import CachedBaseNormPreservation, VerifiedStateConstants
    from rewriter.grouped_norm import GroupedNormObjective
    from baseline.model import GPT
    bank, cache_dir = _tiny_bank_and_cache(tmp_path)
    reader = StateBankReader(bank, verify_once=True, mmap_model=True)
    cache = TeacherCacheReader(cache_dir, verify_once=True, mmap=True)
    cfg = reader.manifest['provenance']['model_config']
    engine = CachedBaseNormPreservation(GPT(**cfg).requires_grad_(False), microbatch_sequences=1, kl_dtype=torch.float64)
    constants = VerifiedStateConstants(engine, GroupedNormObjective)
    stager = CheckpointStager(reader, cache, engine, device='cpu', constants=constants)
    stager.schedule([1, 2, 1])
    try:
        signatures = []
        for s in (1, 2, 1):
            record, panel, grouped, signature = stager.bind(s)
            signatures.append(signature)
            plain = GroupedNormObjective(engine.parameters, zero_reference_policy='freeze')
            for g in plain.reference_squared:
                assert torch.equal(grouped.reference_squared[g], plain.reference_squared[g])
        assert signatures[0] == signatures[2]
        assert constants.hits['groups'] >= 1
    finally:
        stager.close()


def test_stager_report_policy_binds_through_signature_changes(tmp_path):
    from tests.test_step_timing import _tiny_bank_and_cache
    from rewriter.state_bank import StateBankReader
    from rewriter.teacher_cache import TeacherCacheReader
    from rewriter.function_preservation import FunctionalPreservation
    from rewriter.grouped_norm import GroupedNormObjective
    from baseline.model import GPT
    bank, cache_dir = _tiny_bank_and_cache(tmp_path)
    reader = StateBankReader(bank, verify_once=True, mmap_model=True)
    cache = TeacherCacheReader(cache_dir, verify_once=True, mmap=True)
    cfg = reader.manifest['provenance']['model_config']
    engine = FunctionalPreservation(GPT(**cfg).requires_grad_(False), microbatch_sequences=1, kl_dtype=torch.float64)
    stager = CheckpointStager(reader, cache, engine, device='cpu', norm_class=GroupedNormObjective)
    stager.schedule([1, 2])
    try:
        record, panel, grouped, signature = stager.bind(1, expected_signature=(('other',), ('head.bias',)), signature_policy='report')
        assert stager.bound and stager.last_times['signature_changed'] is True
        with pytest.raises(ValueError, match='signature_policy'):
            stager.bind(2, signature_policy='bogus')
    finally:
        stager.close()
