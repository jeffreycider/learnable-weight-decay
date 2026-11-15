import os
os.environ.setdefault('PROFILE_CHISEL_SKIP_SOURCE_GUARD', '1')
import pytest
import torch
import rewriter.state_bank as state_bank
import rewriter.teacher_cache as teacher_cache
import rewriter.verify_prefetch as vp
from rewriter.verify_prefetch import VerifyAhead, verify_state, verify_panel


def _readers(tmp_path):
    from tests.test_step_timing import _tiny_bank_and_cache
    from rewriter.state_bank import StateBankReader
    from rewriter.teacher_cache import TeacherCacheReader
    bank, cache = _tiny_bank_and_cache(tmp_path)
    return StateBankReader(bank, verify_once=True, mmap_model=True), TeacherCacheReader(cache, verify_once=True, mmap=True), bank, cache


def test_verify_ahead_fills_stamps_so_reads_skip_hashing_and_results_match(tmp_path, monkeypatch):
    reader, cache, bank, cache_dir = _readers(tmp_path)
    from rewriter.state_bank import StateBankReader
    from rewriter.teacher_cache import TeacherCacheReader
    plain_reader, plain_cache = StateBankReader(bank, verify_once=True, mmap_model=True), TeacherCacheReader(cache_dir, verify_once=True, mmap=True)
    calls = dict(digest=0, file_hash=0, tensor_hash=0)
    real_digest, real_file_hash, real_tensor_hash = state_bank._digest, teacher_cache.file_hash, teacher_cache.tensor_hash
    def count(name, fn):
        def wrapped(*a, **k):
            calls[name] += 1
            return fn(*a, **k)
        return wrapped
    ahead = VerifyAhead(reader, cache, roles=('train',), depth=2)
    try:
        ahead.schedule([1, 2, 3])
        for s in (1, 2, 3):
            ahead.wait(s)
        assert ahead.hashed == dict(states=3, panels=3)
        # comparison readers hash on THEIR first visits; do that before counting
        plain = {s: (plain_reader.read(s, device='cpu', model_device='cpu'), plain_cache.read(s, 'train')) for s in (1, 2, 3)}
        # consumer reads through the pre-verified readers must now do NO hashing
        monkeypatch.setattr(state_bank, '_digest', count('digest', real_digest))
        monkeypatch.setattr(teacher_cache, 'file_hash', count('file_hash', real_file_hash))
        monkeypatch.setattr(teacher_cache, 'tensor_hash', count('tensor_hash', real_tensor_hash))
        for s in (1, 2, 3):
            record = reader.read(s, device='cpu', model_device='cpu')
            panel = cache.read(s, 'train')
            plain_record, plain_panel = plain[s]
            assert record.assert_verified() == plain_record.assert_verified()
            assert all(torch.equal(record.model[n], plain_record.model[n]) for n in plain_record.model)
            assert torch.equal(panel.features, plain_panel.features) and panel.sequence_ids == plain_panel.sequence_ids
        assert calls == dict(digest=0, file_hash=0, tensor_hash=0)
        # the plain readers hashed on their first visits (same functions, consumer thread)
    finally:
        ahead.close()
    assert verify_state(reader, 1) is False and verify_panel(cache, 1, 'train') is False  # already stamped


def test_verify_ahead_surfaces_corruption_and_rejects_bad_ids(tmp_path):
    reader, cache, bank, cache_dir = _readers(tmp_path)
    with (cache_dir / 'state_0002_train.pt').open('ab') as stream:
        stream.write(b'corrupt')
    ahead = VerifyAhead(reader, cache, depth=1)
    try:
        ahead.schedule([1, 2])
        ahead.wait(1)
        with pytest.raises(ValueError, match='hash'):
            ahead.wait(2)
        assert (2, 'train') not in cache._verified  # nothing stamped for the corrupted panel
        with pytest.raises(ValueError, match='hash'):
            cache.read(2, 'train')  # the ordinary reader rejects it too
    finally:
        ahead.close()
    with pytest.raises(IndexError):
        verify_state(reader, 99)
    with pytest.raises(ValueError, match='role'):
        verify_panel(cache, 1, 'bogus')
    with pytest.raises(ValueError):
        VerifyAhead(reader, cache, depth=0)
