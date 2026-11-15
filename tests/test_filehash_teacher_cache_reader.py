import os
os.environ.setdefault('PROFILE_CHISEL_SKIP_SOURCE_GUARD', '1')
import json
import pytest
import torch
import rewriter.teacher_cache as teacher_cache
from rewriter.filehash_teacher_cache_reader import FileHashTeacherCacheReader
from rewriter.teacher_cache import TeacherCacheReader


def _cache(tmp_path):
    from tests.test_step_timing import _tiny_bank_and_cache
    _, cache = _tiny_bank_and_cache(tmp_path)
    return cache


def test_filehash_reader_matches_reference_without_tensor_rehash(tmp_path, monkeypatch):
    cache = _cache(tmp_path)
    reference = TeacherCacheReader(cache, verify_once=True, mmap=True)
    expected = {(s, r): reference.read(s, r) for s in (1, 2) for r in ('train', 'audit')}
    calls = {'tensor_hash': 0, 'file_hash': 0}
    real_t, real_f = teacher_cache.tensor_hash, teacher_cache.file_hash
    monkeypatch.setattr(teacher_cache, 'tensor_hash', lambda *a, **k: calls.__setitem__('tensor_hash', calls['tensor_hash'] + 1) or real_t(*a, **k))
    import rewriter.filehash_teacher_cache_reader as module
    monkeypatch.setattr(module, 'file_hash', lambda *a, **k: calls.__setitem__('file_hash', calls['file_hash'] + 1) or real_f(*a, **k))
    reader = FileHashTeacherCacheReader(cache, verify_once=True, mmap=True)
    for (s, r), ref in expected.items():
        panel = reader.read(s, r)
        assert torch.equal(panel.inputs, ref.inputs) and torch.equal(panel.features, ref.features)
        assert panel.sequence_ids == ref.sequence_ids and panel.metadata == ref.metadata
        panel2 = reader.read(s, r)  # second visit: stamp hit, no hashing at all
        assert torch.equal(panel2.features, ref.features)
    assert calls['tensor_hash'] == 0 and calls['file_hash'] == 4  # one file hash per panel, never a tensor re-hash


def test_filehash_reader_still_rejects_corruption_and_sidecar_tampering(tmp_path):
    cache = _cache(tmp_path)
    reader = FileHashTeacherCacheReader(cache, verify_once=True, mmap=True)
    reader.read(1, 'train')
    with (cache / 'state_0002_train.pt').open('ab') as stream:
        stream.write(b'x')
    with pytest.raises(ValueError, match='hash mismatch'):
        reader.read(2, 'train')
    sidecar = cache / 'state_0003_train.json'
    info = json.loads(sidecar.read_text())
    info['feature_sha256'] = 'f' * 64
    sidecar.write_text(json.dumps(info))
    with pytest.raises(ValueError, match='provenance'):
        reader.read(3, 'train')
    with pytest.raises(ValueError, match='role'):
        reader.read(1, 'bogus')
