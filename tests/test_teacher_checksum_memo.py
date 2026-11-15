import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from scripts import diagnose_cpu_norm_preservation as diagnostic


@pytest.fixture
def cache_fixture(tmp_path,monkeypatch):
    diagnostic._TEACHER_CHECKSUM_CACHE.clear()
    inputs=torch.tensor([[1,2,3,4]]);ids=[7]
    engine=SimpleNamespace(teacher_id='load-a',_local_cpu_binding=('state-a','load-a'),microbatch_sequences=1,
        kl_dtype=torch.float64,gpt_precision='fp32',device=torch.device('cpu'))
    file=tmp_path/'0000.pt'
    torch.save(dict(inputs=inputs,log_probs=torch.zeros(1,4,8,dtype=torch.float64),sequence_ids=ids),file)
    digest=hashlib.sha256(file.read_bytes()).hexdigest()
    meta=dict(state_sha256='state-a',input_sha256=diagnostic.tensor_sha(inputs),sequence_ids=ids,
        files=[dict(file=file.name,sha256=digest)],precision='CPU FP32 GPT with FP64 log_softmax',torch_version=torch.__version__)
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(meta))
    calls=[];original=diagnostic.sha
    def counted(path):calls.append(str(path));return original(path)
    monkeypatch.setattr(diagnostic,'sha',counted)
    def load(path=tmp_path):return diagnostic.DiskTeacherCache(path,engine,inputs,ids,'state-a')
    yield SimpleNamespace(root=tmp_path,file=file,manifest=manifest,meta=meta,engine=engine,inputs=inputs,ids=ids,calls=calls,load=load,original_sha=original)
    diagnostic._TEACHER_CHECKSUM_CACHE.clear()


def test_repeated_success_canonical_hit_keeps_only_bounded_metadata(cache_fixture,monkeypatch):
    f=cache_fixture;first=f.load();second=f.load(f.root/'.')
    assert len(f.calls)==1
    assert first.checksum_stats['files_hashed']==1 and first.checksum_stats['bytes_hashed']==f.file.stat().st_size
    assert second.checksum_stats['hits']==1 and second.checksum_stats['files_hashed']==0
    assert second.checksum_stats['hash_seconds']==0 and second.checksum_stats['verification_seconds']>=0
    assert torch.equal(second(0,1).inputs,f.inputs)
    assert all(value is None for value in diagnostic._TEACHER_CHECKSUM_CACHE.values())
    monkeypatch.setattr(diagnostic,'_TEACHER_CHECKSUM_CACHE_LIMIT',2)
    for i in range(3):
        file=f.root/f'extra{i}';file.write_bytes(bytes([i]))
        counts=dict(hits=0,files_hashed=0,bytes_hashed=0,hash_seconds=0.)
        diagnostic._verify_teacher_file(file,f.original_sha(file),counts)
    assert len(diagnostic._TEACHER_CHECKSUM_CACHE)==2


def test_changed_expected_digest_rehashes_and_failure_not_memoized(cache_fixture):
    f=cache_fixture;f.load();f.meta['files'][0]['sha256']='0'*64;f.manifest.write_text(json.dumps(f.meta))
    for _ in range(2):
        with pytest.raises(ValueError,match='file changed'):f.load()
    assert len(f.calls)==3 and not diagnostic._TEACHER_CHECKSUM_CACHE


def test_same_size_corruption_restored_mtime_invalidates_ctime(cache_fixture):
    f=cache_fixture;provider=f.load();before=f.file.stat();content=bytearray(f.file.read_bytes());content[-1]^=1
    f.file.write_bytes(content);os.utime(f.file,ns=(before.st_atime_ns,before.st_mtime_ns))
    after=f.file.stat();assert after.st_size==before.st_size and after.st_mtime_ns==before.st_mtime_ns
    assert after.st_ctime_ns!=before.st_ctime_ns
    with pytest.raises(ValueError,match='file changed'):f.load()
    with pytest.raises(ValueError,match='after verification'):provider(0,1)
    assert len(f.calls)==2


def test_replacement_inode_with_same_bytes_forces_successful_rehash(cache_fixture):
    f=cache_fixture;f.load();before=f.file.stat();replacement=f.root/'replacement.pt'
    replacement.write_bytes(f.file.read_bytes());os.utime(replacement,ns=(before.st_atime_ns,before.st_mtime_ns));replacement.replace(f.file)
    assert f.file.stat().st_ino!=before.st_ino
    rebuilt=f.load();assert rebuilt.checksum_stats['files_hashed']==1 and len(f.calls)==2
    assert f.load().checksum_stats['hits']==1


def test_state_input_and_runtime_validation_never_skipped_on_hit(cache_fixture):
    f=cache_fixture;f.load()
    with pytest.raises(ValueError,match='identity'):diagnostic.DiskTeacherCache(f.root,f.engine,f.inputs+1,f.ids,'state-a')
    f.engine.teacher_id='load-b';f.engine._local_cpu_binding=('state-b','load-b')
    with pytest.raises(ValueError,match='identity'):diagnostic.DiskTeacherCache(f.root,f.engine,f.inputs,f.ids,'state-b')
    f.engine.teacher_id='load-a';f.engine._local_cpu_binding=('state-a','load-a')
    f.meta['torch_version']='different-runtime';f.manifest.write_text(json.dumps(f.meta))
    with pytest.raises(ValueError,match='identity'):f.load()
    assert len(f.calls)==1


def test_mutation_while_hashing_is_rejected_not_cached(cache_fixture,monkeypatch):
    f=cache_fixture
    def unstable(path):
        result=f.original_sha(path);before=Path(path).stat()
        data=bytearray(Path(path).read_bytes());data[-1]^=1;Path(path).write_bytes(data)
        os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
        return result
    monkeypatch.setattr(diagnostic,'sha',unstable)
    with pytest.raises(ValueError,match='during checksum'):f.load()
    assert not diagnostic._TEACHER_CHECKSUM_CACHE


def test_producer_success_is_bounded_metadata_and_reader_never_rehashes(tmp_path,monkeypatch):
    diagnostic._TEACHER_CHECKSUM_CACHE.clear()
    monkeypatch.setattr(diagnostic,'_TEACHER_CHECKSUM_CACHE_LIMIT',2)
    calls=[];original=diagnostic.sha
    def counted(path):calls.append(str(path));return original(path)
    monkeypatch.setattr(diagnostic,'sha',counted)
    for i in range(3):
        path=tmp_path/f'payload{i}';path.write_bytes(bytes([i])*1024)
        digest=diagnostic._hash_created_teacher_file(path)
        counts=dict(hits=0,files_hashed=0,bytes_hashed=0,hash_seconds=0.)
        diagnostic._verify_teacher_file(path,digest,counts)
        assert counts['hits']==1 and counts['bytes_hashed']==counts['files_hashed']==0
    assert len(calls)==3 and len(diagnostic._TEACHER_CHECKSUM_CACHE)==2
    for key,value in diagnostic._TEACHER_CHECKSUM_CACHE.items():
        assert value is None and isinstance(key[0],str) and isinstance(key[1],str)
        assert len(key[2])==5 and all(isinstance(v,int) for v in key[2])
    diagnostic._TEACHER_CHECKSUM_CACHE.clear()


@pytest.mark.parametrize('failure',['exception','mutation'])
def test_producer_failed_or_racing_hash_never_memoized(tmp_path,monkeypatch,failure):
    diagnostic._TEACHER_CHECKSUM_CACHE.clear()
    path=tmp_path/'payload';path.write_bytes(b'original')
    diagnostic._hash_created_teacher_file(path)  # Even a previous success is invalidated.
    original=diagnostic.sha
    def failed(file):
        if failure=='exception':raise OSError('read failure')
        digest=original(file);stat=file.stat();file.write_bytes(b'changed!')
        os.utime(file,ns=(stat.st_atime_ns,stat.st_mtime_ns))
        return digest
    monkeypatch.setattr(diagnostic,'sha',failed)
    with pytest.raises(OSError if failure=='exception' else ValueError):diagnostic._hash_created_teacher_file(path)
    assert not diagnostic._TEACHER_CHECKSUM_CACHE
