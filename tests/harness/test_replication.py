import json
import pytest
from chisel.replicate import copy_record
from chisel.storage import sha256


def test_record_is_published_after_copy_and_source_is_retained(tmp_path):
    source=tmp_path/'local';source.mkdir()
    value=source/'state.pt';value.write_bytes(b'weights and gradients')
    receipt=source/'state.ready.json'
    receipt.write_text(json.dumps(dict(file=value.name,bytes=value.stat().st_size,sha256=sha256(value))))
    destination=tmp_path/'shared'
    copy_record(source,destination,receipt)
    copied=json.loads((destination/receipt.name).read_text())
    assert copied['path']==str(destination/value.name)
    assert (destination/value.name).read_bytes()==value.read_bytes()
    copy_record(source,destination,receipt)
    assert value.exists()
    value.write_bytes(b'corrupt')
    receipt.write_text(json.dumps(dict(file=value.name,bytes=7,sha256=sha256(value))))
    with pytest.raises(ValueError,match='collision'):copy_record(source,destination,receipt)


def test_bad_source_cannot_publish_ready_marker(tmp_path):
    source=tmp_path/'local';source.mkdir()
    (source/'state.pt').write_bytes(b'bad')
    receipt=source/'state.ready.json'
    receipt.write_text(json.dumps(dict(file='state.pt',bytes=3,sha256='wrong')))
    destination=tmp_path/'shared'
    with pytest.raises(ValueError,match='differs'):copy_record(source,destination,receipt)
    assert not (destination/receipt.name).exists()


def test_mirror_finishes_only_after_records_and_restart_snapshots_are_copied(tmp_path):
    from chisel.replicate import mirror
    source=tmp_path/'local';source.mkdir()
    (source/'recipe.yaml').write_text('train_steps: 2\n')
    (source/'checkpoint_000002_rank0.pt').write_bytes(b'restart state')
    for step in range(2):
        folder=source/'trajectory'/str(step);folder.mkdir(parents=True)
        value=folder/'state.pt';value.write_bytes(bytes([step]))
        (folder/'state.ready.json').write_text(json.dumps(dict(file=value.name,bytes=1,sha256=sha256(value))))
    (source/'queue_complete.json').write_text('{"steps": 2}')
    destination=tmp_path/'shared'
    mirror(dict(source=str(source),destination=str(destination),workers=2,poll_seconds=.001))
    assert json.loads((destination/'queue_complete.json').read_text())==dict(steps=2)
    assert (destination/'recipe.yaml').read_bytes()==(source/'recipe.yaml').read_bytes()
    assert len(list(destination.rglob('*.ready.json')))==3
    status=json.loads((destination/'replication_status.json').read_text())
    assert status['copied']==status['local_ready']==3 and status['pending']==0


def test_relocation_preserves_original_paths_and_requires_shared_readback(tmp_path):
    from chisel.replicate import relocate_verified
    source=tmp_path/'local';source.mkdir()
    value=source/'state.pt';value.write_bytes(b'complete state')
    receipt=source/'state.ready.json'
    receipt.write_text(json.dumps(dict(file=value.name,bytes=value.stat().st_size,sha256=sha256(value))))
    destination=tmp_path/'shared'
    copy_record(source,destination,receipt,relocate=True)
    assert value.is_symlink() and value.read_bytes()==b'complete state'
    copy_record(source,destination,receipt,relocate=True)
    other=source/'other.pt';other.write_bytes(b'original')
    target=destination/'corrupt.pt';target.write_bytes(b'incorrect')
    with pytest.raises(ValueError,match='readback'):
        relocate_verified(other,target,sha256(other))
    assert other.read_bytes()==b'original' and not other.is_symlink()
