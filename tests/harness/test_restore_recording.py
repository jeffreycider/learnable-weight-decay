import json
from pathlib import Path
from chisel.storage import sha256
from chisel import restore_recording


def test_missing_shard_restores_without_mutating_original(tmp_path,monkeypatch):
    source=tmp_path/'original';rank=source/'rank0';rank.mkdir(parents=True)
    actual=tmp_path/'remote';actual.write_bytes(b'checkpoint')
    receipt=dict(file='step_000000.pt.zst',sha256=sha256(actual),bytes=actual.stat().st_size)
    (rank/'step_000000.pt.ready.json').write_text(json.dumps(receipt))
    (rank/'step_000000.pt.uploaded.json').write_text(json.dumps(dict(bucket='b',key='k',sha256=receipt['sha256'])))
    class Client:
        def download_file(self,bucket,key,path):
            assert (bucket,key)==('b','k');Path(path).write_bytes(actual.read_bytes())
    monkeypatch.setattr(restore_recording,'client_from_config',lambda config:Client())
    destination=tmp_path/'cache';config=dict(source=str(source),destination=str(destination),workers=1)
    restore_recording.restore(config);restore_recording.restore(config)
    assert (destination/'rank0'/receipt['file']).read_bytes()==actual.read_bytes()
    assert not (rank/receipt['file']).exists()


def test_import_cache_survives_original_uploader_deletion(tmp_path,monkeypatch):
    source=tmp_path/'original';rank=source/'rank0';rank.mkdir(parents=True)
    original=rank/'step_000000.pt.zst';original.write_bytes(b'checkpoint')
    receipt=dict(file=original.name,sha256=sha256(original),bytes=original.stat().st_size)
    (rank/'step_000000.pt.ready.json').write_text(json.dumps(receipt))
    monkeypatch.setattr(restore_recording,'client_from_config',lambda config:object())
    destination=tmp_path/'cache'
    restore_recording.restore(dict(source=str(source),destination=str(destination),workers=1))
    original.unlink()
    assert (destination/'rank0'/receipt['file']).read_bytes()==b'checkpoint'
