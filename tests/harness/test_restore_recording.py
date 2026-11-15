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
