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


def test_restore_finishes_each_checkpoint_before_next_step(tmp_path,monkeypatch):
    source=tmp_path/'source'
    for rank in range(2):
        directory=source/f'rank{rank}';directory.mkdir(parents=True)
        for step in range(3):
            path=directory/f'step_{step:06d}.pt.zst';path.write_bytes(b'checkpoint')
            receipt=dict(file=path.name,bytes=path.stat().st_size,sha256=sha256(path))
            path.with_suffix('.ready.json').write_text(json.dumps(receipt))
    copied=[]
    original_copy=restore_recording.shutil.copyfile
    def copy(source,destination):
        copied.append((source.parent.name,source.name))
        return original_copy(source,destination)
    monkeypatch.setattr(restore_recording.shutil,'copyfile',copy)
    monkeypatch.setattr(restore_recording,'client_from_config',lambda config:object())
    restore_recording.restore(dict(source=str(source),destination=str(tmp_path/'cache'),workers=1,priority_steps=[2]))
    assert copied==[(f'rank{rank}',f'step_{step:06d}.pt.zst') for step in [2,0,1] for rank in range(2)]
