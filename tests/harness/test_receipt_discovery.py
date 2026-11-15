import json
from pathlib import Path
from chisel.receipt_discovery import ready_receipts
from chisel.storage import sha256


def test_discovery_skips_symlink_trees_and_broken_checkpoint_links(tmp_path):
    root=tmp_path/'records';root.mkdir()
    nested=root/'trajectory'/'000001';nested.mkdir(parents=True)
    receipt=nested/'state.ready.json';receipt.write_text('{}')
    (nested/'state.pt').symlink_to(tmp_path/'missing.pt')
    outside=tmp_path/'elsewhere';outside.mkdir()
    (outside/'hidden.ready.json').write_text('{}')
    (root/'linked').symlink_to(outside,target_is_directory=True)
    (root/'receipt.ready.json').symlink_to(outside/'hidden.ready.json')
    assert list(ready_receipts(root))==[receipt]


def test_uploader_does_not_probe_every_marker_when_workers_full(tmp_path,monkeypatch):
    import threading
    from chisel import upload
    release=threading.Event()
    probed=[]
    real_exists=Path.exists
    def exists(path):
        if path.name.endswith('.uploaded.json'): probed.append(path)
        return real_exists(path)
    for i in range(20):
        path=tmp_path/f'{i:02}.ready.json';path.write_text('{}')
    monkeypatch.setattr(Path,'exists',exists)
    monkeypatch.setattr(upload,'client_from_config',lambda c:object())
    monkeypatch.setattr(upload,'upload_receipt',lambda *a:release.wait(timeout=3))
    monkeypatch.setattr(upload.time,'sleep',lambda duration:release.set())
    upload.run(dict(root=str(tmp_path),workers=1,polls=1,poll_seconds=0,receipt_glob='**/*.ready.json',
                    bucket='b',prefix='p',retry_seconds=0))
    assert len(probed)==1


def test_existing_upload_marker_is_validated_before_counting(tmp_path,monkeypatch):
    import pytest
    from chisel import upload
    (tmp_path/'state.ready.json').write_text(json.dumps(dict(file='state.pt',sha256='a')))
    (tmp_path/'state.uploaded.json').write_text(json.dumps(dict(bucket='b',key='wrong',sha256='a')))
    monkeypatch.setattr(upload,'client_from_config',lambda c:object())
    with pytest.raises(ValueError,match='marker destination'):
        upload.run(dict(root=str(tmp_path),workers=1,polls=1,poll_seconds=0,receipt_glob='**/*.ready.json',
                    bucket='b',prefix='p',retry_seconds=0))
