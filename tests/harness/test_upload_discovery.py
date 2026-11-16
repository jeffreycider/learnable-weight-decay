from pathlib import Path
import threading
import time
import pytest
from chisel import upload, upload_discovery


def test_upload_starts_before_slow_discovery_finishes(tmp_path, monkeypatch):
    uploaded = threading.Event()
    paths = [tmp_path / f'{i}.ready.json' for i in range(2)]
    for path in paths:
        path.write_text('{}')
    def discover(root):
        yield paths[0]
        assert uploaded.wait(3), 'upload waited for discovery to finish'
        yield paths[1]
    copied = []
    def copy(client, root, receipt, bucket, prefix):
        copied.append(receipt)
        uploaded.set()
    monkeypatch.setattr(upload_discovery, 'ready_receipts', discover)
    monkeypatch.setattr(upload, 'client_from_config', lambda config: object())
    monkeypatch.setattr(upload, 'upload_receipt', copy)
    upload.run(dict(root=str(tmp_path), workers=1, polls=40, poll_seconds=.01,
                    receipt_glob='**/*.ready.json', bucket='b', prefix='p', retry_seconds=0,
                    asynchronous_discovery=True, scan_interval_seconds=300))
    assert copied == paths


def test_full_queue_shutdown_and_discovery_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(upload_discovery, 'ready_receipts', lambda root: iter(Path(str(i)) for i in range(100)))
    inventory = upload_discovery.Inventory(tmp_path, '**/*.ready.json', 300, capacity=1)
    inventory.poll()
    inventory.close()  # Cancellation releases a producer blocked on its bounded queue.
    def fail(root):
        raise OSError('read failure')
    monkeypatch.setattr(upload_discovery, 'ready_receipts', fail)
    inventory = upload_discovery.Inventory(tmp_path, '**/*.ready.json', 300)
    try:
        inventory.poll()
        with pytest.raises(OSError, match='read failure'):
            inventory.future.result(timeout=3)
        with pytest.raises(OSError, match='read failure'):
            inventory.poll()
    finally:
        inventory.close()
