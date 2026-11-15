"""YAML-configured bounded R2 upload workers; safe to restart without local deletion."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
from pathlib import Path
import signal
import time
from chisel.harness import resolve
from chisel.storage import atomic_json, client_from_config, upload_receipt


def run(config):
    root = Path(config['root']).resolve()
    client = client_from_config(config)
    lock = (root / 'uploader.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stopping, pending, failed = False, {}, {}

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        with ThreadPoolExecutor(config['workers']) as pool:
            for _ in range(config['polls']):
                for future, receipt in list(pending.items()):
                    if future.done():
                        del pending[future]
                        try:
                            future.result()
                            failed.pop(str(receipt), None)
                        except Exception as error:
                            failed[str(receipt)] = dict(error=str(error), retry_after=time.time() + config['retry_seconds'])
                receipts = sorted(root.glob(config['receipt_glob']))
                verified = 0
                for receipt in receipts:
                    marker = receipt.with_name(receipt.name.replace('.ready.json', '.uploaded.json'))
                    if marker.exists():
                        row, uploaded = json.loads(receipt.read_text()), json.loads(marker.read_text())
                        expected = config['prefix'].rstrip('/') + '/' + (receipt.parent / row['file']).relative_to(root).as_posix()
                        if (uploaded['bucket'], uploaded['key'], uploaded['sha256']) != (config['bucket'], expected, row['sha256']):
                            raise ValueError(f'upload marker destination or identity mismatch: {marker}')
                        verified += 1
                        continue
                    if stopping or receipt in pending.values() or len(pending) >= config['workers']:
                        continue
                    if failed.get(str(receipt), {}).get('retry_after', 0) > time.time():
                        continue
                    future = pool.submit(upload_receipt, client, root, receipt, config['bucket'], config['prefix'])
                    pending[future] = receipt
                atomic_json(root / 'upload_status.json', dict(ready=len(receipts), verified=verified,
                    pending=len(pending), failures=failed, updated_unix=time.time()))
                if stopping and not pending:
                    break
                time.sleep(config['poll_seconds'])
            for future in pending:
                future.result()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    run(resolve(parser.parse_args().config))
