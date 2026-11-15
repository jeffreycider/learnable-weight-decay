"""Copy completed local records to shared storage without blocking GPT training."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import os
import time
import yaml
from chisel.storage import atomic_json, sha256


def relocate_verified(original, target, digest):
    """Keep the original pathname usable while reclaiming its verified local copy."""
    if original.is_symlink():
        if original.resolve() != target.resolve():
            raise ValueError(f'unexpected staging symlink: {original}')
        return
    if sha256(target) != digest:
        raise ValueError(f'shared readback checksum mismatch: {target}')
    temporary = original.with_suffix(original.suffix + '.relocating')
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(target.resolve())
    temporary.replace(original)


def copy_record(source, destination, receipt, relocate=False):
    source, destination, receipt = Path(source), Path(destination), Path(receipt)
    info = json.loads(receipt.read_text())
    original = receipt.parent / info['file']
    target = destination / original.relative_to(source)
    ready = destination / receipt.relative_to(source)
    if ready.exists():
        previous = json.loads(ready.read_text())
        if previous['sha256'] != info['sha256'] or target.stat().st_size != info['bytes']:
            raise ValueError(f'immutable shared-record collision: {target}')
        if relocate:
            relocate_verified(original, target, info['sha256'])
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + '.copying')
    digest, count = hashlib.sha256(), 0
    with original.open('rb') as incoming, temporary.open('wb') as outgoing:
        for block in iter(lambda: incoming.read(8 << 20), b''):
            digest.update(block)
            count += len(block)
            outgoing.write(block)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    if digest.hexdigest() != info['sha256'] or count != info['bytes']:
        temporary.unlink()
        raise ValueError(f'local record differs from its receipt: {original}')
    temporary.replace(target)
    atomic_json(ready, info | dict(path=str(target.resolve()),
                                  copy_verification='source_sha256_during_copy'))
    if relocate:
        relocate_verified(original, target, info['sha256'])


def copy_metadata(source, destination, seen):
    for path in source.iterdir():
        if not path.is_file() or path.suffix not in ('.yaml', '.json', '.jsonl', '.gz'):
            continue
        if path.name.endswith(('.ready.json', '.uploaded.json')) or path.name == 'queue_complete.json':
            continue
        info = path.stat()
        stamp = (info.st_size, info.st_mtime_ns)
        if seen.get(path.name) == stamp:
            continue
        target = destination / path.name
        temporary = target.with_suffix(target.suffix + '.copying')
        temporary.write_bytes(path.read_bytes())
        temporary.replace(target)
        seen[path.name] = stamp


def mirror(config):
    source, destination = Path(config['source']), Path(config['destination'])
    if source.resolve() == destination.resolve():
        raise ValueError('staging and shared directories must differ')
    destination.mkdir(parents=True, exist_ok=True)
    completed, pending, metadata = set(), {}, {}
    with ThreadPoolExecutor(config['workers']) as pool:
        while True:
            copy_metadata(source, destination, metadata)
            # Restart snapshots are immutable once their atomic rename completes.
            for path in source.glob('checkpoint_*_rank*.pt'):
                receipt = path.with_suffix('.ready.json')
                if not receipt.exists():
                    atomic_json(receipt, dict(file=path.name, path=str(path.resolve()),
                        bytes=path.stat().st_size, sha256=sha256(path)))
            receipts = sorted(source.rglob('*.ready.json'))
            for receipt in receipts:
                if receipt in completed or receipt in pending:
                    continue
                if len(pending) >= 2 * config['workers']:
                    break
                try:
                    json.loads(receipt.read_text())
                except json.JSONDecodeError:
                    continue  # The producer may still be publishing its receipt.
                pending[receipt] = pool.submit(copy_record, source, destination, receipt,
                                              config.get('relocate_after_copy', False))
            for receipt, future in list(pending.items()):
                if future.done():
                    future.result()
                    completed.add(receipt)
                    del pending[receipt]
            atomic_json(destination / 'replication_status.json', dict(
                local_ready=len(receipts), copied=len(completed), pending=len(pending), updated_unix=time.time()))
            if (source / 'queue_complete.json').exists() and len(completed) == len(receipts) and not pending:
                copy_metadata(source, destination, metadata)
                atomic_json(destination / 'queue_complete.json', json.loads((source / 'queue_complete.json').read_text()))
                return
            if config.get('once'):
                for future in pending.values():
                    future.result()
                return
            time.sleep(config.get('poll_seconds', 1))


if __name__ == '__main__':
    import sys
    mirror(yaml.safe_load(Path(sys.argv[1]).read_text()))
