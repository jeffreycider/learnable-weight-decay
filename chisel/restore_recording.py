"""Restore archived trajectory shards to a separate import cache, with full checksums."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import os
import shutil
import yaml
from chisel.storage import client_from_config, sha256


def restore(config):
    source, destination = Path(config['source']), Path(config['destination'])
    client = client_from_config(config)

    def one(receipt):
        info = json.loads(receipt.read_text())
        original = receipt.parent / info['file']
        target = destination / original.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            target.unlink()
        if not target.exists():
            temporary = target.with_suffix(target.suffix + '.restore')
            try:
                # The old uploader may unlink its source after backup. Own the bytes.
                shutil.copyfile(original, temporary)
            except FileNotFoundError:
                marker = receipt.with_name(receipt.name.replace('.ready.json', '.uploaded.json'))
                archived = json.loads(marker.read_text())
                if archived['sha256'] != info['sha256']:
                    raise ValueError('archive receipt differs from original')
                client.download_file(archived['bucket'], archived['key'], str(temporary))
            if temporary.stat().st_size != info['bytes'] or sha256(temporary) != info['sha256']:
                raise ValueError(f'restored checkpoint checksum mismatch: {target}')
            os.replace(temporary, target)
        copied_receipt = destination / receipt.relative_to(source)
        if not copied_receipt.exists():
            copied_receipt.write_bytes(receipt.read_bytes())
        return str(target)

    priority = {step: index for index, step in enumerate(config.get('priority_steps', []))}
    def order(path):
        step = int(path.name.split('_')[1].split('.')[0])
        # Make complete checkpoints available throughout restoration, rather than
        # restoring every shard of rank 0 before starting rank 1.
        return priority.get(step, len(priority)), step, str(path.parent)
    receipts = sorted(source.glob('rank*/step_*.pt.ready.json'),
                      key=order)
    with ThreadPoolExecutor(config['workers']) as pool:
        for index, path in enumerate(pool.map(one, receipts)):
            if index % 100 == 0:
                print(json.dumps(dict(restored_or_copied=index + 1, path=path)), flush=True)


if __name__ == '__main__':
    import sys
    restore(yaml.safe_load(Path(sys.argv[1]).read_text()))
