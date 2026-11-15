"""Verify only the shards a run will consume against pinned HF LFS SHA256s."""
from pathlib import Path
import glob
import hashlib
import json
import struct


def data_manifest(config, verify=True):
    expected = {item['path']: item for item in json.loads((Path(__file__).parent/'vendor/dataset_files.json').read_text())}
    records = []
    for kind, target_batches, batch in [('train', config['max_steps'], config['batch_tokens']), ('val', 1, config['val_tokens'])]:
        files = sorted(glob.glob(config[f'{kind}_data']))
        if not files:
            raise FileNotFoundError(config[f'{kind}_data'])
        remaining = target_batches
        for filename in files:
            path = Path(filename)
            with path.open('rb') as f:
                magic, version, tokens = struct.unpack('<iii', f.read(12))
            if (magic, version) != (20240520, 1) or path.stat().st_size != 1024+2*tokens:
                raise ValueError(f'bad shard header/size: {path}')
            # PR323 advances if pos + batch + 1 >= len(tokens).
            usable_batches = (tokens-2)//batch
            if usable_batches < 1:
                raise ValueError(f'shard too small for batch: {path}')
            row = dict(kind=kind, path=str(path), tokens=tokens, bytes=path.stat().st_size,
                       consumed_batches=min(remaining, usable_batches))
            if verify:
                digest = hashlib.sha256()
                with path.open('rb') as f:
                    for chunk in iter(lambda: f.read(8*1024*1024), b''):
                        digest.update(chunk)
                row['sha256'] = digest.hexdigest()
                if path.name not in expected or row['sha256'] != expected[path.name]['lfs']['oid']:
                    raise ValueError(f'dataset does not match pinned FineWeb revision: {path}')
            records.append(row)
            remaining -= usable_batches
            if remaining <= 0:
                break
        if remaining > 0:
            raise ValueError(f'not enough {kind} data for run: missing {remaining} batches')
    return records
