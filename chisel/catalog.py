"""Build offline manifests only from complete, hash-identified checkpoint records."""
import argparse
import json
import os
from pathlib import Path
import torch
import yaml
from chisel.harness import resolve
from chisel.storage import sha256


def build_catalog(config):
    destination = Path(config['output']).resolve()
    states = []
    for dataset in config['datasets']:
        root = Path(dataset['root']).resolve()
        for receipt in sorted(root.glob(dataset['receipts'])):
            row = json.loads(receipt.read_text())
            path = receipt.parent / row['file']
            if path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
                raise ValueError(f'checkpoint receipt mismatch: {path}')
            record = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
            if not record['has_next_training_step']:
                continue
            for rank in range(record.get('activation_shards', 0)):
                shard = receipt.parent / f'activation_rank{rank:04d}.ready.json'
                if not shard.exists():
                    break
                info = json.loads(shard.read_text())
                if sha256(shard.parent / info['file']) != info['sha256']:
                    raise ValueError(f'activation receipt mismatch: {shard}')
            else:
                states.append(dict(path=os.path.relpath(path, destination.parent), sha256=row['sha256'],
                                   domain=dataset['domain'], step=record['step']))
    if len({entry['sha256'] for entry in states}) != len(states):
        raise ValueError('duplicate checkpoint identity')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(yaml.safe_dump(dict(states=states), sort_keys=False))
    temporary.replace(destination)
    return states


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    build_catalog(resolve(parser.parse_args().config))
