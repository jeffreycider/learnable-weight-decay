"""Verify pinned FineWeb shard identities lazily, before a shard is consumed."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from chisel.storage import sha256


@lru_cache(maxsize=256)
def verify_shard(path, size, modified, changed, expected):
    if sha256(path) != expected:
        raise ValueError(f'FineWeb shard SHA256 mismatch: {path}')


def manifest(path):
    if path is None:
        return {}, None
    contents = Path(path).read_bytes()
    rows = json.loads(contents)
    identities = {row['path']: row['lfs']['oid'] for row in rows if row.get('type') == 'file' and 'lfs' in row}
    if not identities:
        raise ValueError('dataset manifest has no LFS SHA256 identities')
    return identities, hashlib.sha256(contents).hexdigest()
