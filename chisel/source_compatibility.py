"""Explicit, hash-to-hash compatibility declarations for already captured trajectories."""
import hashlib
from pathlib import Path
import yaml


def verify_sources(recorded, compatibility=None):
    allowed = yaml.safe_load(Path(compatibility).read_text()) if compatibility else {}
    for source, captured in recorded.items():
        current = hashlib.sha256(Path(source).read_bytes()).hexdigest()
        if current == captured:
            continue
        entries = allowed.get(source, [])
        if not any(item['captured'] == captured and item['current'] == current for item in entries):
            raise ValueError(f'captured base implementation differs: {source}')
