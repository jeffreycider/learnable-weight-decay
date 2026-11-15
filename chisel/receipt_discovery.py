"""Enumerate ready records without statting checkpoint symlink targets."""
import os
from pathlib import Path


def ready_receipts(root):
    pending = [Path(root)]
    while pending:
        folder = pending.pop()
        with os.scandir(folder) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                elif entry.name.endswith('.ready.json') and entry.is_file(follow_symlinks=False):
                    yield Path(entry.path)
