"""Incremental reader for an append-only metrics.jsonl.

``scripts/train_multistate_chisel.plot_progress`` re-reads and re-parses the
entire metrics file on rank 0 at every epoch end (``json.loads`` of 232 MB took
3.2 s on the development box at update 200, and the file grows ~1.1 MB per
update). Every other rank waits for rank 0 at the next collective, so that time
is critical path. This reader keeps the parsed rows and the byte offset of the
last complete line and parses only appended bytes; truncation or rewrite of the
file (shorter size, or changed inode) resets the cache. Rows are the same JSON
objects the full parse would produce; plotting code is unchanged.
"""
from __future__ import annotations
import json
import os
from pathlib import Path


class IncrementalMetrics:
    def __init__(self, path):
        self.path = Path(path)
        self.rows = []
        self.offset = 0
        self.inode = None

    def read(self):
        """Return the full list of rows parsed so far, parsing only new bytes."""
        stat = os.stat(self.path)
        if self.inode != stat.st_ino or stat.st_size < self.offset:
            self.rows, self.offset, self.inode = [], 0, stat.st_ino
        if stat.st_size == self.offset:
            return self.rows
        with self.path.open('rb') as stream:
            stream.seek(self.offset)
            data = stream.read()
        end = data.rfind(b'\n') + 1  # only complete lines; a partial tail is re-read next time
        for line in data[:end].splitlines():
            if line.strip():
                self.rows.append(json.loads(line))
        self.offset += end
        return self.rows
