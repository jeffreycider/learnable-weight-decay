"""Teacher cache reader that trusts the verified FILE hash for tensor integrity (candidate B).

``TeacherCacheReader.read`` verifies a panel on first visit by (1) SHA256 of
the whole file against the sidecar/manifest and (2) a second SHA256 of each
deserialized tensor (inputs, features) against the hashes recorded at write
time. Step (2) re-reads and copies the full ~404 MB features tensor, i.e. a
second complete pass over the panel bytes for every first visit. The file hash
already pins every byte of the serialized tensors (write_panel hashed the
tensors, then saved, then hashed the file), so (2) can only catch a
deserialization defect, not on-disk or transport corruption.

This opt-in subclass keeps every other check identical (manifest provenance,
stamp, format/softcap, identity) and skips only the tensor re-hash WHEN the
file hash matched in this process. It is a byte-reduction candidate for the
first-visit IO regime; adopting it is an owner/root decision about accepting
the file hash as the integrity guarantee. Not wired into production.
"""
from __future__ import annotations
import json
import torch
from .teacher_cache import TeacherCacheReader, TeacherPanel, file_hash, FORMAT, SOFTCAP


class FileHashTeacherCacheReader(TeacherCacheReader):
    def read(self, state_id, role):
        if role not in ('train', 'audit'):
            raise ValueError('invalid panel role')
        path = self.directory / f'state_{state_id:04d}_{role}.pt'
        info = json.loads(path.with_suffix('.json').read_text())
        if self.manifest is not None:
            expected = self.expected.get((state_id, role))
            if expected is None or any(info[key] != expected[key] for key in ('sha256', 'state_sha256', 'input_sha256', 'feature_sha256', 'sequence_ids')):
                raise ValueError('panel provenance differs from complete cache manifest')
        stat = path.stat()
        stamp = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        checked = self.verify_once and self._verified.get((state_id, role)) == (stamp, info['sha256'])
        if not checked and file_hash(path) != info['sha256']:
            raise ValueError('teacher panel file hash mismatch')
        value = torch.load(path, map_location='cpu', weights_only=True, mmap=self.mmap)
        metadata = value['metadata']
        if metadata['format'] != FORMAT or metadata['softcap'] != SOFTCAP:
            raise ValueError('unsupported target execution')
        if metadata['state_id'] != state_id or metadata['role'] != role:
            raise ValueError('panel identity differs')
        # Candidate B: the recorded tensor hashes must still match the manifest/sidecar
        # (string comparison, no re-read); the file hash above covers the bytes.
        for key, name in [('input_sha256', 'input_sha256'), ('feature_sha256', 'feature_sha256')]:
            if metadata[name] != info[name]:
                raise ValueError('panel tensor hash metadata differs from sidecar')
        after = path.stat()
        if stamp != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError('teacher panel changed during verification')
        self._verified[(state_id, role)] = (stamp, info['sha256'])
        return TeacherPanel(value['inputs'], value['features'], tuple(metadata['sequence_ids']), metadata)
