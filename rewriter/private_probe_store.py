"""Single-writer transactional CPU/disk storage for checkpoint-private Adam.

Immutable private blobs become committed ONLY through a global manifest linking
all private versions to shared parameters/Adam and sampler/RNG state. Unreferenced
files after an interruption are harmless; there is deliberately no automatic GC.
The caller owns one device slot and the distributed schedule. This store performs
no collectives, keeps no all-state tensor cache, and does not preserve optimizer
state tensor addresses for graph-captured Adam.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import uuid
from collections.abc import Mapping
import torch

SCHEMA = 'chisel_private_probe_store_v1'


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device='cpu', copy=True)
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu(v) for v in value)
    return copy.deepcopy(value)


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def _layout(private_state):
    extra = copy.deepcopy(private_state['_extra_state'])
    extra.pop('state_key')
    return hashlib.sha256(_json(extra)).hexdigest()


@dataclass(frozen=True)
class PrivateIdentity:
    state_key: tuple
    state_sha256: str
    layout_sha256: str

    def __post_init__(self):
        if (not isinstance(self.state_key, tuple) or not self.state_key
                or any(type(v) not in (str, int) for v in self.state_key)):
            raise ValueError('explicit full checkpoint namespace required')
        for value in (self.state_sha256, self.layout_sha256):
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('lowercase SHA256 identity required')

    @classmethod
    def from_private(cls, private, *, state_sha256):
        # The caller obtains this digest from its verified immutable bank reader.
        return cls(private.state_key, state_sha256, _layout(private.state_dict()))

    def as_dict(self):
        return dict(state_key=list(self.state_key), state_sha256=self.state_sha256,
                    layout_sha256=self.layout_sha256)

    @property
    def token(self):
        return hashlib.sha256(_json(self.as_dict())).hexdigest()


def _optimizer_layout(slot, optimizer):
    if type(optimizer) is not torch.optim.Adam:
        raise TypeError('private store currently supports exact torch.optim.Adam only')
    names = {id(p): n for n, p in slot.private.named_parameters()}
    groups = []
    seen = []
    for group in optimizer.param_groups:
        ids = [id(p) for p in group['params']]
        if any(i not in names for i in ids):
            raise ValueError('private optimizer owns foreign parameters')
        groups.append([names[i] for i in ids]); seen.extend(ids)
    if len(set(seen)) != len(seen) or set(seen) != set(names):
        raise ValueError('private optimizer must own each private parameter exactly once')
    return groups


def _validate_payload(identity, payload):
    if payload.get('schema') != SCHEMA or payload.get('identity') != identity.as_dict():
        raise ValueError('private payload checkpoint identity mismatch')
    state = payload['private']
    if (state['_extra_state']['state_key'] != identity.state_key
            or _layout(state) != identity.layout_sha256):
        raise ValueError('private checkpoint key/layout mismatch')
    visits = payload['visits']
    if type(visits) is not int or visits < 0:
        raise ValueError('nonnegative private visit counter required')
    groups = payload['optimizer']['param_groups']; names = payload['optimizer_names']
    if len(groups) != len(names): raise ValueError('Adam group layout differs')
    ids = {}; seen_names = []
    for group, group_names in zip(groups, names):
        if len(group['params']) != len(group_names): raise ValueError('Adam parameter layout differs')
        for key, name in zip(group['params'], group_names):
            if key in ids or name in seen_names or name not in state:
                raise ValueError('Adam parameter identity differs')
            ids[key] = name; seen_names.append(name)
    expected = {k for k, v in state.items() if isinstance(v, torch.Tensor)}
    if set(seen_names) != expected:
        raise ValueError('Adam does not cover private tensor layout')
    for name in expected:
        if state[name].device.type != 'cpu' or not bool(torch.isfinite(state[name]).all()):
            raise ValueError('private payload must contain finite CPU tensors')
    for key, moments in payload['optimizer']['state'].items():
        if key not in ids: raise ValueError('unknown Adam parameter state')
        parameter = state[ids[key]]
        if not {'step', 'exp_avg', 'exp_avg_sq'} <= moments.keys():
            raise ValueError('incomplete Adam moments')
        step = float(moments['step'])
        if not 0 <= step <= visits or not step.is_integer():
            raise ValueError('Adam counter exceeds private visits')
        for field, value in moments.items():
            if isinstance(value, torch.Tensor):
                if value.device.type != 'cpu' or not bool(torch.isfinite(value).all()):
                    raise ValueError('Adam state must contain finite CPU tensors')
                if field != 'step' and (value.shape != parameter.shape or value.dtype != parameter.dtype):
                    raise ValueError('Adam moment shape/dtype differs')


class PrivateProbeStore:
    """One writer; independent read-only instances may stage committed versions.

    `snapshot_slot` writes an uncommitted version. `commit` publishes the whole
    shared/private transaction. Restart with `restore` and default `stage`; never
    infer committed state from directory listings. An explicit `version` is only
    for a caller revisiting its own pending version inside the current batch.
    """
    def __init__(self, directory, *, read_only=False, snapshot_backend='reference'):
        if snapshot_backend not in ('reference', 'owned_cpu_v1', 'pinned_d2h_v1'):
            raise ValueError('unknown private snapshot backend')
        self.snapshot_backend = snapshot_backend
        self.directory = Path(directory); self.read_only = read_only; self._lock = None
        self._bound_manifest = None
        self._bound_reference = None
        self._pending = {}
        self._async_snapshot_ticket = None
        if not read_only:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._lock = (self.directory / 'writer.lock').open('a+b')
            try: fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException:
                self._lock.close(); self._lock = None
                raise RuntimeError('private store already has a writer')
        elif not self.directory.is_dir(): raise FileNotFoundError(self.directory)

    def close(self):
        try:
            if self._async_snapshot_ticket is not None:
                self._async_snapshot_ticket.close()
        finally:
            if self._lock is not None:
                fcntl.flock(self._lock, fcntl.LOCK_UN); self._lock.close(); self._lock = None

    def __enter__(self): return self
    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except BaseException as cleanup_error:
            if exc is None:
                raise
            add_note = getattr(exc, "add_note", None)
            if callable(add_note):
                add_note(f'Private store cleanup also failed: {cleanup_error!r}')
            else:
                if cleanup_error.__context__ is exc:
                    cleanup_error.__context__ = None
                exc.__context__ = cleanup_error

    def _writable(self):
        if self.read_only or self._lock is None: raise RuntimeError('store is not an active writer')

    def _path(self, reference):
        name = reference['file']
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError('artifact path must be a local basename')
        return self.directory / name

    def _verify(self, reference):
        path = self._path(reference)
        if path.stat().st_size != reference['bytes'] or _sha(path) != reference['sha256']:
            raise ValueError('private store artifact hash mismatch')
        return path

    def _write(self, value, prefix, *, json_file=False, owned_cpu=False):
        self._writable()
        if owned_cpu:
            def check(item):
                if isinstance(item, torch.Tensor):
                    if item.device.type != 'cpu': raise ValueError('owned snapshot must be entirely CPU resident')
                elif isinstance(item, dict):
                    for child in item.values(): check(child)
                elif isinstance(item, (tuple, list)):
                    for child in item: check(child)
            check(value)
        path = self.directory / f'{prefix}-{uuid.uuid4().hex}.{"json" if json_file else "pt"}'
        with path.open('xb') as stream:
            if json_file: stream.write(_json(value))
            else: torch.save(value if owned_cpu else _cpu(value), stream)
            stream.flush(); os.fsync(stream.fileno())
        _sync_dir(self.directory)
        return dict(file=path.name, bytes=path.stat().st_size, sha256=_sha(path))

    def _manifest_and_reference(self, reference=None):
        pointer = self.directory / 'CURRENT.json'
        if reference is None and not pointer.exists(): return None, None
        ref = reference if reference is not None else json.loads(pointer.read_text())
        result = json.loads(self._verify(ref).read_text())
        if result.get('schema') != SCHEMA or result['generation'] != ref['generation']:
            raise ValueError('global manifest schema/generation differs')
        return result, copy.deepcopy(ref)

    def manifest(self):
        return self._manifest_and_reference()[0]

    def snapshot_slot(self, identity, slot, optimizer, *, visits):
        if self._async_snapshot_ticket is not None:
            raise RuntimeError('finish asynchronous private snapshot before next snapshot')
        payload = self._prepare_slot_snapshot(identity, slot, optimizer, visits=visits)
        reference = self._write_slot_snapshot(identity, payload)
        self._pending[identity.token] = reference
        return reference

    def _prepare_slot_snapshot(self, identity, slot, optimizer, *, visits):
        """Own and validate CPU values before any optional background I/O."""
        self._writable(); slot.require_bound(identity.state_key)
        if self.snapshot_backend == 'pinned_d2h_v1':
            from .owned_cpu_snapshot import owned_cpu_snapshot
            state = owned_cpu_snapshot(dict(private=slot.private.state_dict(), optimizer=optimizer.state_dict()))
        else:
            state = dict(private=slot.export_state(), optimizer=_cpu(optimizer.state_dict()))
        payload = dict(schema=SCHEMA, identity=identity.as_dict(), **state,
                       optimizer_names=_optimizer_layout(slot, optimizer), visits=visits)
        _validate_payload(identity, payload)
        return payload

    def _write_slot_snapshot(self, identity, payload):
        """Persist a prepared CPU payload; publication still belongs to commit."""
        reference = dict(self._write(payload, 'private', owned_cpu=self.snapshot_backend != 'reference'), identity=identity.as_dict(),
                         token=identity.token, visits=payload['visits'])
        return reference

    def read_private(self, identity, *, version=None):
        ticket = self._async_snapshot_ticket
        if ticket is not None and ticket.identity_token == identity.token:
            raise RuntimeError('finish asynchronous private snapshot before reading the same identity')
        if version is None:
            manifest = self._bound_manifest or self.manifest()
            if manifest is None or identity.token not in manifest['private_versions']:
                raise KeyError('checkpoint has no committed private version')
            version = manifest['private_versions'][identity.token]
        if version.get('identity') != identity.as_dict() or version.get('token') != identity.token:
            raise ValueError('requested checkpoint differs from private version identity')
        payload = torch.load(self._verify(version), map_location='cpu', weights_only=False)
        _validate_payload(identity, payload)
        if payload['visits'] != version['visits']: raise ValueError('private visit metadata differs')
        return payload

    def stage(self, identity, slot, optimizer, *, version=None):
        if version is None and self._bound_manifest is None:
            raise RuntimeError('restore the global checkpoint before staging committed private state')
        payload = self.read_private(identity, version=version)
        if _optimizer_layout(slot, optimizer) != payload['optimizer_names']:
            raise ValueError('staging Adam parameter ordering differs')
        # All payload/hash/Adam validation precedes device writes. A device failure
        # aborts the caller's run; it must restore the last global commit.
        slot.stage(payload['private'], state_key=identity.state_key)
        optimizer.load_state_dict(payload['optimizer'])
        return payload['visits']

    def commit(self, *, shared_state, sampler_state, private_updates, expected_generation):
        """Commit only at a completed global optimizer boundary, never mid-batch.

        Caller supplies complete shared model/Adam/RNG and sampler/schedule state.
        Only changed private references are passed; older references are retained.
        Optimizer values are not inspected or altered by this storage layer.
        """
        if self._async_snapshot_ticket is not None:
            raise RuntimeError('finish asynchronous private snapshot before commit')
        self._writable(); previous, previous_ref = self._manifest_and_reference()
        generation = previous['generation'] if previous else 0
        if expected_generation != generation: raise ValueError('stale global transaction generation')
        if generation and (self._bound_manifest is None or self._bound_manifest['generation'] != generation):
            raise ValueError('restore CURRENT before committing a new global transaction')
        if not isinstance(shared_state, Mapping) or not isinstance(sampler_state, Mapping):
            raise TypeError('explicit shared and sampler checkpoint mappings required')
        if not {'model', 'optimizer', 'rng'} <= shared_state.keys():
            raise ValueError('shared checkpoint requires model, optimizer and RNG state')
        cursor = sampler_state.get('next_global_batch')
        if type(cursor) is not int or cursor < 0:
            raise ValueError('explicit nonnegative next_global_batch sampler cursor required')
        versions = copy.deepcopy(previous['private_versions']) if previous else {}
        namespaces = {tuple(v['identity']['state_key']): v['token'] for v in versions.values()}
        for version in private_updates:
            identity = PrivateIdentity(tuple(version['identity']['state_key']),
                version['identity']['state_sha256'], version['identity']['layout_sha256'])
            self.read_private(identity, version=version)  # validates changed blobs only
            if namespaces.get(identity.state_key, identity.token) != identity.token:
                raise ValueError('existing checkpoint namespace cannot change source or layout identity')
            namespaces[identity.state_key] = identity.token
            old = versions.get(identity.token)
            if old is not None and version['visits'] < old['visits']:
                raise ValueError('private visit counter cannot move backwards')
            versions[identity.token] = copy.deepcopy(version)
        global_ref = self._write(dict(schema=SCHEMA, generation=generation+1,
                                      shared=shared_state, sampler=sampler_state), 'shared')
        manifest = dict(schema=SCHEMA, generation=generation+1, global_state=global_ref,
                        private_versions=versions, previous_manifest=previous_ref)
        ref = self._write(manifest, 'manifest', json_file=True)
        ref['generation'] = generation+1
        temporary = self.directory / f'.CURRENT-{uuid.uuid4().hex}'
        with temporary.open('xb') as stream:
            stream.write(_json(ref)); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, self.directory / 'CURRENT.json')
        _sync_dir(self.directory)
        self._bound_manifest = manifest
        self._bound_reference = copy.deepcopy(ref)
        for token, version in list(self._pending.items()):
            if versions.get(token, {}).get('file') == version['file']:
                del self._pending[token]
        return generation+1

    def restore(self, *, manifest_ref=None):
        if self._async_snapshot_ticket is not None:
            raise RuntimeError('finish asynchronous private snapshot before restore')
        manifest, reference = self._manifest_and_reference(manifest_ref)
        if manifest is None: raise FileNotFoundError('no committed global checkpoint')
        state = torch.load(self._verify(manifest['global_state']), map_location='cpu', weights_only=False)
        if state.get('schema') != SCHEMA or state['generation'] != manifest['generation']:
            raise ValueError('shared/private transaction generation differs')
        # Pin this reader's private versions to the SAME shared generation even
        # if another writer later publishes CURRENT before the next stage call.
        self._bound_manifest = manifest
        self._bound_reference = reference
        return dict(state, private_versions=copy.deepcopy(manifest['private_versions']),
                    manifest_ref=copy.deepcopy(reference))

    def _replace_json(self, filename, value):
        temporary = self.directory / f'.{filename}-{uuid.uuid4().hex}'
        with temporary.open('xb') as stream:
            stream.write(_json(value)); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, self.directory / filename)
        _sync_dir(self.directory)

    def _pins(self):
        path = self.directory / 'PINS.json'
        if not path.exists(): return dict(schema=SCHEMA, pins={})
        value = json.loads(path.read_text())
        if value.get('schema') != SCHEMA or not isinstance(value.get('pins'), dict):
            raise ValueError('invalid private-store retention registry')
        return value

    def retain(self, name, *, manifest_ref=None):
        """Pin before handing a generation to a reader; no automatic TTL expiry.

        The writer coordinates reader lifetimes. A read-only process restores
        the returned explicit manifest_ref, and the writer releases its pin only
        after that reader is closed. Named research checkpoints use the same API.
        """
        self._writable()
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
            raise ValueError('safe explicit retention name required')
        manifest, reference = self._manifest_and_reference(manifest_ref)
        if manifest is None: raise FileNotFoundError('no global checkpoint to retain')
        pins = self._pins()
        if name in pins['pins'] and pins['pins'][name] != reference:
            raise ValueError('retention name already pins a different generation')
        pins['pins'][name] = reference
        self._replace_json('PINS.json', pins)
        return copy.deepcopy(reference)

    def release_pin(self, name):
        """Explicit release only; this call deletes no model/optimizer artifact."""
        self._writable(); pins = self._pins()
        if name not in pins['pins']: raise KeyError('unknown retention pin')
        del pins['pins'][name]
        self._replace_json('PINS.json', pins)

    @staticmethod
    def _file_stamp(path):
        value = path.lstat()
        if not stat.S_ISREG(value.st_mode): raise ValueError('store artifact must be a regular file')
        return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]

    def gc_plan(self, *, at_global_barrier=False):
        """Dry-run reachability over STORE-OWNED files only; never deletes files.

        Protect current + previous global transaction, named/live-reader pins,
        and latest pending private versions. Call only when no state writer or
        unregistered reader is active; the store has no distributed rendezvous.
        """
        self._writable()
        if self._async_snapshot_ticket is not None:
            raise RuntimeError('finish asynchronous private snapshot before collection')
        if at_global_barrier is not True: raise RuntimeError('explicit completed global barrier required')
        current, current_ref = self._manifest_and_reference()
        if current is None: raise RuntimeError('restore or commit a global checkpoint before collection')
        if current['generation'] > 1 and 'previous_manifest' not in current:
            raise ValueError('legacy manifest lacks explicit previous-transaction retention')
        pins = self._pins(); protected = set()

        def protect(reference):
            path = self._path(reference)
            stamp = self._file_stamp(path)
            if stamp[2] != reference['bytes']: raise ValueError('protected artifact size differs')
            protected.add(path.name)

        def root(reference):
            manifest, _ = self._manifest_and_reference(reference)
            protect(reference); protect(manifest['global_state'])
            for version in manifest['private_versions'].values(): protect(version)
            # Do not recursively keep every historical previous_manifest link.

        root(current_ref)
        if current.get('previous_manifest') is not None: root(current['previous_manifest'])
        if self._bound_reference is not None: root(self._bound_reference)
        for reference in pins['pins'].values(): root(reference)
        for reference in self._pending.values(): protect(reference)
        candidates = []
        for path in sorted(self.directory.iterdir()):
            if (re.fullmatch(r'(private|shared)-[0-9a-f]{32}\.pt|manifest-[0-9a-f]{32}\.json', path.name)
                    and path.name not in protected):
                candidates.append(dict(file=path.name, stamp=self._file_stamp(path)))
        plan = dict(schema=SCHEMA+'_gc_plan', current=current_ref,
                    writer_bound_reference=self._bound_reference,
                    pins_sha256=hashlib.sha256(_json(pins)).hexdigest(),
                    pending_sha256=hashlib.sha256(_json(self._pending)).hexdigest(),
                    protected_files=sorted(protected), candidates=candidates,
                    reclaimable_bytes=sum(row['stamp'][2] for row in candidates))
        plan['plan_sha256'] = hashlib.sha256(_json(plan)).hexdigest()
        return plan

    def apply_gc(self, plan, *, at_global_barrier=False):
        """Explicit reviewed-plan application; NEVER called by commit or training.

        Revalidate head/pins/pending/files under the existing writer lock. An
        interruption can leave some unreachable files, but never removes a root.
        Tests exercise this on temporary stores only; live deployment is separate.
        """
        actual = self.gc_plan(at_global_barrier=at_global_barrier)
        if _json(plan) != _json(actual): raise ValueError('stale or altered retention plan')
        deleted = []
        for row in actual['candidates']:
            path = self.directory / row['file']
            if self._file_stamp(path) != row['stamp']: raise ValueError('artifact changed during collection')
            path.unlink(); deleted.append(row['file'])
        _sync_dir(self.directory)
        return dict(plan_sha256=actual['plan_sha256'], deleted=deleted,
                    reclaimed_bytes=actual['reclaimable_bytes'])
