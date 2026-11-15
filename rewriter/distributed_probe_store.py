"""Collective single-coordinator commits for identity-owned private probe shards.

CPU Gloo is the qualification target. There is one global/CURRENT.json; rank-local
private stores contain immutable blobs/indexes but never publish CURRENT. All
ranks participate in commit/restore with a bounded process-group timeout. A dead
rank can abort an operation, never publish a partially assembled global commit.
All ranks require the same coherent POSIX filesystem view; independently mounted
remote-volume replicas need an additional publication protocol and are untested.
Explicit collective retention/GC follows global manifest pins. It is never
automatic and must run at a completed optimizer barrier with registered readers.
Automatic resharding is not implemented.
"""
from __future__ import annotations
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping
import torch
import torch.distributed as dist
from .private_probe_store import PrivateIdentity, PrivateProbeStore, _cpu, _json, _sync_dir

SCHEMA = 'chisel_distributed_private_v1'


def private_owner(identity: PrivateIdentity, world_size: int):
    if type(world_size) is not int or world_size < 1:
        raise ValueError('positive fixed world size required')
    return int(identity.token, 16) % world_size


def _gather(value):
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def _raise_collective_errors(error):
    errors = _gather(error)
    if any(errors): raise RuntimeError(f'distributed private transaction aborted: {errors}')


def _digest(value):
    """Value-based exact shared-state digest, independent of tensor addresses."""
    h = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            h.update(_json(['tensor', str(tensor.dtype), list(tensor.shape)]))
            h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, Mapping):
            h.update(b'dict')
            for key in sorted(item, key=lambda k:(type(k).__name__, repr(k))):
                visit(key); visit(item[key])
        elif isinstance(item, (list, tuple)):
            h.update(_json([type(item).__name__, len(item)]))
            for entry in item: visit(entry)
        elif type(item) in (str, int, float, bool, type(None)):
            h.update(_json([type(item).__name__, item]))
        else:
            raise TypeError(f'unsupported shared-state digest value: {type(item)}')
    visit(value)
    return h.hexdigest()


def sum_shared_gradients(parameters):
    """SUM already globally weighted shared gradients; do not call on private ψ.

    Preserve globally-unused grad=None so Adam clocks do not advance spuriously.
    Packed buffers reduce collective count. Layout is checked across ranks; loss
    weighting/clipping and the single shared Adam step remain caller-owned.
    """
    parameters = tuple(parameters)
    if not parameters: raise ValueError('shared parameters required')
    signature = [(tuple(p.shape), str(p.dtype), p.device.type) for p in parameters]
    signatures = _gather(signature)
    if any(s != signature for s in signatures): raise ValueError('shared gradient layout differs across ranks')
    groups = defaultdict(list)
    for i, p in enumerate(parameters): groups[(p.device, p.dtype)].append((i, p))
    for entries in groups.values():
        present = torch.tensor([p.grad is not None for _, p in entries], device=entries[0][1].device, dtype=torch.int32)
        dist.all_reduce(present, op=dist.ReduceOp.MAX)
        buffer = torch.cat([(p.grad.detach() if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                            for _, p in entries])
        dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
        offset = 0
        for flag, (_, p) in zip(present.tolist(), entries):
            count = p.numel()
            p.grad = buffer[offset:offset+count].view_as(p).clone() if flag else None
            offset += count


class DistributedProbeStore:
    """One stable private slot/Adam per rank; immutable state keyed by checkpoint.

    Public calls snapshot_slot/stage are local. commit and restore are collective.
    The caller dispatches checkpoint visits to owner(identity); ownership does not
    depend on the state's position in a shuffled minibatch. Multiple visits to an
    ID on its owner are allowed and retain its exact private Adam counter.
    """
    def __init__(self, directory, *, run_identity, snapshot_backend='reference'):
        if not dist.is_initialized(): raise RuntimeError('initialized process group required')
        if snapshot_backend not in ('reference','pinned_d2h_v1'):
            raise ValueError('unsupported distributed private snapshot backend')
        if not isinstance(run_identity, str) or not run_identity:
            raise ValueError('explicit immutable training-run identity required')
        from pathlib import Path
        self.directory = Path(directory); self.rank = dist.get_rank(); self.world_size = dist.get_world_size()
        self.run_identity = run_identity; self.generation = 0; self._versions = {}; self._index_ref = None
        self._global_manifest_ref = None
        self.global_store = None; self.shard = None
        result = [None]
        if self.rank == 0:
            try: self.global_store = PrivateProbeStore(self.directory/'global')
            except Exception as exc: result[0] = repr(exc)
        dist.broadcast_object_list(result, src=0)
        if result[0] is not None: raise RuntimeError(result[0])
        error = None
        try:
            if self.rank != 0: self.global_store = PrivateProbeStore(self.directory/'global', read_only=True)
            self.shard = PrivateProbeStore(self.directory/'private'/f'rank{self.rank:04d}',snapshot_backend=snapshot_backend)
        except Exception as exc: error = repr(exc)
        try: _raise_collective_errors(error)
        except BaseException:
            self.close(); raise

    def close(self):
        try:
            if self.shard is not None: self.shard.close()
        finally:
            if self.global_store is not None: self.global_store.close()

    def __enter__(self): return self
    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except BaseException as cleanup_error:
            if exc is None:
                raise
            add_note = getattr(exc, "add_note", None)
            if callable(add_note):
                add_note(f'Distributed store cleanup also failed: {cleanup_error!r}')
            else:
                if cleanup_error.__context__ is exc:
                    cleanup_error.__context__ = None
                exc.__context__ = cleanup_error

    def owner(self, identity): return private_owner(identity, self.world_size)

    def _require_owner(self, identity):
        if self.owner(identity) != self.rank:
            raise ValueError('private checkpoint belongs to a different identity owner')

    def snapshot_slot(self, identity, slot, optimizer, *, visits):
        self._require_owner(identity)
        return self.shard.snapshot_slot(identity, slot, optimizer, visits=visits)

    def stage(self, identity, slot, optimizer, *, version=None):
        self._require_owner(identity)
        if version is None:
            if identity.token not in self._versions: raise KeyError('checkpoint has no globally committed private state')
            version = self._versions[identity.token]
        return self.shard.stage(identity, slot, optimizer, version=version)

    def _prepare_index(self, updates):
        versions = dict(self._versions)
        namespaces = {tuple(v['identity']['state_key']): v['token'] for v in versions.values()}
        for version in updates:
            identity = PrivateIdentity(tuple(version['identity']['state_key']), version['identity']['state_sha256'], version['identity']['layout_sha256'])
            self._require_owner(identity)
            self.shard.read_private(identity, version=version)
            if namespaces.get(identity.state_key, identity.token) != identity.token:
                raise ValueError('checkpoint namespace cannot change source/layout')
            namespaces[identity.state_key] = identity.token
            if version['visits'] < versions.get(identity.token, {}).get('visits', 0):
                raise ValueError('private visit counter moved backwards')
            versions[identity.token] = version
        index = dict(schema=SCHEMA, run_identity=self.run_identity, rank=self.rank,
                     world_size=self.world_size, generation=self.generation+1, private_versions=versions)
        return self.shard._write(index, 'index'), versions

    def _validate_published_indices(self, rows):
        namespaces = set()
        for expected_rank, row in enumerate(rows):
            if row['rank'] != expected_rank: raise ValueError('rank publication order differs')
            with PrivateProbeStore(self.directory/'private'/f'rank{expected_rank:04d}', read_only=True) as shard:
                index = torch.load(shard._verify(row['index']), map_location='cpu', weights_only=False)
            if (index.get('schema') != SCHEMA or index['generation'] != self.generation+1
                    or index['rank'] != expected_rank or index['world_size'] != self.world_size
                    or index['run_identity'] != self.run_identity):
                raise ValueError('rank publication belongs to a different transaction')
            for token, version in index['private_versions'].items():
                identity = PrivateIdentity(tuple(version['identity']['state_key']), version['identity']['state_sha256'], version['identity']['layout_sha256'])
                if token != identity.token or version['token'] != token or self.owner(identity) != expected_rank:
                    raise ValueError('private publication identity/owner differs')
                if identity.state_key in namespaces:
                    raise ValueError('checkpoint namespace appears more than once across shards')
                namespaces.add(identity.state_key)

    def commit(self, *, shared_state, rng_state, sampler_state, private_updates):
        """All ranks enter AFTER one completed shared Adam update (or initial save).

        shared_state contains model/optimizer plus identical optional metadata;
        RNG belongs in rng_state, because ranks may differ. sampler_state requires
        common next_global_batch and may contain arbitrary rank-local cursor data.
        """
        error = None; prepared = None; signature = None
        try:
            if self.shard._async_snapshot_ticket is not None:
                raise RuntimeError('finish asynchronous private snapshot before distributed commit')
            if (not isinstance(shared_state, Mapping) or not {'model', 'optimizer'} <= shared_state.keys()
                    or 'rng' in shared_state):
                raise ValueError('shared model/optimizer required; RNG is explicitly per rank')
            if not isinstance(rng_state, Mapping) or not isinstance(sampler_state, Mapping):
                raise TypeError('explicit per-rank RNG and sampler mappings required')
            cursor = sampler_state.get('next_global_batch')
            if type(cursor) is not int or cursor < 0: raise ValueError('nonnegative global batch cursor required')
            signature = (_digest(shared_state), self.generation, cursor, self.run_identity)
            prepared = self._prepare_index(private_updates)
        except Exception as exc: error = repr(exc)
        _raise_collective_errors(error)
        signatures = _gather(signature)
        if any(s != signatures[0] for s in signatures):
            raise ValueError('shared model/Adam, generation or global cursor differ across ranks')
        rows = _gather(dict(rank=self.rank, index=prepared[0], rng=_cpu(rng_state), sampler=_cpu(sampler_state)))
        result = [None]
        if self.rank == 0:
            try:
                self._validate_published_indices(rows)
                generation = self.global_store.commit(
                    shared_state={**shared_state, 'rng': {str(row['rank']):row['rng'] for row in rows}},
                    sampler_state=dict(next_global_batch=signature[2], schema=SCHEMA, run_identity=self.run_identity,
                        world_size=self.world_size, ranks=[dict(rank=row['rank'], index=row['index'], sampler=row['sampler']) for row in rows]),
                    private_updates=[], expected_generation=self.generation)
                result[0] = dict(generation=generation, global_manifest_ref=self.global_store._bound_reference)
            except Exception as exc: result[0] = dict(error=repr(exc))
        dist.broadcast_object_list(result, src=0)
        if 'error' in result[0]: raise RuntimeError('coordinator publication failed: '+result[0]['error'])
        self.generation = result[0]['generation']; self._index_ref = prepared[0]; self._versions = prepared[1]
        self._global_manifest_ref = result[0]['global_manifest_ref']
        for token, version in list(self.shard._pending.items()):
            if self._versions.get(token, {}).get('file') == version['file']: del self.shard._pending[token]
        return self.generation

    def restore(self):
        """Collectively pin one global generation; return CPU shared/rank state.

        No parameter/RNG mutation occurs here. Callers load returned shared model,
        Adam, rank RNG/cursor and then stage private IDs from this pinned index.
        """
        _raise_collective_errors('finish asynchronous private snapshot before distributed restore'
                                 if self.shard._async_snapshot_ticket is not None else None)
        result = [None]
        if self.rank == 0:
            try:
                _, reference = self.global_store._manifest_and_reference()
                if reference is None: raise FileNotFoundError('no global distributed checkpoint')
                result[0] = dict(reference=reference)
            except Exception as exc: result[0] = dict(error=repr(exc))
        dist.broadcast_object_list(result, src=0)
        if 'error' in result[0]: raise RuntimeError(result[0]['error'])
        error = None; state = None; index = None
        try:
            state = self.global_store.restore(manifest_ref=result[0]['reference'])
            sampler = state['sampler']
            if (sampler.get('schema') != SCHEMA or sampler.get('run_identity') != self.run_identity
                    or sampler.get('world_size') != self.world_size):
                raise ValueError('distributed run/world identity changed; explicit resharding required')
            if [row['rank'] for row in sampler['ranks']] != list(range(self.world_size)):
                raise ValueError('global manifest does not cover all ranks exactly once')
            row = sampler['ranks'][self.rank]
            index = torch.load(self.shard._verify(row['index']), map_location='cpu', weights_only=False)
            if (index.get('schema') != SCHEMA or index['run_identity'] != self.run_identity
                    or index['rank'] != self.rank or index['world_size'] != self.world_size
                    or index['generation'] != state['generation']):
                raise ValueError('private shard and shared transaction generations differ')
            for token, version in index['private_versions'].items():
                identity = PrivateIdentity(tuple(version['identity']['state_key']),version['identity']['state_sha256'],version['identity']['layout_sha256'])
                if identity.token != token or version['token'] != token: raise ValueError('private index identity differs')
                self._require_owner(identity)
            if row['sampler']['next_global_batch'] != sampler['next_global_batch']:
                raise ValueError('rank sampler/global cursor differ')
        except Exception as exc: error = repr(exc)
        _raise_collective_errors(error)
        self.generation = state['generation']; self._index_ref = row['index']; self._versions = index['private_versions']
        self._global_manifest_ref = state['manifest_ref']
        self.shard._pending.clear()
        return dict(generation=self.generation, shared={k:v for k,v in state['shared'].items() if k!='rng'},
                    rng=state['shared']['rng'][str(self.rank)], sampler=row['sampler'],
                    global_manifest_ref=state['manifest_ref'], private_index_ref=self._index_ref)

    def retain(self, name, *, manifest_ref=None):
        """Collectively pin a complete shared/private generation before a reader.

        Use named pins for saved research checkpoints and live readers. A reader
        must restore this explicit reference, and close before release_pin. No
        TTL is inferred; unregistered readers are outside the GC contract.
        """
        requests = _gather((name, manifest_ref))
        if any(_json(v) != _json(requests[0]) for v in requests):
            raise ValueError('retention requests differ across ranks')
        result = [None]
        if self.rank == 0:
            try:
                manifest, reference = self.global_store._manifest_and_reference(manifest_ref)
                if manifest is None: raise FileNotFoundError('no distributed checkpoint to retain')
                self._read_global_root(reference)
                result[0] = dict(reference=self.global_store.retain(name, manifest_ref=reference))
            except Exception as exc: result[0] = dict(error=repr(exc))
        dist.broadcast_object_list(result, src=0)
        if 'error' in result[0]: raise RuntimeError('distributed retention failed: '+result[0]['error'])
        return result[0]['reference']

    def release_pin(self, name):
        """Collective explicit release; deletes no checkpoint or index files."""
        names = _gather(name)
        if any(n != names[0] for n in names): raise ValueError('retention releases differ across ranks')
        result = [None]
        if self.rank == 0:
            try: self.global_store.release_pin(name)
            except Exception as exc: result[0] = repr(exc)
        dist.broadcast_object_list(result, src=0)
        if result[0] is not None: raise RuntimeError('distributed pin release failed: '+result[0])

    def _read_global_root(self, reference):
        manifest, _ = self.global_store._manifest_and_reference(reference)
        state = torch.load(self.global_store._verify(manifest['global_state']), map_location='cpu', weights_only=False)
        sampler = state['sampler']
        if (state.get('schema') != manifest['schema'] or state['generation'] != manifest['generation'] or sampler.get('schema') != SCHEMA
                or sampler.get('world_size') != self.world_size or sampler.get('run_identity') != self.run_identity
                or [r['rank'] for r in sampler['ranks']] != list(range(self.world_size))):
            raise ValueError('retention root has incompatible distributed identity/coverage')
        return manifest, sampler

    def _coordinator_gc_plan(self, rank_roots):
        """Reachability follows verified global→rank index→private references."""
        current, current_ref = self.global_store._manifest_and_reference()
        if current is None: raise RuntimeError('no complete distributed transaction to collect')
        if current['generation'] > 1 and 'previous_manifest' not in current:
            raise ValueError('legacy manifest lacks explicit previous-transaction retention')
        pins = self.global_store._pins()
        roots = [current_ref, current.get('previous_manifest')]
        roots += list(pins['pins'].values()) + [r['bound'] for r in rank_roots]
        roots = {r['file']:r for r in roots if r is not None}
        protected = {'global':{}}
        protected.update({f'private/rank{r:04d}':{} for r in range(self.world_size)})

        def protect(store, namespace, reference):
            path = store._path(reference); stamp = store._file_stamp(path)
            if stamp[2] != reference['bytes']: raise ValueError('protected distributed artifact size differs')
            protected[namespace][path.name] = stamp

        for reference in roots.values():
            manifest, sampler = self._read_global_root(reference)
            protect(self.global_store, 'global', reference)
            protect(self.global_store, 'global', manifest['global_state'])
            if manifest['private_versions']: raise ValueError('distributed private blobs must live in rank shards')
            for row in sampler['ranks']:
                rank = row['rank']; namespace = f'private/rank{rank:04d}'
                with PrivateProbeStore(self.directory/namespace, read_only=True) as shard:
                    index = torch.load(shard._verify(row['index']), map_location='cpu', weights_only=False)
                    if (index.get('schema') != SCHEMA or index['run_identity'] != self.run_identity
                            or index['world_size'] != self.world_size or index['rank'] != rank
                            or index['generation'] != manifest['generation']):
                        raise ValueError('protected rank index differs from global transaction')
                    protect(shard, namespace, row['index'])
                    for token, version in index['private_versions'].items():
                        identity = PrivateIdentity(tuple(version['identity']['state_key']), version['identity']['state_sha256'], version['identity']['layout_sha256'])
                        if token != identity.token or version['token'] != token or self.owner(identity) != rank:
                            raise ValueError('protected private identity/owner differs')
                        protect(shard, namespace, version)
        for rank, row in enumerate(rank_roots):
            namespace = f'private/rank{rank:04d}'
            with PrivateProbeStore(self.directory/namespace, read_only=True) as shard:
                for token, version in row['pending'].items():
                    identity = PrivateIdentity(tuple(version['identity']['state_key']), version['identity']['state_sha256'], version['identity']['layout_sha256'])
                    if token != identity.token or version['token'] != token or self.owner(identity) != rank:
                        raise ValueError('pending private identity/owner differs')
                    protect(shard, namespace, version)
        candidates = []
        for namespace in protected:
            directory = self.directory/namespace
            pattern = (r'(shared)-[0-9a-f]{32}\.pt|manifest-[0-9a-f]{32}\.json'
                       if namespace == 'global' else r'(private|index)-[0-9a-f]{32}\.pt')
            for path in sorted(directory.iterdir()):
                if re.fullmatch(pattern, path.name) and path.name not in protected[namespace]:
                    candidates.append(dict(namespace=namespace, file=path.name, stamp=PrivateProbeStore._file_stamp(path)))
        plan = dict(schema=SCHEMA+'_gc_plan', run_identity=self.run_identity, world_size=self.world_size,
                    current=current_ref, pins=pins, rank_roots=rank_roots, protected=protected,
                    candidates=candidates, reclaimable_bytes=sum(v['stamp'][2] for v in candidates))
        plan['plan_sha256'] = hashlib.sha256(_json(plan)).hexdigest()
        return plan

    def gc_plan(self, *, at_global_barrier=False):
        """Collective dry run only; current/previous/pins/bound/pending survive.

        Every rank holds its writer lock and participates. It is a caller error
        to collect during a visit or while an unregistered reader uses old data.
        No shard CURRENT exists, and no rank independently selects a root.
        """
        _raise_collective_errors(None if at_global_barrier is True else 'explicit completed global barrier required')
        _raise_collective_errors('finish asynchronous private snapshot before distributed collection'
                                 if self.shard._async_snapshot_ticket is not None else None)
        dist.barrier()
        rank_roots = _gather(dict(bound=self._global_manifest_ref, pending=self.shard._pending))
        result = [None]
        if self.rank == 0:
            try: result[0] = dict(plan=self._coordinator_gc_plan(rank_roots))
            except Exception as exc: result[0] = dict(error=repr(exc))
        dist.broadcast_object_list(result, src=0)
        if 'error' in result[0]: raise RuntimeError('distributed retention planning failed: '+result[0]['error'])
        return result[0]['plan']

    def _delete_gc_candidates(self, candidates):
        """Local deletion of reviewed unreachable UUID files under owned paths."""
        deleted = []
        for row in candidates:
            path = self.directory/row['namespace']/row['file']
            if PrivateProbeStore._file_stamp(path) != row['stamp']:
                raise ValueError('distributed artifact changed during collection')
            path.unlink(); deleted.append(row)
        for namespace in {r['namespace'] for r in candidates}: _sync_dir(self.directory/namespace)
        return deleted

    def apply_gc(self, plan, *, at_global_barrier=False):
        """Apply one reviewed plan collectively; never called by training/commit.

        Revalidate all roots/pins/pending/files before any mutation. Rank shards
        delete only unreachable files, then acknowledge before global deletion.
        A failure can leave garbage, but cannot remove any retained generation.
        Tests use temporary stores only; live deployment needs an explicit call.
        """
        actual = self.gc_plan(at_global_barrier=at_global_barrier)
        error = None
        try:
            if _json(plan) != _json(actual): error = 'stale or altered distributed retention plan'
        except Exception as exc: error = repr(exc)
        _raise_collective_errors(error)
        local = [r for r in actual['candidates'] if r['namespace'] == f'private/rank{self.rank:04d}']
        error = None; deleted = []
        try: deleted = self._delete_gc_candidates(local)
        except Exception as exc: error = repr(exc)
        _raise_collective_errors(error)
        all_deleted = _gather(deleted)
        result = [None]
        if self.rank == 0:
            try:
                global_deleted = self._delete_gc_candidates([r for r in actual['candidates'] if r['namespace']=='global'])
                rows = [r for part in all_deleted for r in part] + global_deleted
                result[0] = dict(plan_sha256=actual['plan_sha256'], deleted=rows, reclaimed_bytes=sum(r['stamp'][2] for r in rows))
            except Exception as exc: result[0] = dict(error=repr(exc))
        dist.broadcast_object_list(result, src=0)
        if 'error' in result[0]: raise RuntimeError('distributed global collection failed: '+result[0]['error'])
        return result[0]
