"""Explicit quiescent, same-world numerical checkpoint fork; no training.

The source CURRENT must equal the caller's immutable reference. Cooperative
writer locks on the global store and every shard reject live sources. Only the
referenced latest private payloads are copied, byte-for-byte. A complete hidden
sibling is built with the existing collective commit API, then rank0 atomically
renames it into the destination. Failed attempts are retained unpublished;
retries explicitly reject their pending directory rather than deleting evidence.
"""
from __future__ import annotations
import copy
import fcntl
import os
from pathlib import Path
import shutil
import uuid
import torch.distributed as dist
from .distributed_probe_store import DistributedProbeStore, _digest, _gather, _raise_collective_errors
from .private_probe_store import PrivateProbeStore, PrivateIdentity, _json, _sync_dir

SCHEMA='chisel_distributed_checkpoint_fork_v1'


def _broadcast_result(function):
    result=[None]
    if dist.get_rank()==0:
        try:result[0]={'value':function()}
        except Exception as exc:result[0]={'error':repr(exc)}
    dist.broadcast_object_list(result,src=0)
    if 'error' in result[0]:raise RuntimeError('checkpoint fork aborted: '+result[0]['error'])
    return result[0]['value']


def _copy_private(source,target,identity,version):
    # Validate the payload as well as its hash before and after the byte copy.
    source.read_private(identity,version=version)
    original=source._verify(version);destination=target.directory/version['file']
    if original.is_symlink() or not original.is_file():raise ValueError('private payload must be a regular owned file')
    with original.open('rb') as incoming,destination.open('xb') as outgoing:
        shutil.copyfileobj(incoming,outgoing,length=8<<20);outgoing.flush();os.fsync(outgoing.fileno())
    _sync_dir(target.directory)
    target.read_private(identity,version=version)
    return copy.deepcopy(version)


def fork_distributed_checkpoint(source,destination,*,parent_manifest_ref,source_run_identity,
                                destination_run_identity,provenance):
    """Collective CPU-only fork into an absent or empty destination store.

    A new transaction generation starts at1; training clocks are NOT reset.
    Caller supplies explicit source/destination code IDs and migration reason.
    This utility does not rewrite trainer metadata or authorize a backend.
    It returns a provenance record whose destination manifest pins all data.
    """
    if not dist.is_initialized():raise RuntimeError('initialized process group required')
    if dist.get_backend()!='gloo':raise ValueError('checkpoint fork is CPU Gloo only')
    source=Path(source).absolute();destination=Path(destination).absolute()
    request=dict(source=str(source),destination=str(destination),parent_manifest_ref=parent_manifest_ref,
                 source_run_identity=source_run_identity,destination_run_identity=destination_run_identity,provenance=provenance)
    requests=_gather(request)
    if any(r!=requests[0] for r in requests):raise ValueError('fork arguments differ across ranks')
    if (not isinstance(source_run_identity,str) or not source_run_identity or not isinstance(destination_run_identity,str)
            or not destination_run_identity or source_run_identity==destination_run_identity):
        raise ValueError('distinct explicit source and destination run identities required')
    if not isinstance(provenance,dict) or any(not isinstance(provenance.get(k),str) or not provenance[k]
            for k in ('source_code_id','destination_code_id','reason')):raise ValueError('explicit code identities and fork reason required')
    _json(request)  # Require reproducible JSON metadata on every rank.
    world=dist.get_world_size();rank=dist.get_rank();lock=None
    def prepare():
        nonlocal lock
        src=source.resolve(strict=True);dst=destination.resolve()
        if src==dst or src in dst.parents or dst in src.parents:raise ValueError('source/destination stores must be disjoint')
        if source.is_symlink() or destination.is_symlink():raise ValueError('store roots cannot be symlinks')
        for directory in [source/'global',*[source/'private'/f'rank{i:04d}' for i in range(world)]]:
            if not directory.is_dir() or not (directory/'writer.lock').is_file():raise ValueError('source store layout incomplete')
        with PrivateProbeStore(source/'global',read_only=True) as reader:
            current,ref=reader._manifest_and_reference()
            if ref!=parent_manifest_ref:raise ValueError('source CURRENT differs from explicitly pinned parent')
            state=reader.restore(manifest_ref=ref)
            if state['sampler'].get('run_identity')!=source_run_identity:raise ValueError('source run identity differs')
            if state['sampler'].get('world_size')!=world:raise ValueError('world size changed; no implicit resharding')
        destination.parent.mkdir(parents=True,exist_ok=True)
        lock=(destination.parent/('.'+destination.name+'.fork.lock')).open('a+b')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):raise ValueError('destination must be absent or empty')
        pending=list(destination.parent.glob('.'+destination.name+'.fork-pending-*'))
        if pending:raise ValueError('unpublished fork attempt exists; inspect explicitly, no automatic cleanup: '+str(pending))
        temporary=destination.parent/('.'+destination.name+'.fork-pending-'+uuid.uuid4().hex)
        return str(temporary)
    try:
        temporary=Path(_broadcast_result(prepare))
        # Constructor locks fail before source mutation if an active trainer owns it.
        with DistributedProbeStore(source,run_identity=source_run_identity) as parent:
            state=parent.restore()
            if state['global_manifest_ref']!=parent_manifest_ref:raise ValueError('source advanced before quiescent lock')
            def begin():
                temporary.mkdir(exist_ok=False)
                marker=dict(schema=SCHEMA,status='unpublished',request=request)
                with (temporary/'FORK_PENDING.json').open('xb') as f:f.write(_json(marker));f.flush();os.fsync(f.fileno())
                _sync_dir(temporary)
            _broadcast_result(begin)
            with DistributedProbeStore(temporary,run_identity=destination_run_identity) as child:
                copied=[];error=None
                try:
                    for token,version in sorted(parent._versions.items()):
                        identity=PrivateIdentity(tuple(version['identity']['state_key']),version['identity']['state_sha256'],version['identity']['layout_sha256'])
                        if token!=identity.token:raise ValueError('private token differs')
                        copied.append(_copy_private(parent.shard,child.shard,identity,version))
                except Exception as exc:error=repr(exc)
                _raise_collective_errors(error)
                child.commit(shared_state=state['shared'],rng_state=state['rng'],sampler_state=state['sampler'],private_updates=copied)
                restored=child.restore();error=None
                try:
                    for field in ('shared','rng','sampler'):
                        if _digest(restored[field])!=_digest(state[field]):raise ValueError('fork altered numerical '+field)
                    if child._versions!=parent._versions:raise ValueError('fork altered private version identities')
                except Exception as exc:error=repr(exc)
                _raise_collective_errors(error)
                rows=_gather(dict(rank=rank,private_payloads=len(copied),private_bytes=sum(v['bytes'] for v in copied),
                                 shared_digest=_digest(restored['shared']),rng_digest=_digest(restored['rng']),sampler_digest=_digest(restored['sampler'])))
                manifest=copy.deepcopy(restored['global_manifest_ref'])
            # All shard/global handles in the pending destination are now closed.
            dist.barrier()
            def publish():
                if parent.global_store._manifest_and_reference()[1]!=parent_manifest_ref:raise ValueError('source changed while locked')
                if destination.exists() and any(destination.iterdir()):raise ValueError('destination became nonempty')
                record=dict(schema=SCHEMA,parent_manifest_ref=parent_manifest_ref,source=str(source),destination=str(destination),
                    source_run_identity=source_run_identity,destination_run_identity=destination_run_identity,
                    world_size=world,destination_manifest_ref=manifest,provenance=provenance,ranks=rows,
                    numerical_state='unchanged',training_steps=0,source_generation=state['generation'],destination_generation=1)
                with (temporary/'FORK.json').open('xb') as f:f.write(_json(record));f.flush();os.fsync(f.fileno())
                (temporary/'FORK_PENDING.json').unlink();_sync_dir(temporary)
                os.rename(temporary,destination);_sync_dir(destination.parent)
                return record
            result=_broadcast_result(publish)
        return result
    finally:
        if lock is not None:
            fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
