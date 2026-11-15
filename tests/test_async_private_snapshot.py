"""CPU transaction/ownership checks, not a GPU throughput qualification."""
import copy
import threading

import pytest
import torch

from rewriter.async_private_snapshot import AsyncPrivateSnapshot
from rewriter.distributed_probe_store import _digest
from rewriter.private_probe_store import PrivateProbeStore
from tests.test_private_probe_store import context, step, commit, few_threads


@pytest.mark.parametrize('backend', ['reference', 'owned_cpu_v1', 'pinned_d2h_v1'])
def test_owned_values_and_exact_next_adam_with_background_write(tmp_path, monkeypatch, backend):
    ctx = context(); step(ctx)
    expected_private = copy.deepcopy(ctx[1].state_dict())
    expected_adam = copy.deepcopy(ctx[4].state_dict())
    entered, release = threading.Event(), threading.Event()
    with PrivateProbeStore(tmp_path, snapshot_backend=backend) as store:
        original = store._write_slot_snapshot
        def blocked(identity, payload):
            entered.set()
            assert release.wait(10), 'test writer was not released'
            return original(identity, payload)
        monkeypatch.setattr(store, '_write_slot_snapshot', blocked)
        with AsyncPrivateSnapshot(store) as writer:
            writer.submit(ctx[-1], ctx[3], ctx[4], visits=1)
            assert entered.wait(10)
            try:
                # Simulates unrelated compute after submit; the saved values
                # must not alias any subsequently mutated live tensor/moment.
                step(ctx)
                with pytest.raises(RuntimeError, match='before submitting another'):
                    writer.submit(ctx[-1], ctx[3], ctx[4], visits=2)
                with pytest.raises(RuntimeError, match='before next snapshot'):
                    store.snapshot_slot(ctx[-1], ctx[3], ctx[4], visits=2)
                with pytest.raises(RuntimeError, match='before commit'):
                    store.commit(shared_state={}, sampler_state={}, private_updates=[], expected_generation=0)
                with pytest.raises(RuntimeError, match='same identity'):
                    store.read_private(ctx[-1])
                with pytest.raises(RuntimeError, match='before restore'):
                    store.restore()
                with pytest.raises(RuntimeError, match='before collection'):
                    store.gc_plan(at_global_barrier=True)
            finally:
                release.set()
            assert ctx[-1].token not in store._pending
            version = writer.finish()
            assert set(writer.last_timings) == {"prepare_seconds", "write_seconds", "finish_wait_seconds"}
            assert all(value >= 0 for value in writer.last_timings.values())
            payload = store.read_private(ctx[-1], version=version)
            assert _digest(payload['private']) == _digest(expected_private)
            assert _digest(payload['optimizer']) == _digest(expected_adam)
            assert not (tmp_path/'CURRENT.json').exists()
            restored = context()
            restored[0].load_state_dict(ctx[0].state_dict())
            restored[5].load_state_dict(ctx[5].state_dict())
            store.stage(ctx[-1], restored[3], restored[4], version=version)
            reference = context()
            reference[0].load_state_dict(ctx[0].state_dict())
            reference[5].load_state_dict(ctx[5].state_dict())
            reference[1].load_state_dict(expected_private)
            reference[4].load_state_dict(expected_adam)
            step(restored); step(reference)
            assert _digest(restored[1].state_dict()) == _digest(reference[1].state_dict())
            assert _digest(restored[4].state_dict()) == _digest(reference[4].state_dict())


def test_failed_write_leaves_committed_generation_unchanged(tmp_path, monkeypatch):
    ctx = context(); step(ctx)
    with PrivateProbeStore(tmp_path) as store:
        commit(store, ctx, 1, 0)
        pointer = (tmp_path/'CURRENT.json').read_bytes()
        step(ctx)
        def fail(*args):
            raise OSError('injected background write failure')
        monkeypatch.setattr(store, '_write_slot_snapshot', fail)
        with AsyncPrivateSnapshot(store) as writer:
            writer.submit(ctx[-1], ctx[3], ctx[4], visits=2)
            with pytest.raises(OSError, match='background write failure'):
                writer.finish()
        assert store._async_snapshot_ticket is None
        assert (tmp_path/'CURRENT.json').read_bytes() == pointer
        assert store.restore()['sampler']['next_global_batch'] == 1


def test_close_drains_write_and_invalid_payload_never_queues(tmp_path, monkeypatch):
    ctx = context(); step(ctx)
    with PrivateProbeStore(tmp_path) as store:
        writer = AsyncPrivateSnapshot(store)
        with pytest.raises(ValueError, match='counter'):
            writer.submit(ctx[-1], ctx[3], ctx[4], visits=0)
        assert store._async_snapshot_ticket is None
        writer.submit(ctx[-1], ctx[3], ctx[4], visits=1)
        writer.close()
        assert store._async_snapshot_ticket is None
        assert ctx[-1].token in store._pending
        with pytest.raises(RuntimeError, match='closed'):
            writer.submit(ctx[-1], ctx[3], ctx[4], visits=1)


def _distributed_worker(rank, path, rendezvous):
    import datetime
    import torch.distributed as dist
    from rewriter.distributed_probe_store import DistributedProbeStore, private_owner
    from rewriter.private_probe_store import PrivateIdentity
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://'+str(rendezvous), rank=rank,
                            world_size=2, timeout=datetime.timedelta(seconds=45))
    try:
        ctx = context(); step(ctx)
        identity = ctx[-1]
        owner = private_owner(identity, 2)
        with DistributedProbeStore(path, run_identity='async-cpu-qualification') as store:
            entered, release = threading.Event(), threading.Event()
            original = store.shard._write_slot_snapshot
            def blocked(identity, payload):
                entered.set()
                assert release.wait(15)
                return original(identity, payload)
            store.shard._write_slot_snapshot = blocked
            with AsyncPrivateSnapshot(store) as writer:
                if rank == owner:
                    writer.submit(identity, ctx[3], ctx[4], visits=1)
                    assert entered.wait(10)
                else:
                    with pytest.raises(ValueError, match='different identity owner'):
                        writer.submit(identity, ctx[3], ctx[4], visits=1)
                kwargs=dict(shared_state={'model':ctx[0].state_dict(), 'optimizer':ctx[5].state_dict()},
                    rng_state={'torch':torch.get_rng_state()}, sampler_state={'next_global_batch':1})
                try:
                    with pytest.raises(RuntimeError, match='before distributed commit'):
                        store.commit(**kwargs, private_updates=[])
                    with pytest.raises(RuntimeError, match='before distributed restore'):
                        store.restore()
                    with pytest.raises(RuntimeError, match='before distributed collection'):
                        store.gc_plan(at_global_barrier=True)
                    assert not (path/'global/CURRENT.json').exists()
                finally:
                    release.set()
                versions=[writer.finish()] if rank==owner else []
                assert store.commit(**kwargs,private_updates=versions)==1
                restored=store.restore()
                assert _digest(restored['shared'])==_digest(kwargs['shared_state'])
                if rank==owner:
                    assert store.stage(identity,ctx[3],ctx[4])==1
    finally:
        dist.destroy_process_group()


def test_collective_commit_rejects_inflight_and_retains_private_ownership(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_distributed_worker,args=(tmp_path/'store',tmp_path/'rendezvous'),nprocs=2,join=True)


@pytest.mark.parametrize('write_fails', [False, True])
def test_store_cleanup_drains_writer_preserves_original_exception_and_unlocks(tmp_path, monkeypatch, write_fails):
    ctx = context(); step(ctx)
    writer = None
    with pytest.raises(ValueError, match='original compute failure') as error:
        with PrivateProbeStore(tmp_path) as store:
            if write_fails:
                def fail(*args):
                    raise OSError('injected cleanup write failure')
                monkeypatch.setattr(store, '_write_slot_snapshot', fail)
            writer = AsyncPrivateSnapshot(store)
            writer.submit(ctx[-1], ctx[3], ctx[4], visits=1)
            raise ValueError('original compute failure')
    assert writer._closed and writer._future is None
    assert store._lock is None
    if write_fails:
        assert 'cleanup write failure' in error.value.__notes__[0]
    with PrivateProbeStore(tmp_path):
        pass


def test_direct_store_close_propagates_write_failure_and_unlocks(tmp_path, monkeypatch):
    ctx = context(); step(ctx)
    store = PrivateProbeStore(tmp_path)
    def fail(*args):
        raise OSError('injected close failure')
    monkeypatch.setattr(store, '_write_slot_snapshot', fail)
    writer = AsyncPrivateSnapshot(store)
    writer.submit(ctx[-1], ctx[3], ctx[4], visits=1)
    with pytest.raises(OSError, match='close failure'):
        store.close()
    assert writer._closed and store._lock is None
    with PrivateProbeStore(tmp_path):
        pass


def test_other_identity_can_stage_while_snapshot_is_pending(tmp_path, monkeypatch):
    from rewriter.private_probe_store import PrivateIdentity
    ctx = context(); step(ctx)
    from rewriter.coordinate_chisel import CheckpointProbes
    other_private = CheckpointProbes(ctx[2], model_dim=128, latent_tokens=2,
                                     seed=999, state_key=('gpt', 0, 999))
    other = PrivateIdentity.from_private(other_private, state_sha256='b'*64)
    original_state = copy.deepcopy(ctx[1].state_dict())
    release = threading.Event()
    with PrivateProbeStore(tmp_path) as store:
        ctx[3].stage(other_private.state_dict(), state_key=other.state_key)
        version = store.snapshot_slot(other, ctx[3], ctx[4], visits=1)
        ctx[3].stage(original_state, state_key=ctx[-1].state_key)
        original = store._write_slot_snapshot
        def blocked(identity, payload):
            assert release.wait(10)
            return original(identity, payload)
        monkeypatch.setattr(store, '_write_slot_snapshot', blocked)
        with AsyncPrivateSnapshot(store) as writer:
            writer.submit(ctx[-1], ctx[3], ctx[4], visits=1)
            try:
                with pytest.raises(RuntimeError, match='same identity'):
                    store.stage(ctx[-1], ctx[3], ctx[4], version=version)
                assert store.stage(other, ctx[3], ctx[4], version=version) == 1
            finally:
                release.set()
            writer.finish()


@pytest.mark.parametrize('writer_context', [False, True])
def test_python310_exception_without_add_note_is_preserved(tmp_path, monkeypatch, writer_context):
    class OldException(Exception):
        add_note = None
    ctx = context(); step(ctx)
    with pytest.raises(OldException, match='original') as error:
        with PrivateProbeStore(tmp_path) as store:
            def fail(*args):
                raise OSError('cleanup error')
            monkeypatch.setattr(store, '_write_slot_snapshot', fail)
            writer = AsyncPrivateSnapshot(store)
            if writer_context:
                with writer:
                    writer.submit(ctx[-1], ctx[3], ctx[4], visits=1)
                    raise OldException('original')
            else:
                writer.submit(ctx[-1], ctx[3], ctx[4], visits=1)
                raise OldException('original')
    assert isinstance(error.value.__context__, OSError)
    assert writer._closed and store._lock is None


def test_distributed_cleanup_releases_global_store_after_shard_failure():
    from rewriter.distributed_probe_store import DistributedProbeStore
    class OldException(Exception):
        add_note = None
    closed = []
    class Store:
        def __init__(self, name): self.name = name
        def close(self):
            closed.append(self.name)
            if self.name == 'shard': raise OSError('shard cleanup failure')
    store = object.__new__(DistributedProbeStore)
    store.shard, store.global_store = Store('shard'), Store('global')
    with pytest.raises(OldException, match='original') as error:
        with store:
            raise OldException('original')
    assert closed == ['shard', 'global']
    assert isinstance(error.value.__context__, OSError)
