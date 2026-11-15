"""Transactional storage tests use the real coordinate actor and stable slot."""
import copy
import os

import pytest
import torch

from rewriter.private_probe_store import PrivateIdentity, PrivateProbeStore
from rewriter.private_probe_slot import PrivateProbeSlot
from tests.test_coordinate_private_invariants import setup, objective


@pytest.fixture(autouse=True)
def few_threads():
    before = torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def context():
    actor, private, snapshot = setup()
    slot = PrivateProbeSlot(actor, private)
    optimizer = torch.optim.Adam(private.parameters(), lr=1e-3, betas=(.9, .99))
    shared_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    identity = PrivateIdentity.from_private(private, state_sha256='a'*64)
    return actor, private, snapshot, slot, optimizer, shared_optimizer, identity


def step(ctx):
    actor, private, snapshot, slot, optimizer, shared_optimizer, identity = ctx
    slot.require_bound(identity.state_key)
    shared_optimizer.zero_grad(set_to_none=True); optimizer.zero_grad(set_to_none=True)
    objective(actor, private, snapshot).backward()
    optimizer.step(); shared_optimizer.step(); optimizer.zero_grad(set_to_none=True)


def commit(store, ctx, visits, generation):
    actor, private, snapshot, slot, optimizer, shared_optimizer, identity = ctx
    version = store.snapshot_slot(identity, slot, optimizer, visits=visits)
    store.commit(shared_state={'model': actor.state_dict(), 'optimizer': shared_optimizer.state_dict(),
                               'rng': torch.get_rng_state()},
                 sampler_state={'next_global_batch': visits, 'schedule': 13},
                 private_updates=[version], expected_generation=generation)
    return version


def test_uncommitted_private_write_is_ignored_after_restart(tmp_path):
    ctx = context(); step(ctx)
    with PrivateProbeStore(tmp_path) as store:
        committed = commit(store, ctx, 1, 0)
        old = copy.deepcopy(ctx[1].state_dict())
        step(ctx)
        orphan = store.snapshot_slot(ctx[-1], ctx[3], ctx[4], visits=2)
        assert orphan['file'] != committed['file']
    with PrivateProbeStore(tmp_path) as reopened:
        assert reopened.restore()['sampler']['next_global_batch'] == 1
        assert reopened.stage(ctx[-1], ctx[3], ctx[4]) == 1
        for name, value in ctx[1].state_dict().items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, old[name], atol=0, rtol=0)
        assert (tmp_path/orphan['file']).exists()  # No destructive orphan collection.


def test_failure_before_pointer_replace_keeps_shared_and_private_old(tmp_path, monkeypatch):
    ctx = context(); step(ctx)
    with PrivateProbeStore(tmp_path) as store:
        old = commit(store, ctx, 1, 0)
        step(ctx)
        pending = store.snapshot_slot(ctx[-1], ctx[3], ctx[4], visits=2)
        original = os.replace
        def interrupted(source, destination):
            if str(destination).endswith('CURRENT.json'):
                raise OSError('simulated interruption before atomic publication')
            return original(source, destination)
        with monkeypatch.context() as patch:
            patch.setattr(os, 'replace', interrupted)
            with pytest.raises(OSError, match='simulated interruption'):
                store.commit(shared_state={'model': {}, 'optimizer': {}, 'rng': {}, 'changed': True}, sampler_state={'next_global_batch': 2},
                             private_updates=[pending], expected_generation=1)
        restored = store.restore()
        assert restored['generation'] == 1
        assert restored['sampler']['next_global_batch'] == 1
        assert restored['private_versions'][ctx[-1].token] == old


def test_resume_exact_next_joint_update_and_slot_addresses(tmp_path):
    ctx = context(); step(ctx)
    with PrivateProbeStore(tmp_path) as store:
        commit(store, ctx, 1, 0)
        restored = store.restore()
        fresh = context(); before_addresses = fresh[3].storage_signature()
        fresh[0].load_state_dict(restored['shared']['model'])
        fresh[5].load_state_dict(restored['shared']['optimizer'])
        assert store.stage(fresh[-1], fresh[3], fresh[4]) == 1
        assert fresh[3].storage_signature() == before_addresses
        step(ctx); step(fresh)
        for a, b in ((ctx[0], fresh[0]), (ctx[1], fresh[1])):
            for p, q in zip(a.parameters(), b.parameters()):
                torch.testing.assert_close(p, q, atol=0, rtol=0)
        for optimizer, restored_optimizer in ((ctx[4], fresh[4]), (ctx[5], fresh[5])):
            for (p, state), (q, other) in zip(optimizer.state.items(), restored_optimizer.state.items()):
                for key in ('step', 'exp_avg', 'exp_avg_sq'):
                    torch.testing.assert_close(state[key], other[key], atol=0, rtol=0)


def test_cross_seed_source_and_layout_rejected_before_mutation(tmp_path):
    ctx = context(); step(ctx)
    with PrivateProbeStore(tmp_path) as store:
        version = commit(store, ctx, 1, 0)
        identity = ctx[-1]
        wrong = [PrivateIdentity(('gpt', 1, 1), identity.state_sha256, identity.layout_sha256),
                 PrivateIdentity(identity.state_key, 'b'*64, identity.layout_sha256),
                 PrivateIdentity(identity.state_key, identity.state_sha256, 'c'*64)]
        before = [p.clone() for p in ctx[1].parameters()]
        for altered in wrong:
            with pytest.raises(ValueError, match='identity'):
                store.stage(altered, ctx[3], ctx[4], version=version)
            with pytest.raises(KeyError, match='no committed'):
                store.stage(altered, ctx[3], ctx[4])
        for p, old in zip(ctx[1].parameters(), before):
            torch.testing.assert_close(p, old, atol=0, rtol=0)


@pytest.mark.parametrize('target', ['private', 'global', 'manifest'])
def test_tampered_artifacts_rejected(tmp_path, target):
    ctx = context(); step(ctx)
    with PrivateProbeStore(tmp_path) as store:
        version = commit(store, ctx, 1, 0)
        if target == 'private': filename = version['file']
        elif target == 'global': filename = store.manifest()['global_state']['file']
        else:
            import json
            filename = json.loads((tmp_path/'CURRENT.json').read_text())['file']
        with (tmp_path/filename).open('r+b') as file:
            file.seek(13); byte = file.read(1); file.seek(13); file.write(bytes([byte[0] ^ 1]))
        with pytest.raises(ValueError, match='hash'):
            if target == 'private': store.stage(ctx[-1], ctx[3], ctx[4])
            else: store.restore()


def test_single_writer_readonly_and_stale_transaction(tmp_path):
    ctx = context(); step(ctx)
    with PrivateProbeStore(tmp_path) as writer:
        version = commit(writer, ctx, 1, 0)
        with pytest.raises(RuntimeError, match='writer'):
            PrivateProbeStore(tmp_path)
        with PrivateProbeStore(tmp_path, read_only=True) as reader:
            assert reader.restore()['generation'] == 1
            assert reader.read_private(ctx[-1])['visits'] == 1
            with pytest.raises(RuntimeError, match='writer'):
                reader.snapshot_slot(ctx[-1], ctx[3], ctx[4], visits=1)
        with pytest.raises(ValueError, match='stale'):
            writer.commit(shared_state={}, sampler_state={}, private_updates=[version], expected_generation=0)


def test_pending_slot_snapshot_has_independent_cpu_storage(tmp_path):
    ctx = context(); step(ctx)
    with PrivateProbeStore(tmp_path) as store:
        version = store.snapshot_slot(ctx[-1], ctx[3], ctx[4], visits=1)
        payload = store.read_private(ctx[-1], version=version)
        original = {k: v.clone() for k,v in payload['private'].items() if isinstance(v,torch.Tensor)}
        with torch.no_grad():
            for p in ctx[1].parameters(): p.add_(3)
        again = store.read_private(ctx[-1], version=version)
        for name, value in original.items():
            torch.testing.assert_close(again['private'][name], value, atol=0, rtol=0)
        with pytest.raises(FileNotFoundError, match='no committed'):
            store.restore()


def test_reader_pins_private_versions_to_restored_shared_generation(tmp_path):
    ctx = context(); step(ctx)
    with PrivateProbeStore(tmp_path) as writer, PrivateProbeStore(tmp_path, read_only=True) as reader:
        commit(writer, ctx, 1, 0)
        with pytest.raises(RuntimeError, match='restore the global'):
            reader.stage(ctx[-1], ctx[3], ctx[4])
        old_global = reader.restore()
        assert old_global['generation'] == 1
        step(ctx); commit(writer, ctx, 2, 1)
        assert writer.manifest()['generation'] == 2
        # The read-only process must not combine generation1 phi with generation2 psi.
        assert reader.stage(ctx[-1], ctx[3], ctx[4]) == 1
        assert reader.read_private(ctx[-1])['visits'] == 1
        assert reader.restore()['generation'] == 2
        assert reader.stage(ctx[-1], ctx[3], ctx[4]) == 2


def test_global_commit_requires_complete_schema_and_cursor(tmp_path):
    with PrivateProbeStore(tmp_path) as store:
        with pytest.raises(ValueError, match='model, optimizer and RNG'):
            store.commit(shared_state={}, sampler_state={'next_global_batch': 0},
                         private_updates=[], expected_generation=0)
        with pytest.raises(ValueError, match='sampler cursor'):
            store.commit(shared_state={'model': {}, 'optimizer': {}, 'rng': {}}, sampler_state={},
                         private_updates=[], expected_generation=0)
        assert store.manifest() is None


def test_retention_roots_protect_named_checkpoint_previous_and_live_reader(tmp_path):
    ctx = context()
    with PrivateProbeStore(tmp_path) as store:
        step(ctx); commit(store, ctx, 1, 0)
        named = store.retain('research-epoch1')
        step(ctx); commit(store, ctx, 2, 1)
        reader_ref = store.retain('reader-audit')
        with PrivateProbeStore(tmp_path, read_only=True) as reader:
            reader.restore(manifest_ref=reader_ref)
            step(ctx); commit(store, ctx, 3, 2)
            step(ctx); commit(store, ctx, 4, 3)
            assert not store.gc_plan(at_global_barrier=True)['candidates']
            store.release_pin('research-epoch1')
            plan = store.gc_plan(at_global_barrier=True)
            assert named['file'] in [x['file'] for x in plan['candidates']]
            store.apply_gc(plan, at_global_barrier=True)
            assert store.restore()['generation'] == 4
            assert reader.stage(ctx[-1], ctx[3], ctx[4]) == 2
        store.release_pin('reader-audit')
        plan = store.gc_plan(at_global_barrier=True)
        assert reader_ref['file'] in [x['file'] for x in plan['candidates']]
        store.apply_gc(plan, at_global_barrier=True)
        previous = store.manifest()['previous_manifest']
        assert store.restore(manifest_ref=previous)['generation'] == 3
        assert store.stage(ctx[-1], ctx[3], ctx[4]) == 3


def test_pending_versions_coalesce_without_discarding_latest_moments(tmp_path):
    ctx = context(); step(ctx)
    with PrivateProbeStore(tmp_path) as store:
        commit(store, ctx, 1, 0)
        step(ctx); superseded = store.snapshot_slot(ctx[-1], ctx[3], ctx[4], visits=2)
        step(ctx); latest = store.snapshot_slot(ctx[-1], ctx[3], ctx[4], visits=3)
        plan = store.gc_plan(at_global_barrier=True)
        names = {row['file'] for row in plan['candidates']}
        assert superseded['file'] in names
        assert latest['file'] not in names
        store.apply_gc(plan, at_global_barrier=True)
        assert store.stage(ctx[-1], ctx[3], ctx[4], version=latest) == 3
        assert all(float(v['step']) == 3 for v in ctx[4].state.values())
        store.commit(shared_state={'model': ctx[0].state_dict(), 'optimizer': ctx[5].state_dict(), 'rng': torch.get_rng_state()},
                     sampler_state={'next_global_batch': 3}, private_updates=[latest], expected_generation=1)
        assert store.restore()['sampler']['next_global_batch'] == 3
        assert store.stage(ctx[-1], ctx[3], ctx[4]) == 3


@pytest.mark.parametrize('change', ['commit', 'pin', 'pending', 'candidate_file'])
def test_stale_gc_plan_fails_closed(tmp_path, change):
    ctx = context(); step(ctx)
    with PrivateProbeStore(tmp_path) as store:
        commit(store, ctx, 1, 0)
        step(ctx); orphan = store.snapshot_slot(ctx[-1], ctx[3], ctx[4], visits=2)
        step(ctx); store.snapshot_slot(ctx[-1], ctx[3], ctx[4], visits=3)
        plan = store.gc_plan(at_global_barrier=True)
        if change == 'commit': commit(store, ctx, 3, 1)
        elif change == 'pin': store.retain('new-reader')
        elif change == 'pending':
            step(ctx); store.snapshot_slot(ctx[-1], ctx[3], ctx[4], visits=4)
        else:
            with (tmp_path/orphan['file']).open('ab') as stream: stream.write(b'x')
        with pytest.raises(ValueError, match='stale|altered'):
            store.apply_gc(plan, at_global_barrier=True)
        assert (tmp_path/orphan['file']).exists()
        assert store.restore()['generation'] in (1, 2)


def test_gc_only_explicit_barrier_and_owned_artifacts(tmp_path):
    ctx = context(); step(ctx)
    unrelated = tmp_path/'other-project.pt'; unrelated.write_bytes(b'do not delete')
    with PrivateProbeStore(tmp_path) as store:
        commit(store, ctx, 1, 0)
        with pytest.raises(RuntimeError, match='global barrier'): store.gc_plan()
        plan = store.gc_plan(at_global_barrier=True)
        with pytest.raises(RuntimeError, match='global barrier'): store.apply_gc(plan)
        store.apply_gc(plan, at_global_barrier=True)
        assert unrelated.read_bytes() == b'do not delete'
        assert (tmp_path/'CURRENT.json').exists()
        assert (tmp_path/'writer.lock').exists()


def test_32_states_staged_slot_matches_resident_joint_training(tmp_path):
    from rewriter.coordinate_chisel import CheckpointProbes
    from rewriter.coordinate_optimization import joint_state_backward
    actor, private, snapshot, slot, optimizer, shared_optimizer, _ = context()
    reference_actor = copy.deepcopy(actor)
    reference_shared = torch.optim.Adam(reference_actor.parameters(), lr=1e-4)
    reference_private = []
    reference_adam = []
    identities = []
    with PrivateProbeStore(tmp_path) as store:
        versions = []
        for state_id in range(1, 33):
            module = CheckpointProbes(snapshot, model_dim=128, latent_tokens=2,
                                     seed=63+state_id, state_key=('gpt', 0, state_id))
            identities.append(PrivateIdentity.from_private(module, state_sha256=f'{state_id:064x}'))
            reference_private.append(module)
            reference_adam.append(torch.optim.Adam(module.parameters(), lr=1e-3, betas=(.9,.99)))
            slot.stage(module.state_dict(), state_key=identities[-1].state_key)
            optimizer.load_state_dict(reference_adam[-1].state_dict())
            versions.append(store.snapshot_slot(identities[-1], slot, optimizer, visits=0))
        store.commit(shared_state={'model': actor.state_dict(), 'optimizer': shared_optimizer.state_dict(), 'rng': torch.get_rng_state()},
                     sampler_state={'next_global_batch': 0}, private_updates=versions, expected_generation=0)
        generation = 1
        addresses = slot.storage_signature()
        for epoch in range(2):
            for start in range(0, 32, 8):
                shared_optimizer.zero_grad(set_to_none=True); reference_shared.zero_grad(set_to_none=True)
                versions = []
                for index in range(start, start+8):
                    visits = store.stage(identities[index], slot, optimizer)
                    assert visits == epoch
                    joint_state_backward(objective(actor, slot.private, snapshot), state_weight=1/8,
                                         shared_parameters=actor.parameters(), private_parameters=slot.private.parameters())
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
                    versions.append(store.snapshot_slot(identities[index], slot, optimizer, visits=visits+1))
                    ref = reference_private[index]; refopt = reference_adam[index]
                    joint_state_backward(objective(reference_actor, ref, snapshot), state_weight=1/8,
                                         shared_parameters=reference_actor.parameters(), private_parameters=ref.parameters())
                    refopt.step(); refopt.zero_grad(set_to_none=True)
                shared_optimizer.step(); reference_shared.step()
                generation = store.commit(shared_state={'model': actor.state_dict(), 'optimizer': shared_optimizer.state_dict(), 'rng': torch.get_rng_state()},
                    sampler_state={'next_global_batch': generation}, private_updates=versions, expected_generation=generation)
                if start == 16:
                    store.apply_gc(store.gc_plan(at_global_barrier=True), at_global_barrier=True)
                for p, q in zip(actor.parameters(), reference_actor.parameters()):
                    torch.testing.assert_close(p, q, atol=0, rtol=0)
        assert slot.storage_signature() == addresses
        for index in range(32):
            assert store.stage(identities[index], slot, optimizer) == 2
            for p, q in zip(slot.private.parameters(), reference_private[index].parameters()):
                torch.testing.assert_close(p, q, atol=0, rtol=0)
            for state, other in zip(optimizer.state.values(), reference_adam[index].state.values()):
                for field in ('step', 'exp_avg', 'exp_avg_sq'):
                    torch.testing.assert_close(state[field], other[field], atol=0, rtol=0)


def test_interrupted_gc_keeps_current_and_previous_transactions_recoverable(tmp_path, monkeypatch):
    from pathlib import Path
    ctx = context()
    with PrivateProbeStore(tmp_path) as store:
        for i in range(1, 4):
            step(ctx); commit(store, ctx, i, i-1)
        plan = store.gc_plan(at_global_barrier=True)
        assert len(plan['candidates']) >= 3
        original = Path.unlink
        count = 0
        def interrupted(path, *args, **kwargs):
            nonlocal count
            count += 1
            if count == 2: raise OSError('simulated collection interruption')
            return original(path, *args, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(Path, 'unlink', interrupted)
            with pytest.raises(OSError, match='collection interruption'):
                store.apply_gc(plan, at_global_barrier=True)
        assert store.restore()['generation'] == 3
        assert store.stage(ctx[-1], ctx[3], ctx[4]) == 3
        assert store.restore(manifest_ref=store.manifest()['previous_manifest'])['generation'] == 2
        assert store.stage(ctx[-1], ctx[3], ctx[4]) == 2


def test_cannot_commit_historical_shared_state_over_newer_private_generation(tmp_path):
    ctx = context(); step(ctx)
    with PrivateProbeStore(tmp_path) as store:
        commit(store, ctx, 1, 0); old = store.retain('historical')
        step(ctx); commit(store, ctx, 2, 1)
        restored = store.restore(manifest_ref=old)
        with pytest.raises(ValueError, match='restore CURRENT'):
            store.commit(shared_state=restored['shared'], sampler_state=restored['sampler'],
                         private_updates=[], expected_generation=2)
        assert store.restore()['generation'] == 2
