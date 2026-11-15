"""Two CPU Gloo ranks: private ownership, numerical updates and crash publication."""
import copy
import datetime
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rewriter.coordinate_chisel import CheckpointProbes
from rewriter.coordinate_optimization import joint_state_backward
from rewriter.distributed_probe_store import DistributedProbeStore, private_owner, sum_shared_gradients
from rewriter.private_probe_store import PrivateIdentity
from rewriter.private_probe_slot import PrivateProbeSlot
from tests.test_coordinate_private_invariants import setup, objective


def initialize(rank, world, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://'+str(rendezvous), rank=rank,
                            world_size=world, timeout=datetime.timedelta(seconds=90))


def train_local(store, actor, shared_opt, slot, private_opt, snapshot, identities, batch):
    shared_opt.zero_grad(set_to_none=True)
    versions = []
    for index in batch:
        identity = identities[index]
        if store.owner(identity) != store.rank: continue
        visits = store.stage(identity, slot, private_opt)
        joint_state_backward(objective(actor, slot.private, snapshot), state_weight=1/len(batch),
                             shared_parameters=actor.parameters(), private_parameters=slot.private.parameters())
        private_opt.step(); private_opt.zero_grad(set_to_none=True)
        versions.append(store.snapshot_slot(identity, slot, private_opt, visits=visits+1))
    sum_shared_gradients(actor.parameters())
    shared_opt.step()
    return versions


def train_resident(reference, shared_opt, private, optimizers, snapshot, identities, batch):
    # Match the two rank-local accumulation orders, then SUM exactly once.
    totals = None
    for owner in range(2):
        shared_opt.zero_grad(set_to_none=True)
        for index in batch:
            if private_owner(identities[index], 2) != owner: continue
            module, opt = private[index], optimizers[index]
            joint_state_backward(objective(reference, module, snapshot), state_weight=1/len(batch),
                                 shared_parameters=reference.parameters(), private_parameters=module.parameters())
            opt.step(); opt.zero_grad(set_to_none=True)
        grads = [None if p.grad is None else p.grad.clone() for p in reference.parameters()]
        if totals is None: totals = grads
        else:
            totals = [b if a is None else a if b is None else a+b for a,b in zip(totals,grads)]
    for p, value in zip(reference.parameters(), totals): p.grad = value
    shared_opt.step()


def worker(rank, directory, rendezvous):
    initialize(rank, 2, rendezvous)
    try:
        actor, private, snapshot = setup()
        slot = PrivateProbeSlot(actor, private)
        private_opt = torch.optim.Adam(private.parameters(), lr=1e-3, betas=(.9,.99))
        shared_opt = torch.optim.Adam(actor.parameters(), lr=1e-4)
        modules, opts, identities = [], [], []
        for i in range(32):
            module = CheckpointProbes(snapshot, model_dim=128, latent_tokens=2, seed=71+i,
                                      state_key=('gpt', i//16, i%16+1))
            modules.append(module); opts.append(torch.optim.Adam(module.parameters(), lr=1e-3, betas=(.9,.99)))
            identities.append(PrivateIdentity.from_private(module, state_sha256=f'{i+1:064x}'))
        reference = copy.deepcopy(actor)
        reference_opt = torch.optim.Adam(reference.parameters(), lr=1e-4)
        torch.manual_seed(970+rank)
        with DistributedProbeStore(directory, run_identity='cpu-32-state-qualification') as store:
            versions = []
            for i, identity in enumerate(identities):
                if store.owner(identity) != rank: continue
                slot.stage(modules[i].state_dict(), state_key=identity.state_key)
                private_opt.load_state_dict(opts[i].state_dict())
                versions.append(store.snapshot_slot(identity, slot, private_opt, visits=0))
            def save(batch, updates):
                return store.commit(shared_state={'model': actor.state_dict(), 'optimizer': shared_opt.state_dict()},
                    rng_state={'torch': torch.get_rng_state(), 'rank_marker': rank},
                    sampler_state={'next_global_batch': batch, 'rank_cursor': rank*1000+batch}, private_updates=updates)
            assert save(0, versions) == 1
            assert not (store.shard.directory/'CURRENT.json').exists()
            signature = slot.storage_signature()
            other = next(i for i in identities if store.owner(i) != rank)
            with pytest.raises(ValueError, match='different identity owner'):
                store.stage(other, slot, private_opt)
            for global_batch in range(8):
                start = (global_batch % 4)*8
                batch = list(range(start, start+8))
                if global_batch == 1:
                    failed = train_local(store, actor, shared_opt, slot, private_opt, snapshot, identities, batch)
                    original = store._prepare_index
                    if rank == 1:
                        def fail(updates): raise OSError('simulated rank publication interruption')
                        store._prepare_index = fail
                    with pytest.raises(RuntimeError, match='transaction aborted'):
                        save(global_batch+1, failed)
                    store._prepare_index = original
                    recovered = store.restore()
                    assert recovered['generation'] == 2
                    assert recovered['sampler'] == {'next_global_batch': 1, 'rank_cursor': rank*1000+1}
                    assert recovered['rng']['rank_marker'] == rank
                    actor.load_state_dict(recovered['shared']['model'])
                    shared_opt.load_state_dict(recovered['shared']['optimizer'])
                    torch.set_rng_state(recovered['rng']['torch'])
                    for version in failed:
                        assert (store.shard.directory/version['file']).exists()
                        assert version['file'] not in {v['file'] for v in store._versions.values()}
                updates = train_local(store, actor, shared_opt, slot, private_opt, snapshot, identities, batch)
                assert save(global_batch+1, updates) == global_batch+2
                if rank == 0:
                    train_resident(reference, reference_opt, modules, opts, snapshot, identities, batch)
                    for p, q in zip(actor.parameters(), reference.parameters()):
                        torch.testing.assert_close(p, q, atol=3e-7, rtol=3e-6)
                    for state, expected in zip(shared_opt.state.values(), reference_opt.state.values()):
                        for key in ('step','exp_avg','exp_avg_sq'):
                            torch.testing.assert_close(state[key], expected[key], atol=3e-6, rtol=3e-5)
                # Reference private modules on rank1 must also follow the same
                # shared trajectory for exact local state comparisons.
                if rank == 1:
                    train_resident(reference, reference_opt, modules, opts, snapshot, identities, batch)
            assert slot.storage_signature() == signature
            for i, identity in enumerate(identities):
                if store.owner(identity) != rank: continue
                assert store.stage(identity, slot, private_opt) == 2
                for p, q in zip(slot.private.parameters(), modules[i].parameters()):
                    torch.testing.assert_close(p, q, atol=3e-7, rtol=3e-6)
                for state, expected in zip(private_opt.state.values(), opts[i].state.values()):
                    for key in ('step','exp_avg','exp_avg_sq'):
                        torch.testing.assert_close(state[key], expected[key], atol=3e-6, rtol=3e-5)
            restored = store.restore()
            torch.testing.assert_close(restored['rng']['torch'], torch.get_rng_state(), atol=0, rtol=0)
            assert restored['sampler']['rank_cursor'] == rank*1000+8
            # A mismatched shared optimizer/model cannot publish a new generation.
            if rank == 1:
                with torch.no_grad(): next(actor.parameters()).add_(.01)
            with pytest.raises(ValueError, match='differ across ranks'): save(9, [])
            recovered = store.restore()
            assert recovered['generation'] == 9
            actor.load_state_dict(recovered['shared']['model'])
            shared_opt.load_state_dict(recovered['shared']['optimizer'])
            torch.set_rng_state(recovered['rng']['torch'])
            parameter = torch.nn.Parameter(torch.ones(3)); unused = torch.nn.Parameter(torch.zeros(2))
            parameter.grad = torch.ones(3) if rank == 0 else None
            sum_shared_gradients([parameter, unused])
            torch.testing.assert_close(parameter.grad, torch.ones(3), atol=0, rtol=0)
            assert unused.grad is None
            # Leave a ninth update UNCOMMITTED, then a fresh process group must
            # reproduce it from generation9 rather than load these pending files.
            pending = train_local(store, actor, shared_opt, slot, private_opt, snapshot, identities, list(range(8)))
            torch.save(dict(model=actor.state_dict(), optimizer=shared_opt.state_dict(),
                private={v['token']:store.shard.read_private(identities[next(i for i,x in enumerate(identities) if x.token==v['token'])],version=v) for v in pending}),
                Path(directory)/f'rank{rank}-expected-next.pt')
            (Path(directory)/f'rank{rank}-passed').write_text('32 identities;8 joint updates;failed publication recovered\n')
    finally:
        dist.destroy_process_group()


def assert_tree_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual: assert_tree_equal(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for a,b in zip(actual,expected): assert_tree_equal(a,b)
    else: assert actual == expected


def resume_worker(rank, directory, rendezvous):
    initialize(rank, 2, rendezvous)
    try:
        actor, private, snapshot = setup()
        slot = PrivateProbeSlot(actor, private)
        private_opt = torch.optim.Adam(private.parameters(), lr=1e-3, betas=(.9,.99))
        shared_opt = torch.optim.Adam(actor.parameters(), lr=1e-4)
        identities = []
        for i in range(32):
            module = CheckpointProbes(snapshot, model_dim=128, latent_tokens=2, seed=71+i,
                                      state_key=('gpt', i//16, i%16+1))
            identities.append(PrivateIdentity.from_private(module, state_sha256=f'{i+1:064x}'))
        with DistributedProbeStore(directory, run_identity='cpu-32-state-qualification') as store:
            restored = store.restore()
            assert restored['generation'] == 9
            assert restored['sampler'] == {'next_global_batch':8,'rank_cursor':rank*1000+8}
            actor.load_state_dict(restored['shared']['model'])
            shared_opt.load_state_dict(restored['shared']['optimizer'])
            torch.set_rng_state(restored['rng']['torch'])
            updates = train_local(store, actor, shared_opt, slot, private_opt, snapshot, identities, list(range(8)))
            expected = torch.load(Path(directory)/f'rank{rank}-expected-next.pt',weights_only=False)
            assert_tree_equal(actor.state_dict(), expected['model'])
            assert_tree_equal(shared_opt.state_dict(), expected['optimizer'])
            by_token = {identity.token:identity for identity in identities}
            for version in updates:
                payload = store.shard.read_private(by_token[version['token']],version=version)
                assert_tree_equal(payload, expected['private'][version['token']])
            assert store.commit(shared_state={'model':actor.state_dict(),'optimizer':shared_opt.state_dict()},
                rng_state=restored['rng'],sampler_state={'next_global_batch':9,'rank_cursor':rank*1000+9},
                private_updates=updates) == 10
    finally: dist.destroy_process_group()


def test_two_rank_staging_updates_and_atomic_resume(tmp_path):
    root = tmp_path/'store'
    mp.spawn(worker, args=(root, tmp_path/'rendezvous'), nprocs=2, join=True)
    assert all((root/f'rank{i}-passed').exists() for i in range(2))
    mp.spawn(resume_worker, args=(root,tmp_path/'resume-rendezvous'), nprocs=2, join=True)
    # A changed process count cannot silently reinterpret private ownership.
    initialize(0, 1, tmp_path/'world1-rendezvous')
    try:
        with DistributedProbeStore(root, run_identity='cpu-32-state-qualification') as store:
            with pytest.raises(RuntimeError, match='resharding required'):
                store.restore()
    finally: dist.destroy_process_group()


def test_owner_depends_on_identity_not_minibatch_ordinal():
    _, private, _ = setup()
    identities = [PrivateIdentity(('gpt', i//8, i%8+1), f'{i+1:064x}',
                  PrivateIdentity.from_private(private,state_sha256='a'*64).layout_sha256) for i in range(16)]
    expected = {identity.token: private_owner(identity, 2) for identity in identities}
    for permutation in (identities[::-1], identities[5:]+identities[:5]):
        assert {identity.token: private_owner(identity, 2) for identity in permutation} == expected
    assert set(expected.values()) == {0, 1}


def test_snapshot_backend_is_explicit_private_shard_only(tmp_path):
    initialize(0,1,tmp_path/'rendezvous')
    try:
        with DistributedProbeStore(tmp_path/'default',run_identity='default') as store:
            assert store.global_store.snapshot_backend==store.shard.snapshot_backend=='reference'
        with DistributedProbeStore(tmp_path/'pinned',run_identity='pinned',snapshot_backend='pinned_d2h_v1') as store:
            assert store.global_store.snapshot_backend=='reference'
            assert store.shard.snapshot_backend=='pinned_d2h_v1'
        with pytest.raises(ValueError,match='snapshot backend'):
            DistributedProbeStore(tmp_path/'unknown',run_identity='unknown',snapshot_backend='unknown')
    finally:dist.destroy_process_group()
