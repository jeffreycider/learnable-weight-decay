import os
os.environ.setdefault('PROFILE_CHISEL_SKIP_SOURCE_GUARD', '1')
import pytest
import torch
from rewriter.static_snapshot_store import StaticSnapshotStore, layout_signature, _leaves


def _records(tmp_path):
    from tests.chisel_fixtures import tiny_bank_and_cache
    from rewriter.state_bank import StateBankReader
    bank, _ = tiny_bank_and_cache(tmp_path)
    reader = StateBankReader(bank, verify_once=True, mmap_model=True)
    return [reader.read(s, device='cpu', model_device='cpu') for s in (1, 2, 3)]


def test_store_keeps_strides_and_addresses_and_matches_fresh_values(tmp_path):
    r1, r2, r3 = _records(tmp_path)
    store = StaticSnapshotStore(r1.snapshot, r1.train, device='cpu')
    addresses = store.addresses()
    for record in (r2, r3, r1):
        store.refill(record.snapshot, record.train)
        assert store.bound and store.addresses() == addresses
        for name, fresh in _leaves(record.snapshot, record.train):
            held = store.tensor(name)
            assert torch.equal(held, fresh), name
            assert held.stride() == fresh.stride() and held.storage_offset() == 0, name


def test_store_handles_stride_zero_broadcast_leaves(tmp_path):
    import dataclasses
    r1, r2, _ = _records(tmp_path)
    width = r1.snapshot.trajectories.shape[1]
    def with_broadcast(record, seed):
        g = torch.Generator().manual_seed(seed)
        row = torch.randn(1, width, generator=g)
        return dataclasses.replace(record.snapshot, trajectories=row.expand(4, width))  # stride (0, 1)
    s1, s2 = with_broadcast(r1, 1), with_broadcast(r2, 2)
    store = StaticSnapshotStore(s1, r1.train, device='cpu')
    held = store.tensor('trajectories')
    assert held.stride() == (0, width) or held.stride() == s1.trajectories.stride()
    for snapshot in (s2, s1):
        store.refill(snapshot, r1.train)
        assert torch.equal(store.tensor('trajectories'), snapshot.trajectories)
        assert store.tensor('trajectories').stride() == snapshot.trajectories.stride()
        assert store.tensor('trajectories').untyped_storage().nbytes() == width * 4  # only the unique row is stored


def test_store_views_feed_the_actor_bit_identically_and_reject_layout_changes(tmp_path):
    from rewriter.chisel_factory import build_chisel
    r1, r2, _ = _records(tmp_path)
    arch = dict(name='nanogpt_blocks_patch_io_v1', rank=2, trajectory_dim=r1.snapshot.trajectories.shape[1],
                model_dim=128, num_layers=1, patch_size=8)
    torch.manual_seed(1)
    actor = build_chisel(arch)
    with torch.no_grad():
        for head in actor.output_heads:
            head.weight.normal_(0, 1e-3)
    store = StaticSnapshotStore(r1.snapshot, r1.train, device='cpu')
    static_snapshot, _ = store.snapshot(r1.snapshot, r1.train)
    def flat(action):
        return torch.cat([t.flatten() for f in action.matrices.values() for t in (f.P, f.Q)] + [v.flatten() for v in action.vectors.values()])
    for record in (r1, r2, r1):
        store.refill(record.snapshot, record.train)
        with torch.no_grad():
            assert torch.equal(flat(actor(static_snapshot)), flat(actor(record.snapshot)))
    # stride change (transposed anchor) is rejected only when it actually changes the signature
    import dataclasses
    bad_name = next(iter(r2.snapshot.matrices))
    obs = r2.snapshot.matrices[bad_name]
    bad_obs = dataclasses.replace(obs, anchors=dataclasses.replace(obs.anchors, A=obs.anchors.A.contiguous().T.contiguous().T))
    bad = dataclasses.replace(r2.snapshot, matrices=dict(r2.snapshot.matrices, **{bad_name: bad_obs}))
    if layout_signature(bad, r2.train) != store.signature:
        with pytest.raises(ValueError, match='layout'):
            store.refill(bad, r2.train)
        assert not store.bound
    store.refill(r2.snapshot, r2.train)
    assert store.bound


def test_rejected_refill_unbinds_even_when_no_copy_happened(tmp_path):
    """Astra cycle-8 reproduction: a shape change must leave bound=False although the
    layout check fails BEFORE any buffer is touched (previous values remain, and must
    not be replayable); a later matching refill re-binds."""
    import dataclasses
    r1, r2, _ = _records(tmp_path)
    store = StaticSnapshotStore(r1.snapshot, r1.train, device='cpu')
    store.refill(r2.snapshot, r2.train)
    assert store.bound
    generation = store.generation
    rows, width = r2.snapshot.trajectories.shape
    grown = dataclasses.replace(r2.snapshot, trajectories=torch.randn(rows + 1, width))   # unconditional shape mismatch
    assert layout_signature(grown, r2.train) != store.signature
    with pytest.raises(ValueError, match='layout'):
        store.refill(grown, r2.train)
    assert not store.bound                                  # the bug: this was True
    assert store.generation == generation                   # nothing was copied
    assert torch.equal(store.tensor('trajectories'), r2.snapshot.trajectories)   # stale values still there, hence unbound matters
    # dtype mismatch at equal shape is also a layout change
    retyped = dataclasses.replace(r2.snapshot, radius=r2.snapshot.radius.double())
    with pytest.raises(ValueError, match='layout'):
        store.refill(retyped, r2.train)
    assert not store.bound
    # successful rebind recovery
    assert store.refill(r1.snapshot, r1.train) == generation + 1
    assert store.bound and torch.equal(store.tensor('trajectories'), r1.snapshot.trajectories)
