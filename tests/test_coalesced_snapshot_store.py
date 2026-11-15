import dataclasses
import pytest
import torch
from rewriter.coalesced_snapshot_store import CoalescedSnapshotStore, stage_coalesced_snapshot, BankedObservationReader
from rewriter.static_snapshot_store import _leaves
from tests.test_prefix_cache import toy_record


def test_banked_store_exact_values_strides_masks_and_refill_addresses():
    snapshot, train = toy_record()
    snapshot = dataclasses.replace(snapshot, trajectory_mask=torch.tensor([True, False, True]))
    staged = stage_coalesced_snapshot(snapshot, train)
    store = CoalescedSnapshotStore(staged, device='cpu')
    held, measurements = store.snapshot(snapshot, train)
    assert len(store.banks) == 2
    for (name, original), (other, value) in zip(_leaves(snapshot, train), _leaves(held, measurements), strict=True):
        assert name == other and original.stride() == value.stride()
        assert torch.equal(original, value)
        assert staged.offsets[name] * value.element_size() % 256 == 0
    addresses = store.addresses()
    changed = dataclasses.replace(snapshot, trajectories=snapshot.trajectories + 2, trajectory_mask=~snapshot.trajectory_mask)
    next_staged = stage_coalesced_snapshot(changed, train)
    store.refill_staged(next_staged)
    assert addresses == store.addresses()
    assert torch.equal(held.trajectories, changed.trajectories)
    assert torch.equal(held.trajectory_mask, changed.trajectory_mask)
    # The CPU fixture is also a genuine copy, not an alias of the staged bank.
    next_staged.banks[torch.float32].zero_()
    assert torch.equal(held.trajectories, changed.trajectories)


def test_invalid_mask_layout_fails_before_copy_and_unbinds():
    snapshot, train = toy_record()
    store = CoalescedSnapshotStore(stage_coalesced_snapshot(snapshot, train), device='cpu')
    before = {key:value.clone() for key,value in store.banks.items()}
    changed = dataclasses.replace(snapshot, trajectory_mask=torch.tensor([True, False, True]))
    with pytest.raises(ValueError, match='layout'):
        store.refill_staged(stage_coalesced_snapshot(changed, train))
    assert not store.bound
    assert all(torch.equal(value, store.banks[key]) for key,value in before.items())


def test_corrupt_bank_is_rejected_before_any_bank_copy():
    snapshot, train = toy_record()
    staged = stage_coalesced_snapshot(snapshot, train)
    store = CoalescedSnapshotStore(staged, device='cpu')
    corrupt = dataclasses.replace(staged, banks={torch.float32:torch.zeros(1)})
    with pytest.raises(ValueError, match='shape'):
        store.refill_staged(corrupt)
    assert not store.bound


def test_reader_cpu_default_is_unchanged_and_gpu_model_request_fails_closed():
    class Reader:
        manifest = {'split': 'unchanged'}
        def read(self, state_id, **kwargs): return (state_id, kwargs)
    adapter = BankedObservationReader(Reader())
    assert adapter.manifest == {'split': 'unchanged'}
    assert adapter.read(999) == (999, {'device':torch.device('cpu'), 'model_device':None})
    with pytest.raises(ValueError, match='CPU model weights'):
        adapter.read(999, device='cuda', model_device='cuda')


def test_depth_one_prefetch_preserves_namespace_and_failure_is_not_swallowed():
    from rewriter.state_bank import StateRecord
    snapshot, train = toy_record()
    record = StateRecord(6, {}, snapshot, train, 6, None, training_state_id=1006, bank_seed=1)
    class Reader:
        def read(self, state_id, **kwargs):
            if state_id != 1006: raise ValueError('bad immutable record')
            return record
    adapter = BankedObservationReader(Reader())
    try:
        adapter.prefetch(1006, pin=False)
        with pytest.raises(RuntimeError, match='pending'):
            adapter.prefetch(7, pin=False)
        with pytest.raises(ValueError, match='differs'):
            adapter._take(6, False)
        actual, staged = adapter._take(1006, False)
        assert actual is record and actual.state_id == 1006 and actual.completed_steps == 6
        assert staged.signature
        adapter.prefetch(7, pin=False)
        with pytest.raises(ValueError, match='bad immutable record'):
            adapter._take(7, False)
    finally:
        adapter.close()
    with pytest.raises(RuntimeError, match='closed'):
        adapter.read(1006)
