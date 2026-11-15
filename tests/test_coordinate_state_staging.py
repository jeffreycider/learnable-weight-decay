"""Masks and exact layouts are semantic inputs, not optional staging metadata."""
import dataclasses
import pytest
import torch

from rewriter.coordinate_chisel import CoordinateChisel
from rewriter.state_prefetch import stage_snapshot, restore_snapshot
from rewriter.static_snapshot_store import StaticSnapshotStore, _leaves
from tests.test_prefix_cache import toy_record


@pytest.mark.parametrize('masked', [False, True])
def test_staging_preserves_coordinate_action_and_shared_private_vjp(masked):
    torch.set_num_threads(1)
    torch.manual_seed(818)
    snapshot, train = toy_record()
    if masked:
        # Noncontiguous mask catches accidental contiguous-layout substitution.
        mask = torch.tensor([True, False, False, False, True, False])[::2]
        snapshot = dataclasses.replace(snapshot, trajectory_mask=mask)
    staged, staged_train = restore_snapshot(stage_snapshot(snapshot, train, False), 'cpu', False)
    store = StaticSnapshotStore(snapshot, train, device='cpu')
    held, held_train = store.snapshot(snapshot, train)
    actor = CoordinateChisel(rank=2, trajectory_dim=12, model_dim=128, num_layers=2, latent_tokens=3)
    private = actor.make_private(snapshot, state_key=('gpt', 0, 6), seed=7)
    with torch.no_grad():
        for name, parameter in actor.named_parameters():
            if 'output' in name:
                parameter.normal_(std=.01)
    expected = None
    for current, measurements in [(snapshot, train), (staged, staged_train), (held, held_train)]:
        for (name, original), (other_name, copied) in zip(_leaves(snapshot, train), _leaves(current, measurements), strict=True):
            assert name == other_name and original.stride() == copied.stride()
            assert torch.equal(original, copied)
        actor.zero_grad(set_to_none=True); private.zero_grad(set_to_none=True)
        action = actor(current, private)
        loss = action.contraction(current.weights, current.vector_weights)
        loss.backward()
        gradients = [None if p.grad is None else p.grad.clone() for p in [*actor.parameters(), *private.parameters()]]
        if expected is None:
            expected = (loss.detach().clone(), gradients)
        else:
            assert torch.equal(loss, expected[0])
            for value, reference in zip(gradients, expected[1], strict=True):
                assert (value is None) == (reference is None)
                if value is not None:
                    assert torch.equal(value, reference)


def test_static_mask_refill_keeps_address_and_rejects_absent_mask():
    snapshot, train = toy_record()
    snapshot = dataclasses.replace(snapshot, trajectory_mask=torch.tensor([True, False, True]))
    store = StaticSnapshotStore(snapshot, train, device='cpu')
    held, _ = store.snapshot(snapshot, train)
    address = held.trajectory_mask.data_ptr()
    changed = dataclasses.replace(snapshot, trajectory_mask=~snapshot.trajectory_mask)
    store.refill(changed, train)
    assert held.trajectory_mask.data_ptr() == address
    assert torch.equal(held.trajectory_mask, changed.trajectory_mask)
    with pytest.raises(ValueError, match='layout'):
        store.refill(dataclasses.replace(snapshot, trajectory_mask=None), train)
    assert not store.bound
