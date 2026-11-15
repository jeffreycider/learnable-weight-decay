"""Information and gradient checks for the explicit write-frame alternative."""
import copy
import pytest
import torch
from rewriter.coordinate_chisel import CoordinateChisel, WRITE_FRAME_ARCHITECTURE
from tests.test_coordinate_private_invariants import (
    setup, permutations, permuted_snapshot, permute_private, objective,
    assert_private_permutation,
)


def variant():
    torch.set_num_threads(1)
    old, private, snapshot = setup()
    new = CoordinateChisel(rank=2, trajectory_dim=12, model_dim=128,
                           num_layers=1, latent_tokens=2,
                           write_observation_frame='decoder')
    new.load_state_dict(old.state_dict())
    return old, new, private, snapshot


def test_explicit_variant_keeps_parameter_and_private_layout_but_changes_semantics():
    old, new, private, snapshot = variant()
    assert old.configuration().get('write_observation_frame', 'encoder') == 'encoder'
    assert new.configuration()['name'] == WRITE_FRAME_ARCHITECTURE
    assert new.configuration()['write_observation_frame'] == 'decoder'
    assert list(old.state_dict()) == list(new.state_dict())
    assert sum(p.numel() for p in old.parameters()) == sum(p.numel() for p in new.parameters())
    for p, q in zip(old.parameters(), new.parameters()):
        torch.testing.assert_close(p, q, atol=0, rtol=0)
    before = copy.deepcopy(private.get_extra_state())
    new(snapshot, private)
    assert private.get_extra_state() == before
    with pytest.raises(ValueError, match='write observation frame'):
        CoordinateChisel(write_observation_frame='unknown')


def test_changing_decoder_is_hidden_from_old_tokens_but_visible_before_all_new_writes():
    old, new, private, snapshot = variant()
    other = copy.deepcopy(private)
    with torch.no_grad():
        other.matrix_columns[-1].decoder.add_(torch.randn_like(other.matrix_columns[-1].decoder))
    a, old_reads, _ = old._inputs(snapshot, private)
    b, _, _ = old._inputs(snapshot, other)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    c, new_reads, _ = new._inputs(snapshot, private)
    d, _, _ = new._inputs(snapshot, other)
    counts = new.token_counts(snapshot)
    assert new_reads == counts['observation_tokens']
    assert new_reads == old_reads + counts['write_tokens']
    assert c.shape[1] == counts['total_tokens']
    # All original E observations and trajectories stay unchanged.
    torch.testing.assert_close(c[:, :old_reads], d[:, :old_reads], atol=0, rtol=0)
    assert (c[:, old_reads:new_reads] - d[:, old_reads:new_reads]).abs().max() > 1e-4
    # The new D field is in the prefix, visible to even the FIRST final write.
    assert (c[:, new_reads:] - d[:, new_reads:]).abs().max() > 1e-4


def test_later_write_frame_can_change_an_earlier_matrix_decision():
    old, new, private, snapshot = variant()
    other = copy.deepcopy(private)
    with torch.no_grad():
        other.matrix_columns[-1].decoder.add_(torch.randn_like(other.matrix_columns[-1].decoder))
    first = next(iter(snapshot.matrices))
    torch.testing.assert_close(old(snapshot, private).matrices[first].dense(),
                               old(snapshot, other).matrices[first].dense(), atol=0, rtol=0)
    a = new(snapshot, private).matrices[first].dense()
    b = new(snapshot, other).matrices[first].dense()
    assert (a-b).abs().max() > 1e-7


def test_write_fields_use_normalized_decoder_and_preserve_constant_metadata():
    _, actor, private, snapshot = variant()
    frame = private.matrix_rows[0]
    obs = next(iter(snapshot.matrices.values()))
    features = obs.rows.clone()
    features[:, 11:] = torch.arange(features.shape[1]-11, dtype=features.dtype)
    with torch.no_grad(): frame.decoder.normal_()
    encoded = frame.encode_with_constants(features, slice(0, 11), source='decoder')
    basis = torch.nn.functional.normalize(frame.decoder, dim=0)
    torch.testing.assert_close(encoded[:, :11], basis.T @ features[:, :11])
    torch.testing.assert_close(encoded[:, 11:], features[:1, 11:].expand(len(encoded), -1))
    weights = torch.randn_like(encoded)
    grad = torch.autograd.grad((encoded*weights).sum(), frame.decoder)[0]
    assert grad.norm() > 0 and torch.isfinite(grad).all()
    # Unit-column normalization annihilates the radial derivative.
    torch.testing.assert_close((grad*frame.decoder).sum(0), torch.zeros(grad.shape[1]), atol=2e-6, rtol=0)


def test_nonzero_action_and_full_vjp_follow_consistent_coordinate_relabeling():
    _, actor, private, snapshot = variant()
    other_actor, other_private = copy.deepcopy(actor), copy.deepcopy(private)
    perms = permutations(snapshot)
    permute_private(other_private, perms)
    changed = permuted_snapshot(snapshot, perms)
    torch.testing.assert_close(actor._inputs(snapshot, private)[0],
                               other_actor._inputs(changed, other_private)[0], atol=3e-6, rtol=3e-5)
    left, right = actor(snapshot, private), other_actor(changed, other_private)
    assert left.norm_sq() > 1e-5
    for name, factors in left.matrices.items():
        row, col = perms[0][name]
        torch.testing.assert_close(right.matrices[name].dense(), factors.dense()[row][:, col], atol=5e-6, rtol=5e-5)
    objective(actor, private, snapshot).backward()
    objective(other_actor, other_private, changed).backward()
    assert_private_permutation(private, other_private, perms, gradients=True)
    for (_, p), (_, q) in zip(actor.named_parameters(), other_actor.named_parameters()):
        if p.grad is not None: torch.testing.assert_close(p.grad, q.grad, atol=3e-5, rtol=3e-4)


def test_prepared_indices_do_not_change_variant_values_or_gradients():
    _, actor, private, snapshot = variant()
    other_actor, other_private = copy.deepcopy(actor), copy.deepcopy(private)
    other_actor.prepare_static_indices(snapshot)
    torch.testing.assert_close(actor._inputs(snapshot, private)[0],
                               other_actor._inputs(snapshot, other_private)[0], atol=0, rtol=0)
    objective(actor, private, snapshot).backward()
    objective(other_actor, other_private, snapshot).backward()
    for p, q in zip(private.parameters(), other_private.parameters()):
        if p.grad is not None: torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
