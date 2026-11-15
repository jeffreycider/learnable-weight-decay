"""Independent representation/Adam invariants, not a distributed-store qualification."""
import copy
import dataclasses
import io

import pytest
import torch

from rewriter.algebra import Anchors
from rewriter.coordinate_chisel import CoordinateChisel, CheckpointProbes
from tests.test_controller import make_snapshot


@pytest.fixture(autouse=True)
def few_cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def setup():
    torch.manual_seed(9103)
    snapshot = make_snapshot()
    actor = CoordinateChisel(rank=2, trajectory_dim=12, model_dim=128,
                             num_layers=1, latent_tokens=2)
    private = CheckpointProbes(snapshot, model_dim=128, latent_tokens=2,
                               seed=63, state_key=('gpt', 0, 1))
    # A zero output head would make the edit-equivariance check vacuous.
    with torch.no_grad():
        for head in (actor.matrix_output, actor.vector_output):
            head.weight.normal_(std=.02)
        for block in actor.blocks:
            block.attn.proj.weight.normal_(std=.01)
            block.mlp.proj.weight.normal_(std=.01)
    return actor, private, snapshot


def permutations(snapshot):
    gen = torch.Generator().manual_seed(918)
    matrix = {name: (torch.randperm(obs.rows.shape[0], generator=gen),
                     torch.randperm(obs.columns.shape[0], generator=gen))
              for name, obs in snapshot.matrices.items()}
    vector = {name: torch.randperm(obs.weight.numel(), generator=gen)
              for name, obs in snapshot.vectors.items()}
    return matrix, vector


def permuted_snapshot(snapshot, perms):
    matrix, vector = perms
    matrices = {}
    for name, obs in snapshot.matrices.items():
        row, col = matrix[name]
        matrices[name] = dataclasses.replace(obs, rows=obs.rows[row], columns=obs.columns[col],
            anchors=Anchors(obs.anchors.A[row], obs.anchors.B[col]),
            weight_probes=(obs.weight_probes[0][row], obs.weight_probes[1][col]))
    vectors = {name: dataclasses.replace(obs, coordinates=obs.coordinates[vector[name]],
                                         weight=obs.weight[vector[name]])
               for name, obs in snapshot.vectors.items()}
    return dataclasses.replace(snapshot, matrices=matrices, vectors=vectors)


def frame_permutations(private, perms):
    matrix, vector = perms
    result = {}
    for i, name in enumerate(private.matrix_names):
        for prefix, frames, perm in [('matrix_rows', private.matrix_rows, matrix[name][0]),
                                     ('matrix_columns', private.matrix_columns, matrix[name][1])]:
            for kind in ('encoder', 'decoder'):
                result[f'{prefix}.{i}.{kind}'] = (getattr(frames[i], kind), perm)
    for i, name in enumerate(private.vector_names):
        for kind in ('encoder', 'decoder'):
            result[f'vector_frames.{i}.{kind}'] = (getattr(private.vector_frames[i], kind), vector[name])
    return result


def permute_private(private, perms, optimizer=None):
    with torch.no_grad():
        for parameter, perm in frame_permutations(private, perms).values():
            parameter.copy_(parameter[perm])
            if optimizer is not None:
                for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                    value = optimizer.state[parameter].get(key)
                    if value is not None:
                        value.copy_(value[perm])


def objective(actor, private, snapshot):
    action = actor(snapshot, private)
    return action.contraction(snapshot.weights, snapshot.vector_weights) + .1 * action.norm_sq()


def assert_private_permutation(left, right, perms, *, gradients=False):
    mapping = frame_permutations(left, perms)
    right_named = dict(right.named_parameters())
    for name, value in left.named_parameters():
        other = right_named[name]
        if gradients:
            value, other = value.grad, other.grad
            assert (value is None) == (other is None), name
            if value is None:
                continue
        expected = value[mapping[name][1]] if name in mapping else value
        torch.testing.assert_close(other, expected, atol=3e-5, rtol=3e-4, msg=name)


def test_coordinate_permutation_preserves_tokens_and_transforms_nonzero_edit_and_vjp():
    actor, private, snapshot = setup()
    other_actor, other_private = copy.deepcopy(actor), copy.deepcopy(private)
    perms = permutations(snapshot)
    changed = permuted_snapshot(snapshot, perms)
    permute_private(other_private, perms)
    original_tokens = actor._inputs(snapshot, private)[0]
    changed_tokens = other_actor._inputs(changed, other_private)[0]
    torch.testing.assert_close(changed_tokens, original_tokens, atol=3e-6, rtol=3e-5)
    action, other = actor(snapshot, private), other_actor(changed, other_private)
    assert action.norm_sq() > 1e-5
    for name, factor in action.matrices.items():
        row, col = perms[0][name]
        torch.testing.assert_close(other.matrices[name].dense(), factor.dense()[row][:, col], atol=5e-6, rtol=5e-5)
    for name, value in action.vectors.items():
        torch.testing.assert_close(other.vectors[name], value[perms[1][name]], atol=5e-6, rtol=5e-5)
    # Both sides include fixed-anchor canonicalization, not only E^T H.
    loss = objective(actor, private, snapshot)
    other_loss = objective(other_actor, other_private, changed)
    torch.testing.assert_close(other_loss, loss, atol=1e-5, rtol=1e-5)
    loss.backward(); other_loss.backward()
    for frame in private.matrix_rows:
        assert frame.encoder.grad.norm() > 0
        assert frame.decoder.grad.norm() > 0
    for (name, p), (_, q) in zip(actor.named_parameters(), other_actor.named_parameters()):
        if p.grad is not None:
            torch.testing.assert_close(q.grad, p.grad, atol=3e-5, rtol=3e-4, msg=name)
    assert_private_permutation(private, other_private, perms, gradients=True)
    # Negative control: changing feature coordinates alone changes the read tokens.
    unmatched_tokens = actor._inputs(changed, private)[0]
    assert (unmatched_tokens - original_tokens).abs().max() > 1e-3


def test_adam_moments_must_follow_coordinate_relabeling():
    actor, private, snapshot = setup()
    opt = torch.optim.Adam(private.parameters(), lr=1e-3, betas=(.9, .99))
    objective(actor, private, snapshot).backward(); opt.step(); opt.zero_grad(set_to_none=True)
    restored = copy.deepcopy(private)
    other_opt = torch.optim.Adam(restored.parameters(), lr=1e-3, betas=(.9, .99))
    other_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    perms = permutations(snapshot)
    permute_private(restored, perms, other_opt)
    objective(actor, private, snapshot).backward()
    objective(actor, restored, permuted_snapshot(snapshot, perms)).backward()
    opt.step(); other_opt.step()
    assert_private_permutation(private, restored, perms)
    for name, (p, perm) in frame_permutations(private, perms).items():
        q = dict(restored.named_parameters())[name]
        for key in ('exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(other_opt.state[q][key], opt.state[p][key][perm], atol=3e-6, rtol=3e-4)
        assert other_opt.state[q]['step'] == opt.state[p]['step'] == 2


def test_private_adam_locality_and_exact_save_restore_next_step():
    actor, private, snapshot = setup()
    untouched = CheckpointProbes(snapshot, model_dim=128, latent_tokens=2,
                                seed=63, state_key=('gpt', 1, 1))
    # Same local checkpoint across seeds is a distinct parameter/optimizer owner.
    untouched_before = [p.detach().clone() for p in untouched.parameters()]
    opt = torch.optim.Adam(private.parameters(), lr=1e-3)
    unvisited_opt = torch.optim.Adam(untouched.parameters(), lr=1e-3)
    assert not ({id(p) for p in actor.parameters()} & {id(p) for p in private.parameters()})
    objective(actor, private, snapshot).backward(); opt.step(); opt.zero_grad(set_to_none=True)
    for before, p in zip(untouched_before, untouched.parameters()):
        torch.testing.assert_close(p, before, atol=0, rtol=0)
    assert not unvisited_opt.state
    stream = io.BytesIO()
    torch.save({'private': private.state_dict(), 'adam': opt.state_dict()}, stream)
    stream.seek(0); saved = torch.load(stream, weights_only=False)
    restored = CheckpointProbes(snapshot, model_dim=128, latent_tokens=2,
                               seed=999, state_key=('gpt', 0, 1))
    restored.load_state_dict(saved['private'])
    restored_opt = torch.optim.Adam(restored.parameters(), lr=1e-3)
    restored_opt.load_state_dict(saved['adam'])
    for current, optimizer in ((private, opt), (restored, restored_opt)):
        optimizer.zero_grad(set_to_none=True)
        objective(actor, current, snapshot).backward(); optimizer.step()
    for p, q in zip(private.parameters(), restored.parameters()):
        torch.testing.assert_close(q, p, atol=0, rtol=0)
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(restored_opt.state[q][key], opt.state[p][key], atol=0, rtol=0)
    with pytest.raises((ValueError, RuntimeError), match='identity|key|checkpoint'):
        untouched.load_state_dict(saved['private'])
    for before, p in zip(untouched_before, untouched.parameters()):
        torch.testing.assert_close(p, before, atol=0, rtol=0)
