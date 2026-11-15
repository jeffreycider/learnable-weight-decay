import dataclasses

import torch

from rewriter.accelerated_controller import PackedGlobalRewriter
from tests.test_controller import make_snapshot, small_controller


def flattened(action):
    return torch.cat([value.flatten() for factor in action.matrices.values()
                      for value in (factor.P, factor.Q)]
                     + [value.flatten() for value in action.vectors.values()])


def test_packed_outputs_all_parameter_gradients_and_changed_snapshot():
    actor = small_controller()
    with torch.no_grad():
        for decoder in (actor.row_decoder, actor.column_decoder, actor.vector_decoder):
            decoder[-1].weight.normal_(std=.03)
    fast = PackedGlobalRewriter(actor)
    snapshot = make_snapshot()
    # Exercise equal-shape batched groups as well as ragged ones.
    snapshot = dataclasses.replace(snapshot, matrices=dict(snapshot.matrices,
        third=snapshot.matrices['first']), vectors=dict(snapshot.vectors,
        other=snapshot.vectors['gain']))
    for offset in (0., .4):
        changed = dataclasses.replace(snapshot, trajectories=snapshot.trajectories + offset)
        a, b = flattened(actor(changed)), flattened(fast(changed))
        torch.testing.assert_close(a, b, atol=3e-7, rtol=3e-5)
        direction = torch.randn_like(a)
        ga = torch.autograd.grad((a * direction).sum(), tuple(actor.parameters()))
        gb = torch.autograd.grad((b * direction).sum(), tuple(actor.parameters()))
        flat_a, flat_b = torch.cat([x.flatten() for x in ga]), torch.cat([x.flatten() for x in gb])
        assert (flat_a - flat_b).norm() / flat_a.norm() < 2e-5
        torch.testing.assert_close(flat_a, flat_b, atol=3e-6, rtol=2e-4)


def test_packed_zero_heads_and_shared_parameter_identity():
    actor = small_controller()
    fast = PackedGlobalRewriter(actor)
    assert list(fast.parameters()) == list(actor.parameters())
    result = fast(make_snapshot())
    assert flattened(result).count_nonzero() == 0
    result.contraction(make_snapshot().weights, make_snapshot().vector_weights).backward()
    assert actor.row_decoder[-1].weight.grad.norm() > 0
