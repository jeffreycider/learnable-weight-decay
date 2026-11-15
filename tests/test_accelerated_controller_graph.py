import dataclasses

import pytest
import torch

from rewriter.accelerated_controller import PackedGlobalRewriter
from rewriter.accelerated_controller_graph import GraphedGlobalRewriter
from scripts.profile_controller import move_tree
from tests.test_accelerated_controller import flattened
from tests.test_controller import make_snapshot, small_controller


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graph capture')
def test_graph_actor_values_vjp_new_snapshot_and_parameter_updates():
    actor = small_controller().cuda()
    with torch.no_grad():
        for decoder in (actor.row_decoder, actor.column_decoder, actor.vector_decoder):
            decoder[-1].weight.normal_(std=.01)
    snapshot = move_tree(make_snapshot(), 'cuda')
    reference = PackedGlobalRewriter(actor)
    graph = GraphedGlobalRewriter(actor, snapshot)
    for offset in (0., .2):
        changed = dataclasses.replace(snapshot, trajectories=snapshot.trajectories+offset)
        a = flattened(reference(changed))
        direction = torch.randn_like(a)
        ga = torch.autograd.grad((a*direction).sum(), tuple(actor.parameters()))
        b = flattened(graph(changed))
        gb = torch.autograd.grad((b*direction).sum(), tuple(actor.parameters()))
        torch.testing.assert_close(a,b,atol=3e-7,rtol=2e-5)
        for x,y in zip(ga,gb):torch.testing.assert_close(x,y,atol=3e-6,rtol=2e-4)
        with torch.no_grad():actor.row_decoder[-1].weight.mul_(1.1)
    # The offline trainer accumulates across checkpoint states before Adam.
    # Verify replay buffers do not overwrite an earlier parameter cotangent.
    direction = torch.randn_like(flattened(reference(snapshot)))
    actor.zero_grad(set_to_none=True)
    for offset in (0., .3):
        changed=dataclasses.replace(snapshot,trajectories=snapshot.trajectories+offset)
        (flattened(reference(changed))*direction).sum().backward()
    expected=[p.grad.clone() for p in actor.parameters()]
    actor.zero_grad(set_to_none=True)
    with torch.no_grad():graph(snapshot)
    for offset in (0., .3):
        changed=dataclasses.replace(snapshot,trajectories=snapshot.trajectories+offset)
        (flattened(graph(changed))*direction).sum().backward()
    for actual,wanted in zip(actor.parameters(),expected):
        torch.testing.assert_close(actual.grad,wanted,atol=5e-6,rtol=2e-4)
