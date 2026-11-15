import dataclasses

import pytest
import torch

from baseline.model import Block, RMSNorm
from rewriter.controller import GlobalRewriter
from rewriter.transformer_chisel import TransformerChisel
from tests.test_controller import make_snapshot


def model():
    torch.manual_seed(751)
    return TransformerChisel(rank=2, trajectory_dim=12, model_dim=128,
                             num_layers=2, patch_size=4)


def active_model():
    net = model()
    with torch.no_grad():
        for block in net.blocks:
            block.attn.proj.weight.normal_(std=.03)
            block.mlp.proj.weight.normal_(std=.03)
        for head in net.output_heads:
            head.weight.normal_(std=.03)
    return net


def flatten(action):
    return torch.cat([v.flatten() for f in action.matrices.values() for v in (f.P, f.Q)]
                     + [v.flatten() for v in action.vectors.values()])


def test_reuses_exact_harness_blocks_and_residual_initialization():
    net = model()
    assert all(type(block) is Block for block in net.blocks)
    assert type(net.norm1) is RMSNorm and type(net.norm2) is RMSNorm
    x = torch.randn(1, 13, 128)
    for block in net.blocks:
        assert block.attn.head_dim == 128
        assert block.mlp.fc.out_features == 4 * 128
        torch.testing.assert_close(block(x), x, atol=0, rtol=0)
    assert not any(isinstance(m, (torch.nn.MultiheadAttention, torch.nn.TransformerEncoderLayer))
                   for m in net.modules())


def test_zero_outputs_trainable_and_core_receives_gradients_after_first_step():
    net, snapshot = model(), make_snapshot()
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    initial = net(snapshot)
    assert flatten(initial).count_nonzero() == 0
    initial.contraction(snapshot.weights, snapshot.vector_weights).backward()
    assert all(head.weight.grad.norm() > 0 for head in net.output_heads)
    opt.step()
    opt.zero_grad(set_to_none=True)
    net(snapshot).contraction(snapshot.weights, snapshot.vector_weights).backward()
    assert net.matrix_input.weight.grad.norm() > 0
    assert net.blocks[0].attn.proj.weight.grad.norm() > 0
    assert net.blocks[0].mlp.proj.weight.grad.norm() > 0


def test_later_matrix_and_trajectory_can_influence_earliest_matrix_write():
    net, snapshot = active_model(), make_snapshot()
    before = net(snapshot)
    matrices = dict(snapshot.matrices)
    obs = matrices['second']
    matrices['second'] = dataclasses.replace(obs, rows=obs.rows + 2., columns=obs.columns - 3.)
    changed = net(dataclasses.replace(snapshot, matrices=matrices))
    assert (before.matrices['first'].Q - changed.matrices['first'].Q).norm() > 1e-6
    changed = net(dataclasses.replace(snapshot, trajectories=snapshot.trajectories + 3.))
    assert (before.matrices['first'].Q - changed.matrices['first'].Q).norm() > 1e-6
    # An early observation must influence a later write as well.
    matrices = dict(snapshot.matrices)
    matrices['first'] = dataclasses.replace(matrices['first'], rows=matrices['first'].rows + 3.)
    changed = net(dataclasses.replace(snapshot, matrices=matrices))
    assert (before.matrices['second'].Q - changed.matrices['second'].Q).norm() > 1e-6
    assert (before.vectors['gain'] - changed.vectors['gain']).norm() > 1e-6


def test_padded_patch_outputs_trimmed_and_action_canonical():
    net, snapshot = active_model(), make_snapshot()
    result = net(snapshot)
    for name, f in result.matrices.items():
        obs = snapshot.matrices[name]
        assert f.P.shape == obs.anchors.A.shape
        assert f.Q.shape == obs.anchors.B.shape
        torch.testing.assert_close(f.anchors.A.T @ f.P, torch.zeros(2, 2), atol=2e-6, rtol=0)
        torch.testing.assert_close(f.norm_sq(), f.dense().square().sum(), atol=2e-5, rtol=2e-5)
    assert result.vectors['gain'].shape == snapshot.vectors['gain'].weight.shape
    tokens, reads, _, _, _ = net._inputs(snapshot)
    counts = net.token_counts(snapshot)
    assert reads == counts['observation_tokens']
    assert tokens.shape[1] == counts['total_tokens']
    assert counts['total_tokens'] == 2 * counts['write_tokens'] + 3


def test_compile_reuses_graph_for_new_values_and_parameter_updates():
    torch._dynamo.reset()
    net, snapshot = model(), make_snapshot()
    graphs = []
    def backend(graph, example_inputs):
        graphs.append(graph)
        return graph.forward
    compiled = torch.compile(net, backend=backend, fullgraph=True, dynamic=False)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    for offset in (0., .5, -.25):
        current = dataclasses.replace(snapshot, trajectories=snapshot.trajectories + offset)
        opt.zero_grad(set_to_none=True)
        output = compiled(current)
        torch.testing.assert_close(flatten(output), flatten(net(current)))
        output.contraction(current.weights, current.vector_weights).backward()
        opt.step()
    assert len(graphs) == 1
    torch._dynamo.reset()


def test_old_architecture_checkpoint_cannot_load_and_hyperparameters_checked():
    with pytest.raises(RuntimeError):
        model().load_state_dict(GlobalRewriter().state_dict())
    with pytest.raises(ValueError, match='head_dim'):
        TransformerChisel(model_dim=192)
    with pytest.raises(ValueError, match='positive'):
        TransformerChisel(patch_size=0)
    with pytest.raises(ValueError, match='output patch'):
        TransformerChisel(model_dim=256, patch_size=128)
