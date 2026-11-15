import pytest
import torch
from rewriter.static_grouped_norm import StaticPointerTables, build_static_grouped_norm


def test_pointer_tables_are_persistent_keyed_and_replay_safe():
    tables = StaticPointerTables('cpu')
    a, b = torch.zeros(4), torch.zeros(5)
    with pytest.raises(RuntimeError, match='prepare'):
        tables.lookup((a, b))
    table = tables.prepare((a, b))
    assert table.tolist() == [a.data_ptr(), b.data_ptr()] and table.dtype == torch.int64
    assert tables.prepare((a, b)) is table and tables.lookup((a, b)) is table and len(tables) == 1
    c = torch.zeros(4)
    with pytest.raises(RuntimeError, match='prepare'):
        tables.lookup((c, b))
    assert tables.prepare((c, b)) is not table and len(tables) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA Triton fused norm')
def test_static_grouped_norm_matches_fused_reference_and_rejects_unprepared():
    from baseline.model import GPT
    from rewriter.accelerated_grouped_norm import AcceleratedGroupedNormObjective
    torch.manual_seed(5)
    model = GPT(64, 1, 128).float().cuda()
    parameters = {n: p.detach().clone() for n, p in model.named_parameters()}
    reference = AcceleratedGroupedNormObjective(parameters, zero_reference_policy='freeze')
    static = build_static_grouped_norm(parameters, zero_reference_policy='freeze')
    edits = {n: torch.randn_like(v) * 1e-3 for n, v in parameters.items()}
    leaves = {n: e.clone().requires_grad_() for n, e in edits.items()}
    def replacements(leaves):
        return {n: (parameters[n] + leaves[n]) if n in static.names else parameters[n] for n in parameters}
    rep = replacements(leaves)
    with pytest.raises(RuntimeError, match='prepare'):
        static.evaluate(rep)
    static.prepare(rep)
    expected = reference.evaluate(rep)
    actual = static.evaluate(rep)
    assert torch.equal(actual.mean_relative_change, expected.mean_relative_change)
    ga = torch.autograd.grad(actual.mean_relative_change, [leaves[n] for n in static.names])
    rep2 = replacements(leaves)
    expected2 = reference.evaluate(rep2)
    ge = torch.autograd.grad(expected2.mean_relative_change, [leaves[n] for n in static.names])
    for x, y in zip(ga, ge):
        assert torch.equal(x, y)
