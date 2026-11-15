"""Independent dense-autograd oracle; no scored-run model evaluations."""
import pytest
import torch

from rewriter.adaptive_linear import AdaptiveLinear, AdaptiveEmbedding


@pytest.mark.parametrize('storage,factor', [(torch.float64, torch.float64),
    (torch.float32, torch.float32), (torch.float32, torch.float64)])
@pytest.mark.parametrize('bias', [False, True])
@pytest.mark.parametrize('guarded', [False, True])
def test_dense_forward_and_all_factor_guard_cotangents(storage, factor, bias, guarded):
    torch.set_num_threads(1)
    gen = torch.Generator().manual_seed(81)
    base = torch.randn(7, 9, generator=gen, dtype=storage)
    base_bias = torch.randn(7, generator=gen, dtype=storage) if bias else None
    initial = [torch.randn(2, 3, 9, generator=gen, dtype=storage),
               torch.randn(7, 2, generator=gen, dtype=factor),
               torch.randn(2, 3, generator=gen, dtype=factor),
               torch.randn(9, 3, generator=gen, dtype=factor)]
    if bias: initial.append(torch.randn(7, generator=gen, dtype=factor))
    cotangent = torch.randn(2, 3, 7, generator=gen, dtype=storage)
    values, grads = [], []
    for thin in (False, True):
        leaves = [t.clone().requires_grad_() for t in initial]
        x, u, c, v = leaves[:4]
        b = leaves[4] if bias else None
        raw_scale = torch.tensor(.3, dtype=factor, requires_grad=True)
        scale = raw_scale / (1 + c.square().sum()).sqrt() if guarded else raw_scale
        w = (base.to(factor) + scale * (u @ c @ v.T)).to(storage)
        wb = (base_bias.to(factor) + scale * b).to(storage) if bias else None
        if thin:
            y = AdaptiveLinear.apply(x, w.detach(), None if wb is None else wb.detach(), u, c, v, b, scale)
        else:
            y = torch.nn.functional.linear(x, w, wb)
        values.append(y.detach())
        grads.append(torch.autograd.grad((y * cotangent).sum(), leaves + [raw_scale]))
    torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)
    for dense, thin in zip(*grads):
        tolerance = 2e-12 if storage == torch.float64 else 3e-5
        torch.testing.assert_close(dense, thin, rtol=tolerance, atol=tolerance)


def test_zero_core_has_trainable_core_gradient_and_gradcheck():
    torch.set_num_threads(1); torch.manual_seed(3)
    base = torch.randn(4, 5, dtype=torch.float64)
    x = torch.randn(2, 5, dtype=torch.float64, requires_grad=True)
    u = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
    c = torch.zeros(2, 2, dtype=torch.float64, requires_grad=True)
    v = torch.randn(5, 2, dtype=torch.float64, requires_grad=True)
    s = torch.tensor(.2, dtype=torch.float64, requires_grad=True)
    def run(x, u, c, v, s):
        realized = (base + s * (u @ c @ v.T)).detach()
        return AdaptiveLinear.apply(x, realized, None, u, c, v, None, s)
    assert torch.autograd.gradcheck(run, (x, u, c, v, s))
    gu, gc, gv = torch.autograd.grad(run(x, u, c, v, s).sum(), (u, c, v))
    assert gu.count_nonzero() == gv.count_nonzero() == 0
    assert gc.norm() > 0


def test_reject_live_dense_weight():
    x = torch.ones(2, 5)
    with pytest.raises(ValueError, match='detached'):
        AdaptiveLinear.apply(x, torch.ones(4, 5, requires_grad=True), None,
            torch.ones(4, 2), torch.ones(2, 2), torch.ones(5, 2), None, torch.tensor(.2))


def test_embedding_repeated_ids_orientation_and_all_cotangents():
    torch.set_num_threads(1); torch.manual_seed(15)
    base = torch.randn(11, 7, dtype=torch.float64)
    ids = torch.tensor([[0, 3, 3], [10, 0, 3]])
    initial = [torch.randn(7, 2, dtype=torch.float64), torch.randn(2, 3, dtype=torch.float64),
               torch.randn(11, 3, dtype=torch.float64), torch.tensor(.2, dtype=torch.float64)]
    outcomes, gradients = [], []
    for thin in (False, True):
        u, c, v, scale = [x.clone().requires_grad_() for x in initial]
        weight = base + scale * (u @ c @ v.T).T
        out = (AdaptiveEmbedding.apply(ids, weight.detach(), u, c, v, scale) if thin
               else torch.nn.functional.embedding(ids, weight))
        outcomes.append(out)
        gradients.append(torch.autograd.grad(out.square().sum(), (u,c,v,scale)))
    torch.testing.assert_close(*outcomes, rtol=0, atol=0)
    for dense, thin in zip(*gradients):
        torch.testing.assert_close(dense, thin, rtol=1e-12, atol=1e-12)
    assert gradients[1][2][1].count_nonzero() == 0
