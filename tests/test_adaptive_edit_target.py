import pytest
import torch

from rewriter.adaptive_action import AdaptiveAction, AdaptiveFactors
from rewriter.adaptive_edit_target import edit_target_terms, matrix_edit_distance


def factors(n, m, r, s, dtype=torch.float64, grad=False):
    return AdaptiveFactors(*(torch.randn(shape, dtype=dtype).requires_grad_(grad)
                             for shape in ((n, r), (r, s), (m, s))))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('dims', [(11, 7, 2, 3), (3, 4, 5, 6)])
def test_dense_value_and_all_factor_gradients(dtype, dims):
    torch.manual_seed(18)
    n, m, r, s = dims
    p, t = factors(*dims, dtype, True), factors(n, m, 3, 2, dtype)
    loss = matrix_edit_distance(p, t)
    oracle = (p.dense() - t.dense()).square().sum()
    variables = (p.U, p.C, p.V)
    actual = torch.autograd.grad(loss, variables)
    expected = torch.autograd.grad(oracle, variables)
    tolerance = 3e-5 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(loss, oracle, rtol=tolerance, atol=tolerance)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=tolerance, atol=tolerance)


def test_finite_difference_and_no_dense_materialization(monkeypatch):
    torch.manual_seed(19)
    p, t = factors(7, 9, 2, 2, grad=True), factors(7, 9, 3, 3)
    monkeypatch.setattr(AdaptiveFactors, 'dense', lambda _: pytest.fail('dense edit materialized'))
    assert torch.autograd.gradcheck(lambda u, c, v: matrix_edit_distance(AdaptiveFactors(u, c, v), t),
                                    (p.U, p.C, p.V))


def test_distance_is_basis_invariant_and_stable_near_target():
    torch.manual_seed(20)
    t = factors(9, 12, 3, 3)
    q = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64)).Q
    p = AdaptiveFactors((t.U @ q).requires_grad_(), (q.T @ t.C).requires_grad_(),
                        t.V.clone().requires_grad_())
    loss = matrix_edit_distance(p, t)
    assert 0 <= loss < 1e-25
    for g in torch.autograd.grad(loss, (p.U, p.C, p.V)):
        assert g.abs().max() < 1e-11
    perturbed = AdaptiveFactors(t.U, t.C + 1e-8, t.V)
    torch.testing.assert_close(matrix_edit_distance(perturbed, t),
                               (perturbed.dense()-t.dense()).square().sum(), rtol=1e-6, atol=1e-28)


def test_zero_output_has_trainable_core_and_degenerate_target_is_valid():
    torch.manual_seed(21)
    p, t = factors(9, 12, 3, 3, grad=True), factors(9, 12, 3, 3)
    with torch.no_grad():
        p.C.zero_()
        t.U[:, 1:].zero_()
    loss = matrix_edit_distance(p, t)
    gu, gc, gv = torch.autograd.grad(loss, (p.U, p.C, p.V))
    assert gu.count_nonzero() == 0 and gv.count_nonzero() == 0
    assert gc.norm() > 0
    zero = AdaptiveFactors(t.U, torch.zeros_like(t.C), t.V)
    assert matrix_edit_distance(p, zero) == 0


def test_action_inventory_vectors_and_frozen_targets():
    p, t = factors(5, 7, 2, 2, grad=True), factors(5, 7, 2, 2)
    v = torch.randn(5, dtype=torch.float64, requires_grad=True)
    teacher = torch.randn_like(v)
    terms = edit_target_terms(AdaptiveAction({'w':p}, {'b':v}), AdaptiveAction({'w':t}, {'b':teacher}))
    assert set(terms) == {'w', 'b'}
    torch.testing.assert_close(torch.autograd.grad(terms['b'], v)[0], 2*(v-teacher))
    with pytest.raises(ValueError, match='frozen'):
        matrix_edit_distance(p, p)
    with pytest.raises(ValueError, match='inventories'):
        edit_target_terms(AdaptiveAction({'w':p}, {}), AdaptiveAction({'other':t}, {}))
    with pytest.raises(ValueError, match='dimensions'):
        matrix_edit_distance(p, factors(7, 5, 2, 2))


def test_actual_transformer_and_private_probes_can_learn_frozen_edit_targets():
    # Integration test only: the teacher is another copy of this architecture,
    # not a function-preservation oracle. No GPT forward is performed.
    from tests.test_adaptive_preservation_actor import actual_actor_case
    _, actor, private, snapshot, _, _ = actual_actor_case()
    teacher_private = actor.make_private(snapshot, state_key=('test', 0, 6), seed=88)
    with torch.no_grad():
        target = actor(snapshot, teacher_private)
        initial_private = private.matrix_rows[0].raw.clone()
        initial_body = actor.blocks[0].attn.proj.weight.clone()
    optimizer = torch.optim.Adam(list(actor.parameters())+list(private.parameters()), lr=2e-4)
    normalizer = target.norm_sq().detach()
    losses = []
    for step in range(30):
        optimizer.zero_grad(set_to_none=True)
        loss = sum(edit_target_terms(actor(snapshot, private), target).values()) / normalizer
        losses.append(float(loss.detach()))
        loss.backward()
        assert actor.matrix_output.weight.grad.norm() > 0
        assert private.matrix_rows[0].raw.grad.norm() > 0
        optimizer.step()
    assert losses[-1] < .8 * losses[0]
    assert not torch.equal(private.matrix_rows[0].raw, initial_private)
    assert not torch.equal(actor.blocks[0].attn.proj.weight, initial_body)
