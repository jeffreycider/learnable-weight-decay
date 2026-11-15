import pytest
import torch
from chisel.preservation_fit import ranking_objective, preservation_gradient, logits
from chisel.preservation import prediction_terms
from chisel.harness import RunState


def test_maximum_ranking_violation_is_not_diluted_by_preserved_positions():
    values = torch.zeros(8192, requires_grad=True)
    with torch.no_grad():
        values[5] = .01
    loss = ranking_objective(values, dict(ranking_reduction='microbatch_max'))
    loss.backward()
    assert values.grad[5] == 1 and values.grad.count_nonzero() == 1
    torch.testing.assert_close(loss, values[5])
    torch.testing.assert_close(ranking_objective(values, {}), values.mean())
    with pytest.raises(ValueError):
        ranking_objective(values, dict(ranking_reduction='unknown'))


def test_chunked_maximum_ranking_gradient_matches_explicit_weighted_objective():
    torch.manual_seed(5)
    model = torch.nn.Linear(4, 5).double()
    base = {name: p.detach().clone() for name, p in model.named_parameters()}
    effective = {'weight': (base['weight'] + .2 * torch.randn_like(base['weight'])).requires_grad_()}
    batch = dict(inputs=torch.randn(7, 4, dtype=torch.float64))
    cfg = dict(microbatch_size=3, runtime=dict(autocast=None), preservation=dict(top_k=4,
        margin_fraction=.5, ranking_power=1, ranking_reduction='microbatch_max',
        ranking_weight=3., logit_mse_weight=0.))
    state = RunState(extras=dict(logits_forward=logits, device=torch.device('cpu')))
    gradient, observed = preservation_gradient(model, base, base | effective, effective, batch, cfg, state, 0.)
    loss = 0.
    for start in range(0, 7, 3):
        x = batch['inputs'][start:start+3]
        terms = prediction_terms(logits(model, base | effective, x), logits(model, base, x), 4, .5, 1)
        loss += len(x) / 7 * 3 * terms['ranking'].max()
    expected = torch.autograd.grad(loss, effective['weight'])[0]
    torch.testing.assert_close(gradient['weight'], expected, rtol=1e-10, atol=1e-12)
    assert abs(observed['ranking_loss'] * 3 - loss.item()) < 1e-12
