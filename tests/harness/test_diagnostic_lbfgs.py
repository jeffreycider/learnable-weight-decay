import torch
from chisel.diagnostic_lbfgs import ClosureLBFGS


def test_optimizer_closure_minimizes_actual_objective_and_restores_history():
    p = torch.nn.Parameter(torch.tensor([5., -3.]))
    opt = ClosureLBFGS([p], lr=1., max_iter=8, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad(set_to_none=True)
        loss = (p - 2).square().sum()
        loss.backward()
        return loss.detach()
    opt.closure = closure
    opt.step()
    torch.testing.assert_close(p, torch.full_like(p, 2.))
    saved = opt.state_dict()
    other = ClosureLBFGS([p], lr=1., max_iter=8)
    other.load_state_dict(saved)
    assert other.state_dict()['state'][0]['func_evals'] == saved['state'][0]['func_evals']
