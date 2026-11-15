"""A full-batch optimizer control for diagnosing a single fitted checkpoint."""
import torch
from chisel.overfit_diagnostics import backward


class ClosureLBFGS:
    def __init__(self, params, **kwargs):
        self.optimizer = torch.optim.LBFGS(params, **kwargs)
        self.closure = None

    def step(self):
        if self.closure is None:
            raise RuntimeError('the task must install its objective before step')
        return self.optimizer.step(self.closure)

    def zero_grad(self, **kwargs):
        self.optimizer.zero_grad(**kwargs)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)


def setup(lr=1., max_iter=5, history_size=20, tolerance_grad=1e-10, tolerance_change=1e-12):
    def hook(config, state):
        if not config.get('diagnostic_lbfgs') or config['gradient_clip'] != float('inf'):
            raise ValueError('L-BFGS closure needs unclipped, unmoved parameters')
        parameters = list(state.model.parameters())
        if state.extras['fit']['kind'] == 'chisel':
            parameters += list(state.extras['fit']['probes'].parameters())
        state.optimizer = ClosureLBFGS(parameters, lr=lr, max_iter=max_iter,
            history_size=history_size, line_search_fn='strong_wolfe',
            tolerance_grad=tolerance_grad, tolerance_change=tolerance_change)
        state.backward = prepare
    return hook


def prepare(config, state):
    def closure():
        state.optimizer.zero_grad(set_to_none=True)
        backward(config, state)
        state.extras['closure_evaluations'] = state.extras.get('closure_evaluations', 0) + 1
        state.extras['metrics']['closure_evaluations'] = state.extras['closure_evaluations']
        return torch.tensor(state.extras['train_loss'], dtype=torch.float64,
                            device=state.extras['device'])
    state.optimizer.closure = closure
