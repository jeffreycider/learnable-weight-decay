"""Fixed-example equality constraints for a single-checkpoint overfit."""
import torch
from torch import nn
from chisel.preservation_fit import sampled_inputs


class LogitConstraints(nn.Module):
    def __init__(self, teacher, penalty):
        super().__init__()
        if penalty <= 0:
            raise ValueError('the augmented-Lagrangian penalty must be positive')
        self.register_buffer('teacher', teacher.detach().clone())
        self.register_buffer('multipliers', torch.zeros_like(teacher))
        self.register_buffer('penalty', teacher.new_tensor(penalty))
        self.register_buffer('updates', torch.zeros((), dtype=torch.long, device=teacher.device))

    @torch.no_grad()
    def ascend(self, student):
        self.multipliers.add_(self.penalty * (student - self.teacher))
        self.updates.add_(1)


def predictions(config, state, parameters):
    fit = state.extras['fit']
    outputs = []
    for inputs in sampled_inputs(fit['batch'], config).split(config['microbatch_size']):
        # Keep the teacher, student and audit compilation signatures identical.
        candidate = parameters | {name: parameters[name].detach().requires_grad_()
                                  for name in fit['snapshot']}
        with torch.enable_grad():
            outputs.append(state.extras['logits_forward'](fit['model'], candidate, inputs).detach())
        state.extras['training_forwards'] = state.extras.get('training_forwards', 0) + 1
    return torch.cat(outputs)


def setup():
    def hook(config, state):
        objective = config['preservation']
        if (not config.get('fit_logit_constraints') or state.extras['world_size'] != 1
                or not config.get('diagnostic_lbfgs') or config['runtime'].get('autocast')
                or config['diagnostic_kl_weight'] != 0 or objective['ranking_weight'] != 0
                or objective.get('center_logits', True)):
            raise ValueError('logit multipliers require a fixed, single-device raw-logit L-BFGS fit')
        teacher = predictions(config, state, state.extras['fit']['base'])
        constraints = LogitConstraints(teacher, 2 * objective['logit_mse_weight'])
        state.extras['logit_constraints'] = constraints
        state.extras.setdefault('checkpointables', {})['logit_constraints'] = constraints
    return hook


def update(every=25):
    if every <= 0:
        raise ValueError('multiplier update period must be positive')
    def hook(config, state):
        if (state.step + 1) % every:
            return
        from chisel.overfit_diagnostics import edited_weights
        fit = state.extras['fit']
        with torch.no_grad():
            effective = edited_weights(config, state)
            constraints = state.extras['logit_constraints']
            constraints.ascend(predictions(config, state, fit['base'] | effective))
        # A new multiplier changes the objective. Do not reuse its old curvature.
        state.optimizer.optimizer.state.clear()
        state.extras['metrics']['multiplier_updates'] = constraints.updates.item()
        state.extras['metrics']['multiplier_rms'] = constraints.multipliers.square().mean().sqrt().item()
    return hook
