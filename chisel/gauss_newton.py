"""Offline nonlinear least-squares refinement through Chisel and the frozen GPT."""
import math
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend


def dot(left, right):
    return (left.double() * right.double()).sum()


def conjugate_gradient(product, rhs, iterations, tolerance):
    solution = torch.zeros_like(rhs)
    residual, direction = rhs.clone(), rhs.clone()
    squared = dot(residual, residual)
    initial = squared.clone()
    used = 0
    for _ in range(iterations):
        if squared <= initial * tolerance**2 or squared == 0:
            break
        mapped = product(direction)
        used += 1
        curvature = dot(direction, mapped)
        if not torch.isfinite(curvature) or curvature <= 0:
            raise FloatingPointError('nonpositive or nonfinite Gauss-Newton curvature')
        alpha = squared / curvature
        solution = solution + alpha * direction
        residual = residual - alpha * mapped
        updated = dot(residual, residual)
        direction = residual + (updated / squared) * direction
        squared = updated
    return solution, used, (squared / initial.clamp_min(1e-300)).sqrt().item()


class GaussNewton:
    """Minimum-parameter-motion correction, with a finite-loss backtracking test.

    CG solves (J J^T + damping I) v = residual; the update is -J^T v.
    Only a single forward graph is retained per correction. No dense Jacobian.
    """
    def __init__(self, parameters, damping=1e-6, cg_steps=50, cg_tolerance=1e-3,
                 backtracks=10, relative_damping=0.):
        if damping <= 0 or relative_damping < 0 or cg_steps < 1 or not 0 < cg_tolerance < 1 or backtracks < 1:
            raise ValueError('invalid Gauss-Newton settings')
        self.parameters = list(parameters)
        self.damping, self.cg_steps = damping, cg_steps
        self.cg_tolerance, self.backtracks = cg_tolerance, backtracks
        self.relative_damping = relative_damping
        self.steps, self.evaluations, self.products = 0, 0, 0
        self.closure, self.metrics = None, {}

    def evaluate(self):
        self.evaluations += 1
        value = self.closure()
        if not torch.isfinite(value).all():
            raise FloatingPointError('nonfinite refinement residual')
        return value

    def step(self):
        with sdpa_kernel(SDPBackend.MATH):
            residual = self.evaluate()
            before = dot(residual.detach(), residual.detach())
            if before == 0:
                self.steps += 1
                self.metrics = dict(refinement_accepted=False, refinement_scale=0., cg_iterations=0,
                    cg_relative_residual=0., jacobian_products=self.products, closure_evaluations=self.evaluations)
                return
            cotangent = torch.zeros_like(residual, requires_grad=True)
            symbolic = torch.autograd.grad(residual, self.parameters, cotangent,
                                           create_graph=True, allow_unused=True)
            active = [(p, value) for p, value in zip(self.parameters, symbolic) if value is not None]
            if not active:
                raise ValueError('refinement residual has no trainable dependence')
            parameters, symbolic = zip(*active)

            def transpose(vector):
                return torch.autograd.grad(residual, parameters, vector, retain_graph=True)

            def product(vector):
                self.products += 1
                return torch.autograd.grad(symbolic, cotangent, transpose(vector),
                                           retain_graph=True)[0]

            # Scale damping to this parameterization's measured curvature; a fixed
            # absolute constant can be negligible for large learned decoder gains.
            rayleigh = dot(residual.detach(), product(residual.detach())) / before
            damping = self.damping + self.relative_damping * rayleigh.clamp_min(0)

            dual, iterations, relative = conjugate_gradient(
                lambda value: product(value) + damping * value,
                residual.detach(), self.cg_steps, self.cg_tolerance)
            delta = tuple(-value.detach() for value in transpose(dual))
            del product, transpose, residual, cotangent, symbolic, active
            saved = [p.detach().clone() for p in parameters]
            accepted, fraction = False, 1.
            try:
                with torch.no_grad():
                    for _ in range(self.backtracks):
                        for p, original, change in zip(parameters, saved, delta):
                            p.copy_(original + fraction * change)
                        candidate = self.evaluate()
                        if dot(candidate, candidate) < before:
                            accepted = True
                            break
                        fraction *= .5
            finally:
                if not accepted:
                    with torch.no_grad():
                        for p, original in zip(parameters, saved):
                            p.copy_(original)
            if not accepted:
                with torch.no_grad():
                    self.evaluate()  # Restore metrics to the retained parameters too.
            self.steps += 1
            self.metrics = dict(refinement_accepted=accepted, refinement_scale=fraction if accepted else 0.,
                cg_iterations=iterations, cg_relative_residual=relative,
                damping=damping.item(), initial_rayleigh=rayleigh.item(),
                jacobian_products=self.products, closure_evaluations=self.evaluations)

    def zero_grad(self, **kwargs):
        for p in self.parameters:
            p.grad = None

    def state_dict(self):
        return dict(steps=self.steps, evaluations=self.evaluations, products=self.products)

    def load_state_dict(self, saved):
        self.steps, self.evaluations, self.products = (saved[k] for k in ('steps', 'evaluations', 'products'))


def setup(damping=1e-6, cg_steps=50, cg_tolerance=1e-3, backtracks=10,
          norm_retention_weight=1., relative_damping=0.):
    def hook(config, state):
        from chisel.overfit_diagnostics import edited_weights
        from chisel.preservation import group_norm_change
        fit = state.extras['fit']
        if (fit['kind'] != 'chisel' or state.extras['world_size'] != 1
                or config.get('diagnostic_lbfgs') or config.get('compile_proposal')
                or config['runtime'].get('autocast') or norm_retention_weight <= 0):
            raise ValueError('refinement requires single-checkpoint eager FP32 Chisel')
        inputs = fit['batch']['inputs']
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            teacher = torch.func.functional_call(fit['model'], fit['base'], (inputs,)).detach()
            target = group_norm_change(fit['snapshot'], edited_weights(config, state)).detach()
        targets = torch.nn.Module()
        targets.register_buffer('teacher', teacher)
        targets.register_buffer('norm_change', target)
        state.extras.setdefault('checkpointables', {})['refinement_targets'] = targets
        state.extras['training_forwards'] = state.extras.get('training_forwards', 0) + 1
        state.optimizer = GaussNewton(list(state.model.parameters()) + list(fit['probes'].parameters()),
                                      damping, cg_steps, cg_tolerance, backtracks, relative_damping)

        def closure():
            effective = edited_weights(config, state)
            student = torch.func.functional_call(fit['model'], fit['base'] | effective, (inputs,))
            difference = student - targets.teacher
            norm = group_norm_change(fit['snapshot'], effective)
            residual = torch.cat((difference.flatten() / math.sqrt(difference.numel()),
                                  (norm_retention_weight * (norm - targets.norm_change)).reshape(1)))
            state.extras['train_loss'] = dot(residual.detach(), residual.detach()).item()
            state.extras['metrics'] = dict(group_norm_reduction_pct=-100 * norm.item(),
                norm_target_reduction_pct=-100 * targets.norm_change.item(),
                logit_mse=difference.detach().double().square().mean().item(),
                maximum_logit_change=difference.detach().abs().max().item(),
                evaluated_positions=len(inputs), norm_retention_weight=norm_retention_weight)
            state.extras['training_forwards'] += 1
            return residual

        state.optimizer.closure = closure
        state.backward = lambda config, state: None
    return hook


def progress():
    def hook(config, state):
        state.extras['metrics'].update(state.optimizer.metrics)
    return hook
