"""Read-only norm/preservation gradient decomposition for offline diagnostics."""
import math

import torch


class GradientComponents:
    """Measure a panel's gradient increment before clipping or an optimizer step.

    One additional norm-only VJP retains the existing proposal graph. The normal
    panel subsequently supplies its combined norm/preservation backward. No GPT
    computation is added. Preservation is obtained by subtraction, so FP32
    gradient accumulation can introduce cancellation error. It includes every
    functional-preservation term, not just ranking when KL terms are active.
    """
    def __init__(self, regularizer, groups, *, weight=1., actor_credit=None):
        self.groups = {name: tuple(p for p in parameters if p.requires_grad)
                       for name, parameters in groups.items()}
        parameters = tuple(p for values in self.groups.values() for p in values)
        if len({id(p) for p in parameters}) != len(parameters):
            raise ValueError('Gradient roles must have disjoint parameters')
        if not parameters or not math.isfinite(weight) or weight <= 0:
            raise ValueError('Trainable parameters and positive finite weight required')
        self.parameters = parameters
        self.versions = [p._version for p in parameters]
        self.before = [None if p.grad is None else p.grad.detach().clone() for p in parameters]
        if actor_credit is None:
            self.norm = torch.autograd.grad(weight * regularizer, parameters,
                                            retain_graph=True, allow_unused=True)
        else:
            # The normal panel bridges detached action leaves back to Chisel.
            # Apply that same bridge using only the norm's factor cotangents;
            # autograd.grad leaves .grad buffers untouched at both boundaries.
            original, leaves = actor_credit
            active = [(o, leaf) for o, leaf in zip(original, leaves)
                      if o.requires_grad and leaf.requires_grad]
            gradients = (torch.autograd.grad(weight * regularizer, [leaf for _, leaf in active],
                                            retain_graph=True, allow_unused=True) if active else ())
            pairs = [(o, g) for (o, _), g in zip(active, gradients) if g is not None]
            self.norm = (torch.autograd.grad([o for o, _ in pairs], parameters,
                         grad_outputs=[g for _, g in pairs], retain_graph=True, allow_unused=True)
                         if pairs else (None,) * len(parameters))
        self.used = False

    @torch.no_grad()
    def finish(self):
        if self.used:
            raise RuntimeError('Gradient measurement is one-shot')
        if any(p._version != version for p, version in zip(self.parameters, self.versions)):
            raise RuntimeError('Parameters changed before gradient measurement')
        self.used = True
        result = {}
        index = 0
        for name, parameters in self.groups.items():
            n2 = f2 = total2 = dot = 0.
            for p in parameters:
                before, norm = self.before[index], self.norm[index]
                total = torch.zeros_like(p, dtype=torch.float64) if p.grad is None else p.grad.detach().double()
                if before is not None:
                    total = total - before.double()
                norm = torch.zeros_like(total) if norm is None else norm.detach().double()
                preservation = total - norm
                n2 += float(norm.square().sum())
                f2 += float(preservation.square().sum())
                total2 += float(total.square().sum())
                dot += float((norm * preservation).sum())
                index += 1
            result[name] = dict(norm_gradient_l2=math.sqrt(n2),
                                preservation_gradient_l2=math.sqrt(f2),
                                combined_gradient_l2=math.sqrt(total2),
                                norm_preservation_dot=dot,
                                norm_preservation_cosine=(dot / math.sqrt(n2 * f2) if n2 and f2 else None))
        self.before = self.norm = self.parameters = ()
        self.groups.clear()
        return dict(roles=result, scope='Per-panel gradients before clipping, including its state weight',
                    preservation_terms='Ranking and any active KL penalties',
                    subtraction_warning='Includes rounding from ordinary gradient accumulation',
                    extra_norm_vjps=1, extra_gpt_forwards=0, extra_gpt_backwards=0)
