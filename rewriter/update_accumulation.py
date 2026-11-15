"""Opt-in accumulation of optimizer increments below FP32 parameter resolution.

This module is not wired into a trainer. The primitive's caller computes the
increment and owns a separate zero-initialized FP32 remainder for each parameter.
Remainders are learned optimizer state: checkpoint, restore and reset them with
the corresponding parameters. All three tensors must have disjoint storage.
"""
import torch


@torch.no_grad()
def compensated_add_fp32_(parameter, update, remainder):
    """Apply a signed increment and retain its addition-rounding remainder.

    Forward computations still use the FP32 parameter. This compensates repeated
    addition; it cannot recover rounding in gradients or the update calculation,
    and it does not make each individual sub-ULP increment visible to a forward.
    """
    if any(t.dtype != torch.float32 for t in (parameter, update, remainder)):
        raise ValueError('FP32 parameter, update and remainder required')
    if parameter.shape != update.shape or parameter.shape != remainder.shape:
        raise ValueError('Accumulator shapes must match')
    if parameter.device != update.device or parameter.device != remainder.device:
        raise ValueError('Accumulator devices must match')
    adjusted = update + remainder
    next_parameter = parameter + adjusted
    remainder.copy_(adjusted - (next_parameter - parameter))
    parameter.copy_(next_parameter)


class CompensatedAdam(torch.optim.Adam):
    """Reference Adam with FP32 addition remainders; FP64 parameters use Adam.

    Native Adam advances the moments exactly once. For FP32 parameters, replace
    its rounded addition with an explicit FP32 increment and compensated add.
    This uses extra parameter copies and is not a fused/capturable implementation.
    Only zero-decay, real FP32/FP64 Adam with scalar hyperparameters is supported.
    An ordinary Adam checkpoint cannot be silently loaded as an exact resume.
    """
    marker = 'fp32_remainder_v1'

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=0,
                         foreach=False, fused=False)
        self.defaults['update_accumulation'] = self.marker
        for group in self.param_groups:
            group['update_accumulation'] = self.marker
        self._validate_groups()

    def _validate_groups(self):
        for group in self.param_groups:
            if group.get('update_accumulation') != self.marker:
                raise ValueError('Compensated Adam group marker required')
            if any(group.get(k, False) for k in ('weight_decay', 'amsgrad', 'maximize',
                                                'capturable', 'differentiable', 'foreach', 'fused')):
                raise ValueError('Unsupported compensated Adam option')
            if any(isinstance(v, torch.Tensor) for v in (group['lr'], group['eps'], *group['betas'])):
                raise ValueError('Scalar Adam hyperparameters required')
            if any(p.dtype not in (torch.float32, torch.float64) for p in group['params']):
                raise ValueError('Real FP32/FP64 Adam parameters required')

    @torch.no_grad()
    def step(self, closure=None):
        self._validate_groups()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        before = {p: p.detach().clone() for group in self.param_groups for p in group['params']
                  if p.grad is not None and p.dtype == torch.float32}
        super().step()
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p not in before:
                    continue
                state = self.state[p]
                if 'fp32_update_remainder' not in state:
                    state['fp32_update_remainder'] = torch.zeros_like(p)
                if group['lr'] == 0:
                    continue
                step = float(state['step'])
                denom = (state['exp_avg_sq'].sqrt() / (1 - beta2 ** step) ** 0.5).add_(group['eps'])
                increment = (state['exp_avg'] / denom).mul_(-group['lr'] / (1 - beta1 ** step))
                p.copy_(before[p])
                compensated_add_fp32_(p, increment, state['fp32_update_remainder'])
        return loss

    def load_state_dict(self, state_dict):
        if any(g.get('update_accumulation') != self.marker for g in state_dict['param_groups']):
            raise ValueError('Cannot silently convert ordinary Adam to compensated Adam')
        result = super().load_state_dict(state_dict)
        self._validate_groups()
        for group in self.param_groups:
            for p in group['params']:
                state = self.state.get(p, {})
                if p.dtype == torch.float32 and state:
                    remainder = state.get('fp32_update_remainder')
                    if remainder is None or remainder.shape != p.shape or remainder.dtype != p.dtype:
                        raise ValueError('Missing or incompatible saved FP32 remainder')
        return result
