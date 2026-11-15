"""Thin direct next-gradient credit for an executed adaptive U C V.T edit.

Project the ordinary dense task gradients before the base optimizer mutates
them, then give the returned thin tensor tree to LagOneHypergradient.consume.
The retained graph is Chisel/QR/guard algebra only; no GPT replay or GPT graph
is needed. The generic lag-one owner must guard shared AND private parameters.
"""
from dataclasses import dataclass
from types import MappingProxyType
import uuid

import torch

from .adaptive_action import AdaptiveAction


@dataclass(frozen=True, init=False, eq=False)
class AdaptiveCreditRecord:
    """Bind one EXECUTED adaptive action to the caller's single lag-one owner.

    The caller commits the action before constructing this record, supplies all
    shared/private hyperparameters, and supplies immutable observation tensors.
    Resume reconstructs the action at saved parameters/inputs and creates a new
    record before any next-gradient projection. Projection payloads are not a
    reusable checkpoint format or old-task training data.
    """
    action: AdaptiveAction
    scale: torch.Tensor
    _identity: torch.Tensor

    def __init__(self, lag, action, scale, *, action_step, hyperparameters,
                 guarded_observations=()):
        if not isinstance(action, AdaptiveAction):
            raise TypeError('explicit adaptive action required')
        if not isinstance(scale, torch.Tensor) or scale.numel() != 1:
            raise ValueError('one differentiable or constant tensor scale required')
        # Own immutable inventories: a caller may recycle its action dictionaries
        # without replacing the factors that this record executed and guarded.
        owned = AdaptiveAction(MappingProxyType(dict(action.matrices)),
                               MappingProxyType(dict(action.vectors)))
        object.__setattr__(self, 'action', owned)
        object.__setattr__(self, 'scale', scale)
        # OS randomness, independent of all GPT/Chisel/data RNG streams.
        token = uuid.uuid4().bytes
        identity = torch.tensor([int.from_bytes(token[i:i+8], 'little') & ((1<<63)-1)
                                 for i in (0,8)], device=scale.device, dtype=torch.int64)
        object.__setattr__(self, '_identity', identity)
        factors = {n:{key:getattr(f,key) for key in ('U','C','V')}
                   for n,f in owned.matrices.items()}
        lag.from_contraction(action_step=action_step, hyperparameters=hyperparameters,
                             contraction=self._contraction,
                             guarded_tensors=(factors, owned.vectors, scale,
                                              self._identity, guarded_observations))

    def project(self, gradients, **normalization):
        projected = project_adaptive_gradients(self.action, gradients, **normalization)
        projected['action_identity'] = self._identity.clone()
        return projected

    def _contraction(self, projected):
        identity = projected.get('action_identity')
        if (not isinstance(identity, torch.Tensor) or identity.dtype != torch.int64 or
                identity.shape != self._identity.shape or not torch.equal(identity,self._identity)):
            raise ValueError('projection belongs to a different executed action identity')
        values = {key:value for key,value in projected.items() if key != 'action_identity'}
        return adaptive_credit_contraction(self.action, self.scale, values)


@torch.no_grad()
def project_adaptive_gradients(action, gradients, *, global_token_count,
                               loss_reduction, transpose_storage=()):
    if not isinstance(action, AdaptiveAction):
        raise TypeError('explicit adaptive action required')
    if type(global_token_count) is not int or global_token_count < 1:
        raise ValueError('positive global token count required')
    if loss_reduction not in ('global_token_sum', 'global_mean'):
        raise ValueError('explicit globally reduced gradient units required')
    names = set(action.matrices) | set(action.vectors)
    if set(gradients) != names:
        raise ValueError('ordinary gradient inventory differs from action')
    transpose_storage = set(transpose_storage)
    if not transpose_storage.issubset(action.matrices):
        raise ValueError('transpose names must identify matrices')
    denominator = global_token_count if loss_reduction == 'global_token_sum' else 1
    matrices, vectors, values = {}, {}, []
    for name,factor in action.matrices.items():
        if factor.U.dtype not in (torch.float32, torch.float64):
            raise ValueError('adaptive credit requires FP32 or FP64 factors')
        if not isinstance(gradients[name], torch.Tensor):
            raise ValueError('ordinary task gradients must be tensors, including explicit zeros')
        gradient = gradients[name].detach()
        if name in transpose_storage:
            gradient = gradient.T
        # Own one converted buffer: normalizing in place must never mutate the
        # ordinary GPT gradient, even when it already has the factor dtype.
        gradient = gradient.to(factor.U, copy=True).div_(denominator)
        if gradient.ndim != 2:
            raise ValueError('one globally reduced gradient matrix required')
        factor._validate_dense(gradient)
        # Two dense-gradient scans supply all three factor cotangents AND the
        # scalar needed for the guard derivative. These are detached action-time
        # measurements, not a cached projection reused after U/V change.
        right = gradient @ factor.V
        left = gradient.T @ factor.U
        core = left.T @ factor.V
        matrices[name] = dict(U=right @ factor.C.T, C=core, V=left @ factor.C)
        values.append((core*factor.C).sum())
    for name,vector in action.vectors.items():
        if vector.dtype not in (torch.float32, torch.float64):
            raise ValueError('adaptive credit requires FP32 or FP64 vectors')
        if not isinstance(gradients[name], torch.Tensor):
            raise ValueError('ordinary task gradients must be tensors, including explicit zeros')
        gradient = gradients[name].detach().to(vector, copy=True).div_(denominator)
        if gradient.shape != vector.shape:
            raise ValueError('vector gradient shape differs')
        vectors[name] = gradient
        values.append((vector*gradient).sum())
    return dict(matrices=matrices, vectors=vectors, value=sum(values).clone())


def adaptive_credit_contraction(action, scale, projected):
    """Value and first derivative equal <g, scale * action> at execution.

    Summing <U,g_U>+<C,g_C>+<V,g_V> alone has the right factor derivative but
    THREE times the action value, which gives a wrong guard derivative. The
    zero-valued differences below retain the factor derivatives while the
    separately measured contraction supplies the correct value for d(scale).
    This is one exact direct VJP, not a counterfactual loss or reusable reward.
    """
    if (set(projected) != {'matrices','vectors','value'} or
            projected['matrices'].keys() != action.matrices.keys() or
            projected['vectors'].keys() != action.vectors.keys()):
        raise ValueError('projected adaptive inventory differs')
    scalar = projected['value']
    if scalar.numel() != 1 or scalar.requires_grad:
        raise ValueError('detached scalar contraction required')
    for name,factor in action.matrices.items():
        bank = projected['matrices'][name]
        if set(bank) != {'U','C','V'}:
            raise ValueError('all adaptive factor cotangents required')
        for key in ('U','C','V'):
            value, gradient = getattr(factor,key), bank[key]
            if gradient.requires_grad or gradient.shape != value.shape:
                raise ValueError('detached matching factor cotangent required')
            scalar = scalar + ((value-value.detach())*gradient).sum()
    for name,value in action.vectors.items():
        gradient = projected['vectors'][name]
        if gradient.requires_grad or gradient.shape != value.shape:
            raise ValueError('detached matching vector cotangent required')
        scalar = scalar + ((value-value.detach())*gradient).sum()
    return scale*scalar
