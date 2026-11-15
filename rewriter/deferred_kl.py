"""Mean-KL penalty backward with one host readback per checkpoint.

``scripts.train_multistate_chisel.mean_kl_and_backward`` reads the FP64 KL of
every GPT microbatch back to the host (``float(kl.detach())``) before launching
that microbatch's backward, which drains the device queue four times per
checkpoint on top of the timing synchronizations. This variant accumulates the
fraction-weighted mean on the device in FP64 with the identical sequential
operation order (``total = total + kl*fraction`` starting from exact zero), so
the resulting mean KL, penalty coefficient and every gradient are bit-identical
to the reference; only the number of host/device synchronizations changes.
Component gradient diagnostics are not supported here (they need the reference).
"""
from __future__ import annotations
import math
import torch
import scripts.train_multistate_chisel as train


def timed(timings, name, fn, device):
    # Resolve at call time so a benchmark's timing arm (rewriter.step_timing.install)
    # applies here exactly as it applies to the reference function.
    return train.timed(timings, name, fn, device)


def mean_kl_and_backward_deferred(engine, parameters, regularizer, inputs, *, kl_budget, state_weight=1.,
                                  actor_credit=None, timings=None, teacher_provider=None,
                                  diagnostic_parameters=None, primal_dual=None, state_id=None):
    if diagnostic_parameters is not None:
        raise ValueError('deferred readback does not implement component gradient diagnostics; use the reference')
    if len(inputs) < 1 or state_weight <= 0:
        raise ValueError('nonempty inputs and positive state weight required')
    leaves = {n: value.detach().requires_grad_(value.requires_grad) for n, value in parameters.items()}
    device = next(iter(parameters.values())).device
    total = None
    for start in range(0, len(inputs), engine.microbatch_sequences):
        selected = inputs[start:start + engine.microbatch_sequences]
        teacher = timed(timings, 'teacher_target_total_seconds' if teacher_provider else 'teacher_cache_total_seconds',
                        lambda: teacher_provider(start, start + len(selected)) if teacher_provider
                        else engine.cache_teacher(selected, data_role='past_training')[0], device)
        kl = timed(timings, 'student_kl_forward_total_seconds', lambda: engine.student_parameter_loss(leaves, teacher), device)
        fraction = len(selected) / len(inputs)
        # Same IEEE-754 double operations and order as the host accumulation:
        # 0.0 + x == x exactly, then sequential fused-free multiply and add.
        contribution = kl.detach().double() * fraction
        total = contribution if total is None else total + contribution
        timed(timings, 'gpt_kl_backward_seconds', lambda: (kl * fraction).backward(), device)
        del teacher, kl
    # One readback per checkpoint for the KL mean and the regularizer check.
    total_value, regularizer_value = torch.stack((total.reshape(()), regularizer.detach().double().reshape(()))).tolist()
    if not math.isfinite(total_value):
        raise FloatingPointError('nonfinite raw KL')
    if not math.isfinite(regularizer_value):
        raise FloatingPointError('nonfinite norm/trust objective')
    constraint = primal_dual.terms(total_value, state_id) if primal_dual is not None else {}
    penalty, coefficient = ((constraint['kl_penalty'], constraint['kl_coefficient']) if constraint
                            else train.deadband(total_value, kl_budget))
    outputs = [regularizer]
    cotangents = [regularizer.new_tensor(state_weight)]
    for name, value in parameters.items():
        if value.requires_grad:
            gradient = leaves[name].grad
            if gradient is None:
                gradient = torch.zeros_like(value)
            outputs.append(value)
            cotangents.append(gradient * (coefficient * state_weight))
    timed(timings, 'norm_edit_backward_seconds' if actor_credit is not None else 'controller_norm_edit_backward_seconds',
          lambda: torch.autograd.backward(outputs, cotangents), device)
    if actor_credit is not None:
        original, credit_leaves = actor_credit
        pairs = [(value, leaf.grad) for value, leaf in zip(original, credit_leaves)
                 if value.requires_grad and leaf.grad is not None]
        if pairs:
            timed(timings, 'actor_backward_seconds',
                  lambda: torch.autograd.backward([x[0] for x in pairs], [x[1] for x in pairs]), device)
    return dict(mean_kl=total_value, kl_penalty=penalty, objective=regularizer_value + penalty,
                **{k: v for k, v in constraint.items() if k != 'kl_penalty'})
