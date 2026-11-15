"""Opt-in, detached record46 A/B finite-action comparison; no GPT evaluations.

Supply the exact action-time BasisSnapshot and X, before its basis/state update.
Post-refresh optimizer state is not a substitute. This helper never installs
itself in a trainer and never changes precision, algorithms or global settings.
Results describe this device/runtime only; CPU BF16 is not an H100 replay proof.
"""
import dataclasses
import torch
from .soap46_map import BasisSnapshot, soap_direction, finite_muon, postprocess, nu


def _comparison(learned, zero):
    if learned.shape != zero.shape:
        raise ValueError('action comparison shapes differ')
    if not torch.isfinite(learned).all() or not torch.isfinite(zero).all():
        raise ValueError('nonfinite finite-action diagnostic')
    a, b = learned.double(), zero.double()
    an, bn = a.norm().item(), b.norm().item()
    difference = (a-b).norm().item()
    changed = int((learned != zero).sum().item())
    return dict(dtype=str(learned.dtype), shape=list(learned.shape),
                learned_norm=an, zero_residual_norm=bn,
                difference_norm=difference,
                relative_difference_vs_zero_norm=difference/bn if bn else None,
                cosine=(a*b).sum().item()/(an*bn) if an and bn else None,
                zero_reference_norm=(bn == 0), elements=learned.numel(),
                changed_elements=changed, changed_fraction=changed/learned.numel(),
                max_abs_difference=(a-b).abs().max().item())


@torch.no_grad()
def compare_basis_action(record: BasisSnapshot, x_left, x_right, *, parameter_name):
    """JSON-safe comparison with a zero-X counterfactual on ONE immutable record.

    Relative differences use the zero-X stage norm, returning None at norm zero.
    The final stage is committed displacement, not the much larger base weight.
    Cosine compares the learned and zero-X stage tensors, not their differences.
    Postprocess rounding sites are aggregated in the final displacement metric.
    Exact equality uses stored numeric values (no tolerance). Cast-adjoint
    hypergradients are intentionally not evaluated by this detached diagnostic.
    """
    if not isinstance(record, BasisSnapshot):
        raise TypeError('an exact action-time BasisSnapshot is required')
    if not isinstance(parameter_name, str) or not parameter_name:
        raise ValueError('a parameter name is required')
    if record.q_left is None or record.q_right is None:
        raise ValueError('basis-free first action has no learned basis residual')
    if x_left.shape != record.q_left.shape or x_right.shape != record.q_right.shape:
        raise ValueError('residual shape differs from its action-time basis')
    if torch.is_autocast_enabled(record.weight.device.type):
        raise ValueError('record46 diagnostic requires original eager autocast-off execution')
    if x_left.dtype != record.q_left.dtype or x_right.dtype != record.q_right.dtype:
        raise ValueError('residual dtype differs from action-time basis')
    if x_left.device != record.weight.device or x_right.device != record.weight.device:
        raise ValueError('residual device differs from action-time record')
    if record.weight.ndim != 2 or not record.weight.numel():
        raise ValueError('one nonempty matrix record is required')
    # Own all inputs: no caller version, gradient, optimizer or RNG is changed.
    r = dataclasses.replace(record, **{
        f.name: getattr(record, f.name).detach().clone()
        for f in dataclasses.fields(record)
        if isinstance(getattr(record, f.name), torch.Tensor)})
    left, right = x_left.detach().clone(), x_right.detach().clone()
    if any(not torch.isfinite(t).all() for t in (left, right)):
        raise ValueError('nonfinite residual')
    stages = []
    for l, h in ((left, right), (torch.zeros_like(left), torch.zeros_like(right))):
        pre = soap_direction(r, l, h)
        cast = pre.to(r.polar_dtype)
        oriented = cast.mT if pre.shape[-2] > pre.shape[-1] else cast
        normalized = oriented / nu(oriented, keepdim=True, eps=1e-7).to(oriented.dtype)
        update = finite_muon(pre, r.polar_dtype)
        committed = postprocess(r, update)
        # Match executed storage precision first, then subtract in FP64 to avoid
        # adding a second diagnostic-only cancellation/rounding boundary.
        displacement = committed.double()-r.weight.double()
        stages.append(dict(pre_muon_direction=pre, muon_cast_input=cast,
                           muon_normalized_input=normalized, finite_muon_output=update,
                           committed_displacement=displacement))
    return dict(schema='soap46_finite_action_diagnostic_v1',
                parameter_name=parameter_name, action_step=int(r.step),
                variant=r.variant, polar_dtype=str(r.polar_dtype),
                parameter_dtype=str(r.weight.dtype), device=str(r.weight.device),
                torch_version=torch.__version__,
                cuda_matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32) if r.weight.is_cuda else None,
                cuda_device_name=torch.cuda.get_device_name(r.weight.device) if r.weight.is_cuda else None,
                autocast_enabled=False,
                lr=float(r.lr), cwd=float(r.cwd),
                scope='original A/B; exact action-time fixed W,d,G,Q,V; no state refresh',
                base_parameter_norm=r.weight.double().norm().item(),
                residuals=dict(left=_comparison(r.q_left+left, r.q_left),
                               right=_comparison(r.q_right+right, r.q_right),
                               left_x_norm=left.double().norm().item(),
                               right_x_norm=right.double().norm().item(),
                               left_x_relative_basis_norm=(left.double().norm()/r.q_left.double().norm()).item() if r.q_left.double().norm().item() else None,
                               right_x_relative_basis_norm=(right.double().norm()/r.q_right.double().norm()).item() if r.q_right.double().norm().item() else None),
                stage_units=dict(pre_muon_direction='SOAP direction after nu normalization and attention gate',
                                 muon_cast_input='same direction rounded to configured Muon dtype',
                                 muon_normalized_input='transposed if tall, divided by nu in Muon dtype',
                                 finite_muon_output='12 finite iterations including original aspect factor',
                                 committed_displacement='stored committed weight minus original weight, subtracted in FP64; parameter units'),
                comparisons={key:_comparison(stages[0][key],stages[1][key]) for key in stages[0]},
                gpt_forwards=0, gpt_backwards=0,
                warning='Nonzero straight-through cast hypergradient does not guarantee a nonzero finite action difference.')


def validate_action_samples(steps, names, *, stop_step, residual_frame, compile_map):
    """Bound opt-in work by an explicit Cartesian product; no regex/all selector."""
    steps, names = tuple(steps), tuple(names)
    if bool(steps) != bool(names):
        raise ValueError('action diagnostics require both steps and exact parameter names')
    if not steps:
        return steps, names
    if residual_frame != 'basis' or compile_map:
        raise ValueError('action diagnostics require original eager basis-frame A/B')
    if (len(steps) > 32 or len(names) > 8 or len(steps)*len(names) > 128
            or len(set(steps)) != len(steps) or len(set(names)) != len(names)):
        raise ValueError('action diagnostics require unique selections: at most 32 steps, 8 names, 128 pairs')
    if any(type(step) is not int or not 0 <= step < stop_step for step in steps):
        raise ValueError('action diagnostic steps must be zero-based action indices within this run')
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError('action diagnostics require exact nonempty parameter names')
    return steps, names
