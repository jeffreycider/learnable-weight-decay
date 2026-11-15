"""Isolated opt-in leaf compilation for instrumented strict-FP32 GPT probes.

Module hooks stay outside compiled forward callables. No production switch is
installed here; full online optimizer/sensor/resume qualification is still needed.
"""
from contextlib import contextmanager

import torch

from baseline.execution import softcap_cross_entropy
from baseline.model import GPT, Linear, Rotary


@contextmanager
def compiled_sensor_leaves(model, *, backend='inductor', compile_loss=True):
    """Temporarily compile pure leaf forwards and loss, preserving module identity.

Enter before a forward and leave after its backward. No warmup, RNG draw, state
copy or parameter replacement occurs. Explicit RMS/attention/residual operations
and every collector callback remain eager. Do not wrap the enclosing GPT in
torch.compile: that would recapture the mutable sensor hooks.
    Set compile_loss=False only for a matched loss-boundary isolation probe;
    this keeps the original eager loss while compiling the same leaf forwards.
    """
    if not isinstance(model, GPT) or not hasattr(model, '_fp32_sensors'):
        raise ValueError('explicit FP32 GPT execution required')
    if model._fp32_loss is not softcap_cross_entropy:
        raise ValueError('uncompiled FP32 loss required; nested contexts unsupported')
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('FP32 parameters required')
    leaves = [m for m in model.modules() if type(m) in (Linear, Rotary)]
    for module in leaves:
        if getattr(module.forward, '__func__', None) is not type(module).forward:
            raise ValueError('leaf already has an execution override')
    changed = []
    original_loss = model._fp32_loss
    try:
        for module in leaves:
            present = 'forward' in module.__dict__
            previous = module.__dict__.get('forward')
            compiled = torch.compile(module.forward, backend=backend,
                                     fullgraph=True, dynamic=False)
            changed.append((module, present, previous))
            module.forward = compiled
        if compile_loss:
            model._fp32_loss = torch.compile(original_loss, backend=backend,
                                            fullgraph=True, dynamic=False)
        yield model
    finally:
        model._fp32_loss = original_loss
        for module, present, previous in reversed(changed):
            if present:
                module.forward = previous
            else:
                del module.forward
