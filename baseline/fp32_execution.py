"""Experimental FP32 GPT training equations, matched to offline Chisel readouts.

This changes GPT arithmetic, not the model architecture or Muon implementation.
Callers must promote model/master and optimizer state explicitly when converting
a captured mixed-precision prefix. This module performs no checkpoint conversion.
"""
from types import MethodType

import torch

from .execution import softcap_cross_entropy
from rewriter.gpt_precision import configure_precision, fp32_head_inputs


def _forward_fp32(self, inputs, targets):
    hidden = fp32_head_inputs(self, inputs, sensors=self._fp32_sensors)
    return self._fp32_loss(self.proj(hidden), targets)


def enable_fp32_execution(model, *, sensors=None, compile_loss=False):
    """Install before ordinary passes; preserve sum-of-token loss units.

    A supplied collector receives the explicit residual/normalization callbacks.
    Linear and embedding hooks continue to execute through their normal modules.
    No synthetic compile warmup or additional GPT evaluation is performed.
    """
    if hasattr(model, '_fp32_loss') or hasattr(model, '_compiled_loss'):
        raise ValueError('a specialized GPT execution path is already installed')
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError('promote GPT parameters explicitly before FP32 execution')
    configure_precision('fp32')
    model._fp32_sensors = sensors
    model._fp32_loss = (torch.compile(softcap_cross_entropy, fullgraph=True, dynamic=False)
                        if compile_loss else softcap_cross_entropy)
    model.forward = MethodType(_forward_fp32, model)
    return model
