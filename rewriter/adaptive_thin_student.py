"""Default-off exact-stored-forward GPT with adaptive thin first-order VJPs.

No trainer dispatch uses this module. Reordered derivative contractions need
full-size GPU accuracy/runtime qualification before adoption. Teacher, losses,
norm objective and the final shared/private graph remain owned by the panel.
"""
import torch
from .adaptive_action import AdaptiveAction, AdaptiveFactors
from .coordinate_thin_student import _ThinCredit
from .factorized_fp32 import factorized_logits
from .tokenwise_reference import tokenwise_panel


class AdaptiveExecutionCapture:
    def __init__(self):
        self.action = self.scale = None

    def __call__(self, action, scale):
        if self.action is not None:
            raise RuntimeError('one adaptive proposal per execution capture')
        if not isinstance(action, AdaptiveAction):
            raise TypeError('adaptive action required')
        self.action, self.scale = action, scale


class _AdaptiveThinForward:
    def __init__(self, engine, action, scale, realized):
        self.engine, self.action, self.scale, self.realized = engine, action, scale, realized

    def __getattr__(self, name):
        return getattr(self.engine, name)

    def _forward(self, parameters, inputs):
        return factorized_logits(self.engine.view.gpt, self.engine.parameters, self.engine.buffers,
            inputs, self.action, self.scale, realized_parameters=self.realized,
            linear_backend='ieee', dx_backend='ieee', softcap_backend='native', activation_backend='native')


def adaptive_thin_tokenwise_panel(engine, capture, parameters, regularizer, inputs, **kwargs):
    if capture.action is None or capture.scale is None:
        raise ValueError('capture the realized adaptive proposal first')
    if engine.gpt_precision != 'fp32' or engine.kl_dtype != torch.float64:
        raise ValueError('strict FP32 GPT and FP64 teacher loss required')
    transpose = getattr(engine, 'transpose_storage', {})
    if (not transpose.get('embed.weight', False)
            or any(transpose.get(n, False) for n in capture.action.matrices if n != 'embed.weight')):
        raise ValueError('adaptive thin GPT supports transposed embedding storage only')
    backward = kwargs.get('backward', True)
    originals, leaves = [], []
    def leaf(value):
        result = value.detach().requires_grad_(value.requires_grad and backward)
        originals.append(value); leaves.append(result)
        return result
    action = AdaptiveAction({n:AdaptiveFactors(leaf(f.U), leaf(f.C), leaf(f.V))
                             for n,f in capture.action.matrices.items()},
                            {n:leaf(v) for n,v in capture.action.vectors.items()})
    scale = leaf(capture.scale)
    realized = {n:p.detach() for n,p in parameters.items()}
    if backward:
        regularizer = _ThinCredit.apply(regularizer, leaves, *originals)
    wrapper = _AdaptiveThinForward(engine, action, scale, realized)
    return tokenwise_panel(wrapper, realized, regularizer, inputs, **kwargs)
