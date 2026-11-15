"""Experimental offline execution of anchored edits without dense weight gradients.

Real arithmetic matches (W + scale*(P B.T + A Q.T)) @ x. FP32 operation
ordering differs from materializing and rounding the edited weights, so this
backend requires measured logit/gradient qualification before production use.
With realized_parameters supplied, AnchoredLinear uses the actual stored FP32
weight and ordinary dense forward, while retaining thin edit gradients. This is
the explicit offline graph path. Frozen base parameters must not require gradients.
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from .gpt_precision import RMS_EPSILON


def factorized_logits(model, parameters, buffers, inputs, action, scale, *, realized_parameters=None, softcap_backend="native", linear_backend="ieee", activation_backend="native",dx_split_weights=None,dx_backend=None):
    """FP32 GPT equations with gradients only for thin edits and activations."""
    from .adaptive_action import AdaptiveAction
    adaptive = isinstance(action, AdaptiveAction)
    if adaptive and (realized_parameters is None or linear_backend != 'ieee'
                     or dx_backend not in (None, 'ieee') or dx_split_weights is not None):
        raise ValueError('experimental adaptive thin execution requires realized weights and IEEE arithmetic')
    if adaptive and any(v.requires_grad for n,v in action.vectors.items() if n.endswith('.gains')):
        raise ValueError('adaptive thin execution requires read-only normalization gains')
    if any(p.requires_grad or p.dtype != torch.float32 for p in parameters.values()):
        raise ValueError('factorized execution requires frozen FP32 base parameters')
    def linear(x, name):
        factor = action.matrices[name + '.weight']
        if adaptive:
            from .adaptive_linear import AdaptiveLinear
            return AdaptiveLinear.apply(x, realized_parameters[name+'.weight'].detach(),
                realized_parameters[name+'.bias'].detach(), factor.U, factor.C, factor.V,
                action.vectors[name+'.bias'], scale)
        if realized_parameters is not None:
            from .anchored_linear import AnchoredLinear
            split=(None,None) if dx_split_weights is None else dx_split_weights.get(name+'.weight',(None,None))
            return AnchoredLinear.apply(x, realized_parameters[name+'.weight'].detach(),
                realized_parameters[name+'.bias'].detach(),factor.P,factor.Q,factor.anchors.A,
                factor.anchors.B,action.vectors[name+'.bias'],scale,linear_backend,*split,dx_backend)
        if linear_backend!='ieee':raise ValueError('experimental linear backend requires realized weights')
        bias = parameters[name + '.bias'] + scale * action.vectors[name + '.bias']
        base = F.linear(x, parameters[name + '.weight'], bias)
        return base + scale * ((x @ factor.anchors.B) @ factor.P.T
                               + (x @ factor.Q) @ factor.anchors.A.T)

    def norm(x, name=None):
        if adaptive:
            gains = None if name is None else realized_parameters[name + '.gains'].detach()
        else:
            gains = None if name is None else parameters[name + '.gains'] + scale * action.vectors[name + '.gains']
        return F.rms_norm(x, (x.size(-1),), weight=gains, eps=RMS_EPSILON)

    def rotary(x, name):
        position = torch.arange(x.size(1), dtype=torch.float32, device=x.device)
        theta = torch.outer(position, buffers[name + '.angular_freq'])[None, :, None, :]
        cosine, sine = theta.cos(), theta.sin()
        first, second = x.chunk(2, dim=-1)
        return torch.cat((first * cosine + second * sine,
                          first * (-sine) + second * cosine), 3)

    factor = action.matrices['embed.weight']
    if adaptive:
        from .adaptive_linear import AdaptiveEmbedding
        x=AdaptiveEmbedding.apply(inputs,realized_parameters['embed.weight'].detach(),
                                  factor.U,factor.C,factor.V,scale)
    elif realized_parameters is not None:
        from .anchored_linear import AnchoredEmbedding
        x=AnchoredEmbedding.apply(inputs,realized_parameters['embed.weight'].detach(),factor.P,
                                  factor.Q,factor.anchors.A,factor.anchors.B,scale)
    else:
        x = F.embedding(inputs, parameters['embed.weight'])
        x = x + scale * (F.embedding(inputs, factor.anchors.B) @ factor.P.T
                         + F.embedding(inputs, factor.Q) @ factor.anchors.A.T)
    x = norm(x, 'norm1')
    for index, block in enumerate(model.blocks):
        prefix = f'blocks.{index}'
        u = norm(x, prefix + '.norm1')
        batch, length = inputs.shape
        heads, dim = block.attn.num_heads, block.attn.head_dim
        q, k, v = (linear(u, prefix + '.attn.' + kind).view(batch, length, heads, dim)
                   for kind in ('q', 'k', 'v'))
        q = rotary(norm(q), prefix + '.attn.rotary')
        k = rotary(norm(k), prefix + '.attn.rotary')
        with sdpa_kernel(SDPBackend.MATH):
            y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                              v.transpose(1, 2), scale=.12, is_causal=True).transpose(1, 2)
        x = x + linear(y.contiguous().view(batch, length, heads * dim), prefix + '.attn.proj')
        hidden = linear(norm(x, prefix + '.norm2'), prefix + '.mlp.fc')
        if activation_backend=="fused":
            from .fused_relu_square import fused_relu_square
            hidden=fused_relu_square(hidden)
        elif activation_backend=="native":hidden=hidden.relu().square()
        else:raise ValueError('unknown activation backend')
        x = x + linear(hidden, prefix + '.mlp.proj')
    logits = linear(norm(x, 'norm2'), 'proj')
    if softcap_backend == "fused":
        from .fused_softcap import fused_softcap
        return fused_softcap(logits)
    if softcap_backend != "native":raise ValueError("unknown softcap backend")
    return 15 * logits * (logits.square() + 225).rsqrt()
