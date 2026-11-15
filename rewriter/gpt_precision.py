"""Explicit arithmetic for the extra-evaluation FP32 Chisel experiment.

The baseline module stays untouched. FP32 evaluation uses its stored master
weights, promotes the embedding, and explicitly matches the pinned baseline's
RMS epsilon. Torch 2.11 CUDA uses the FP32 opmath epsilon even for BF16 inputs;
this was checked in both eager and compiled baseline execution.
This is an experimental precision change, not the pinned benchmark execution.
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

BASELINE = 'baseline'
FP32 = 'fp32'
RMS_EPSILON = 2.0 ** -23


def precision_metadata(mode):
    if mode not in (BASELINE, FP32):
        raise ValueError('unknown GPT precision')
    return dict(mode=mode, parameter_policy='stored_master_values_promoted_to_float32' if mode==FP32 else 'original_stored_dtypes',
                residual_dtype='float32' if mode==FP32 else 'bfloat16',
                rms_epsilon=RMS_EPSILON, rms_epsilon_policy='pinned_torch211_cuda_opmath_v2',
                attention_backend='math' if mode==FP32 else 'baseline_flash',
                tf32=False if mode==FP32 else 'baseline_policy')


def configure_precision(mode):
    precision_metadata(mode)
    if mode==FP32:
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False


def _norm(x, gains=None):
    return F.rms_norm(x, (x.size(-1),), weight=gains, eps=RMS_EPSILON)


def fp32_head_inputs(model, inputs, *, sensors=None):
    """Same residual/attention/MLP equations with explicit original epsilon.

All parameters and activations must be FP32. Existing Linear, MLP and RoPE
modules are reused. Residual and q/k normalization explicitly use the observed
baseline default, 2**-23. Earlier FP32 experiment code incorrectly assumed that
BF16 inputs implied 2**-7; its cached teachers are incompatible with this path.

An optional SensorCollector receives explicit normalization and residual
callbacks because this path bypasses those module forwards. Linear/embedding
hooks still run normally. Passing no collector leaves the offline path unchanged.
"""
    x=model.embed(inputs)
    if x.dtype!=torch.float32:
        raise ValueError('FP32 GPT requires a promoted FP32 embedding')
    def norm(value, module):
        output=_norm(value,module.gains)
        if sensors is not None:
            sensors.observe_explicit_norm(module,value,output,epsilon=RMS_EPSILON)
        return output
    x=norm(x,model.norm1)
    if sensors is not None:sensors.observe_residual_boundary(0,x)
    for index,block in enumerate(model.blocks):
        u=norm(x,block.norm1)
        attn=block.attn;batch,length=u.shape[:2]
        q=attn.q(u).view(batch,length,attn.num_heads,attn.head_dim)
        k=attn.k(u).view(batch,length,attn.num_heads,attn.head_dim)
        v=attn.v(u).view(batch,length,attn.num_heads,attn.head_dim)
        q,k=attn.rotary(_norm(q)),attn.rotary(_norm(k))
        with sdpa_kernel(SDPBackend.MATH):
            y=F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),
                                             scale=.12,is_causal=True).transpose(1,2)
        y=y.contiguous().view(batch,length,attn.num_heads*attn.head_dim)
        x=x+attn.proj(y)
        if sensors is not None:sensors.observe_residual_boundary(2*index+1,x)
        x=x+block.mlp(norm(x,block.norm2))
        if sensors is not None:sensors.observe_residual_boundary(2*index+2,x)
    return norm(x,model.norm2)
