from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

# Defaults preserve the pinned Track-3 architecture. Optional arguments support
# explicitly configured CIFAR and Chisel models; forwards retain NanoGPT blocks.


class RMSNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), weight=self.gains.type_as(x))


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, backward_backend="native") -> None:
        super().__init__(in_features, out_features, bias=True)
        if backward_backend not in ('native', 'tf32x3'):
            raise ValueError('unknown linear backward backend')
        self.backward_backend = backward_backend

    def forward(self, x: Tensor) -> Tensor:
        if self.backward_backend == 'tf32x3':
            from chisel.linear_backward import linear
            return linear(x, self.weight.type_as(x), self.bias.type_as(x))
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x))


class Rotary(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        # half-truncate RoPE (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim//4)]))

    def forward(self, x_BTHD: Tensor) -> Tensor:
        pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int = 128, *, causal=True,
                 attention_scale=0.12, learned_qk_scale=False, linear_backward="native") -> None:
        super().__init__()
        if dim % head_dim or head_dim % 4:
            raise ValueError("width must divide into heads; RoPE needs head_dim divisible by four")
        self.causal, self.attention_scale = causal, attention_scale
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim, linear_backward)
        self.k = Linear(dim, hdim, linear_backward)
        self.v = Linear(dim, hdim, linear_backward)
        self.proj = Linear(hdim, dim, linear_backward)
        self.rotary = Rotary(head_dim)
        self.log_scale = nn.Parameter(torch.zeros(self.num_heads)) if learned_qk_scale else None

    def forward(self, x: Tensor) -> Tensor:
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = self.rotary(q), self.rotary(k)
        if self.log_scale is not None:
            q = q * self.log_scale.exp()[None, None, :, None]
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           v.transpose(1, 2), scale=self.attention_scale, is_causal=self.causal).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, dim: int, expansion: int = 4, activation_backend="native", linear_backward="native") -> None:
        super().__init__()
        if activation_backend not in ("native", "fused_fp32"):
            raise ValueError("unknown MLP activation backend")
        self.activation_backend = activation_backend
        hdim = int(expansion * dim)
        self.fc = Linear(dim, hdim, linear_backward)
        self.proj = Linear(hdim, dim, linear_backward)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        if self.activation_backend == "fused_fp32" and x.dtype == torch.float32:
            from chisel.fused_relu_square import fused_relu_square
            x = fused_relu_square(x)
        else:
            x = x.relu().square()
        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(self, dim: int, *, expansion=4, activation_backend="native", linear_backward="native", **attention) -> None:
        super().__init__()
        self.attn = CausalSelfAttention(dim, linear_backward=linear_backward, **attention)
        self.mlp = MLP(dim, expansion, activation_backend, linear_backward)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int, **block_args) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([Block(model_dim, **block_args) for _ in range(num_layers)])
        self.proj = Linear(model_dim, vocab_size, block_args.get('linear_backward', 'native'))
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)

    def forward(self, inputs: Tensor, targets: Tensor | None = None) -> Tensor:
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        if targets is None:
            return logits
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")


def initialize_parameters_like_record(model: nn.Module) -> None:
    """The anchor's parameter init loop, verbatim: same iteration order, same
    normal_ call sequence -- including the std=0 gains draw, which consumes RNG.
    Callers must seed immediately before this and must not touch the global RNG
    between it and the initial broadcast.

    Raises:
        Exception: On a parameter name the record's init loop does not cover.
    """
    for name, p in model.named_parameters():
        w = p.data
        if name.endswith("weight"):
            if "proj" in name:
                w.zero_()
            elif "embed" in name:
                w.normal_()  # default torch init
            else:
                w.normal_(std=0.33**0.5 / w.size(-1)**0.5)  # default torch init
        elif name.endswith("bias"):
            w.zero_()
        elif name.endswith("gains"):
            w.normal_(mean=1, std=0)
        else:
            raise Exception(f"Uninitialized parameter: {name}")
