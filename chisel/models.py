"""Image and learned-weight interfaces using the same NanoGPT residual blocks."""
import torch
from torch import nn
from baseline.model import Block, Linear, RMSNorm


@torch.no_grad()
def initialize_muonh(model):
    """PR267 post-constructor initialization; leave gains and ordinary biases intact."""
    for name, parameter in model.named_parameters():
        if name.endswith('.attn.proj.weight'):
            parameter.mul_(1.25)
        elif name.endswith('.mlp.proj.weight'):
            parameter.mul_(3.0)
        elif name.endswith('.mlp.fc.weight'):
            parameter.mul_(1.5)
        elif name == 'proj.weight' or 'proj' in name:
            parameter.zero_()


class CifarGPT(nn.Module):
    def __init__(self, width, depth, patch, readout, **block_args):
        super().__init__()
        if 32 % patch or readout not in ('mean', 'cls'):
            raise ValueError('invalid patch size or readout')
        self.patch, self.readout = patch, readout
        self.embed = nn.Linear(3 * patch * patch, width)
        self.class_token = nn.Parameter(torch.randn(1, 1, width) * .02)
        self.norm1 = RMSNorm(width)
        self.blocks = nn.ModuleList([Block(width, **block_args) for _ in range(depth)])
        self.norm2 = RMSNorm(width)
        self.proj = Linear(width, 10)
        initialize_muonh(self)

    def forward(self, images):
        batch, channels, height, width = images.shape
        patch = self.patch
        tokens = images.reshape(batch, channels, height // patch, patch, width // patch, patch)
        tokens = tokens.permute(0, 2, 4, 1, 3, 5).reshape(batch, -1, channels * patch * patch)
        hidden = torch.cat((self.embed(tokens), self.class_token.expand(batch, -1, -1)), dim=1)
        hidden = self.norm1(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = hidden[:, :-1].mean(dim=1) if self.readout == 'mean' else hidden[:, -1]
        logits = self.proj(self.norm2(hidden)).float()
        return 15 * logits * (logits.square() + 15**2).rsqrt()
