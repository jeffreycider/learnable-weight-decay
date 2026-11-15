"""Global NanoGPT transformer with trainable, checkpoint-private read/write maps."""
import math
import torch
from torch import nn
from baseline.model import Block, RMSNorm

BANKS = ('raw', 'consolidated', 'gradient', 'momentum', 'proposal')


class PrivateProbes(nn.Module):
    def __init__(self, shapes, rank):
        super().__init__()
        self.names, self.rank = tuple(shapes), rank
        self.read_left, self.read_right = nn.ParameterList(), nn.ParameterList()
        self.write_left, self.write_right = nn.ParameterList(), nn.ParameterList()
        for rows, columns in shapes.values():
            if min(rows, columns) < rank:
                raise ValueError('probe rank exceeds a matrix dimension')
            for bank, size in [(self.read_left, rows), (self.read_right, columns),
                               (self.write_left, rows), (self.write_right, columns)]:
                bank.append(nn.Parameter(torch.randn(size, rank) / math.sqrt(size)))

    def tokens(self, snapshot):
        tokens = []
        for index, name in enumerate(self.names):
            item = snapshot[name]
            left, right = self.read_left[index], self.read_right[index]
            features, scales = [], []
            for bank in BANKS:
                value = item[bank].detach()
                scale = value.square().mean().sqrt().clamp_min(1e-12)
                features.append((left.T @ (value / scale) @ right).flatten())
                scales.append(scale.log1p().reshape(1))
            activity = item.get('activity', {})
            for bank, probe in [('input', right), ('adjoint', left)]:
                value = activity.get(bank, probe.new_zeros(probe.shape[0])).detach()
                scale = value.square().mean().sqrt().clamp_min(1e-12)
                features.append((value / scale) @ probe)
                scales.append(scale.log1p().reshape(1))
            rows, columns = item['raw'].shape
            dimensions = left.new_tensor([math.log(rows), math.log(columns)])
            tokens.append(torch.cat((*features, *scales, dimensions)))
        return torch.stack(tokens)

    def decode(self, codes, snapshot):
        result = {}
        for index, name in enumerate(self.names):
            left, right = self.write_left[index], self.write_right[index]
            scale = snapshot[name]['raw'].detach().square().mean().sqrt()
            result[name] = (left @ codes[index] @ right.T) * scale
        return result


class Chisel(nn.Module):
    def __init__(self, rank, width, depth, **block_args):
        super().__init__()
        self.rank = rank
        self.embed = nn.Linear(5 * rank * rank + 2 * rank + 9, width)
        self.blocks = nn.ModuleList([Block(width, **block_args) for _ in range(depth)])
        self.norm = RMSNorm(width)
        self.proj = nn.Linear(width, rank * rank)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, tokens):
        hidden = self.embed(tokens.unsqueeze(0))
        for block in self.blocks:
            hidden = block(hidden)
        return self.proj(self.norm(hidden))[0].reshape(-1, self.rank, self.rank)


def propose(actor, probes, snapshot):
    return probes.decode(actor(probes.tokens(snapshot)), snapshot)
