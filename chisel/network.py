"""Global NanoGPT transformer with trainable, checkpoint-private read/write maps."""
import math
import torch
from torch import nn
from baseline.model import Block, RMSNorm
from chisel.sensor_cache import BANKS, normalized


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
            cache = item.get('encoder_cache')
            if cache is None:
                cache = normalized(item)
            features = [(left.T @ value @ right).flatten() for value in cache['banks']]
            features += [value @ probe for value, probe in zip(cache['activity'], (right, left))]
            tokens.append(torch.cat((*features, *cache['scales'], cache['dimensions'])))
        return torch.stack(tokens)

    def decode(self, codes, snapshot):
        result = {}
        for index, name in enumerate(self.names):
            left, right = self.write_left[index], self.write_right[index]
            item = snapshot[name]
            scale = (item['encoder_cache']['raw_scale'] if 'encoder_cache' in item
                     else item['raw'].detach().square().mean().sqrt())
            result[name] = {bank: (left @ codes[index, side] @ right.T) * scale
                            for side, bank in enumerate(('raw', 'consolidated'))}
        return result


class Chisel(nn.Module):
    def __init__(self, rank, width, depth, **block_args):
        super().__init__()
        self.rank = rank
        self.embed = nn.Linear(5 * rank * rank + 2 * rank + 9, width)
        self.blocks = nn.ModuleList([Block(width, **block_args) for _ in range(depth)])
        self.norm = RMSNorm(width)
        self.proj = nn.Linear(width, 2 * rank * rank)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, tokens):
        hidden = self.embed(tokens.unsqueeze(0))
        for block in self.blocks:
            hidden = block(hidden)
        return self.proj(self.norm(hidden))[0].reshape(-1, 2, self.rank, self.rank)


def propose(actor, probes, snapshot):
    return probes.decode(actor(probes.tokens(snapshot)), snapshot)
