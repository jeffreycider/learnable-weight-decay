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
        gains = codes.get('gains') if isinstance(codes, dict) else None
        normalize_gains = codes.get('normalize_gains', False) if isinstance(codes, dict) else False
        codes = codes['low_rank'] if isinstance(codes, dict) else codes
        result = {}
        for index, name in enumerate(self.names):
            left, right = self.write_left[index], self.write_right[index]
            item = snapshot[name]
            scale = (item['encoder_cache']['raw_scale'] if 'encoder_cache' in item
                     else item['raw'].detach().square().mean().sqrt())
            result[name] = {bank: (left @ codes[index, side] @ right.T) * scale
                            for side, bank in enumerate(('raw', 'consolidated'))}
            if gains is not None:
                effective = item['raw'] + item['consolidated']
                for side, bank in enumerate(('raw', 'consolidated')):
                    values = gains[index, side]
                    row = left @ values[1:1 + self.rank]
                    column = right @ values[1 + self.rank:]
                    uniform = values[0]
                    if normalize_gains:
                        rows, columns = effective.shape
                        uniform = uniform / math.sqrt(rows * columns)
                        row, column = row / math.sqrt(columns), column / math.sqrt(rows)
                    result[name][bank] = result[name][bank] + effective * (
                        uniform + row[:, None] + column[None, :])
        return result


class Chisel(nn.Module):
    def __init__(self, rank, width, depth, full_rank_gains=False, normalize_gains=False,
                 attention_branch=False, **block_args):
        super().__init__()
        if normalize_gains and not full_rank_gains:
            raise ValueError('gain normalization requires the full-rank gain head')
        self.normalize_gains = normalize_gains
        self.rank = rank
        self.embed = nn.Linear(5 * rank * rank + 2 * rank + 9, width)
        self.blocks = nn.ModuleList([Block(width, **block_args) for _ in range(depth)])
        self.norm = RMSNorm(width)
        self.proj = nn.Linear(width, 2 * rank * rank)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.gain_proj = nn.Linear(width, 2 * (1 + 2 * rank)) if full_rank_gains else None
        if self.gain_proj is not None:
            nn.init.zeros_(self.gain_proj.weight)
            nn.init.zeros_(self.gain_proj.bias)
        self.attention_proj = nn.Linear(width, 2 * rank * rank) if attention_branch else None
        if self.attention_proj is not None:
            nn.init.zeros_(self.attention_proj.weight)
            nn.init.zeros_(self.attention_proj.bias)

    def forward(self, tokens):
        hidden = self.embed(tokens.unsqueeze(0))
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.norm(hidden)
        codes = self.proj(hidden)[0].reshape(-1, 2, self.rank, self.rank)
        if self.gain_proj is None and self.attention_proj is None:
            return codes
        result = dict(low_rank=codes, normalize_gains=self.normalize_gains)
        if self.gain_proj is not None:
            result['gains'] = self.gain_proj(hidden)[0].reshape(-1, 2, 1 + 2 * self.rank)
        if self.attention_proj is not None:
            result['attention'] = self.attention_proj(hidden)[0].reshape(-1, 2, self.rank, self.rank)
        return result


def propose(actor, probes, snapshot):
    codes = actor(probes.tokens(snapshot))
    edits = probes.decode(codes, snapshot)
    if isinstance(codes, dict) and 'attention' in codes:
        from chisel.attention_identity_fit import decode
        attention = decode(codes['attention'], probes, snapshot)
        for name in edits:
            edits[name]['raw'] = edits[name]['raw'] + attention[name]['raw']
    return edits
