"""Stateless globally coupled neural functional with shared coordinate heads."""
from dataclasses import dataclass
import torch
from torch import Tensor, nn
from .algebra import Anchors, Factors, RawAction


@dataclass(frozen=True)
class MatrixObservation:
    anchors: Anchors
    rows: Tensor
    columns: Tensor
    weight_probes: tuple[Tensor, Tensor]


@dataclass(frozen=True)
class VectorObservation:
    coordinates: Tensor
    weight: Tensor


@dataclass(frozen=True)
class ControllerSnapshot:
    matrices: dict[str, MatrixObservation]
    vectors: dict[str, VectorObservation]
    trajectories: Tensor
    weight_norm: Tensor
    radius: Tensor
    # New mixed-domain stages explicitly mark padded trajectory boundaries.
    trajectory_mask: Tensor | None = None

    def __post_init__(self):
        mask=self.trajectory_mask
        if mask is not None:
            if mask.ndim!=1 or mask.dtype!=torch.bool or not bool(mask.any()):
                raise ValueError('trajectory mask must be a nonempty one-dimensional bool mask')
            if self.trajectories.shape[-1] % mask.numel():
                raise ValueError('trajectory mask must divide flattened trajectory width')

    @property
    def weights(self):
        return {n: m.weight_probes for n, m in self.matrices.items()}

    @property
    def vector_weights(self):
        return {n: v.weight for n, v in self.vectors.items()}


def coordinate_features(banks, anchor, rms, metadata, scale_floor=1e-12):
    """Each bank has one detached RMS scale, retained as log-scale metadata.

    No normalization depends on controller candidates. All storage is cloned.
    Four projected banks + anchor + activity + 4 scales + 10 metadata = 5r+15.
    """
    pieces, scales = [], []
    for bank in banks:
        bank = bank.detach().float()
        scale = bank.square().mean().sqrt().clamp_min(scale_floor)
        pieces.append(bank / scale)
        scales.append(scale.log().expand(bank.shape[0], 1))
    act_scale = rms.square().mean().sqrt().clamp_min(scale_floor)
    pieces += [anchor.float(), (rms / act_scale).unsqueeze(-1)] + scales
    meta = torch.as_tensor(metadata, device=anchor.device, dtype=torch.float32).clone()
    if meta.numel() != 10:
        raise ValueError('expected 10 metadata fields')
    # RMS absolute scale is retained in the metadata's final field.
    meta[-1] = act_scale.log()
    pieces.append(meta.expand(anchor.shape[0], -1))
    return torch.cat(pieces, -1).detach().clone()


def vector_features(banks, metadata, scale_floor=1e-12):
    pieces = []
    for bank in banks:
        bank = bank.detach().float()
        scale = bank.square().mean().sqrt().clamp_min(scale_floor)
        pieces += [(bank / scale).unsqueeze(-1), scale.log().expand(bank.numel(), 1)]
    meta = torch.as_tensor(metadata, device=banks[0].device, dtype=torch.float32)
    pieces.append(meta.expand(banks[0].numel(), -1))
    return torch.cat(pieces, -1).detach().clone()


class GlobalRewriter(nn.Module):
    """Legacy pre-transformer architecture, retained for historical tests only.

    Current fitting uses rewriter.transformer_chisel.TransformerChisel through
    chisel_factory. Do not initialize a current experiment from this class.
    """
    def __init__(self, rank=8, trajectory_dim=400, coordinate_width=64, global_width=128,
                 pool_queries=4, latents=64, blocks=3, heads=4, ff_width=512):
        super().__init__()
        self.rank = rank
        self.coordinate_encoder = nn.Sequential(nn.Linear(5 * rank + 15, coordinate_width),
                                                nn.GELU(), nn.Linear(coordinate_width, coordinate_width))
        self.vector_encoder = nn.Sequential(nn.Linear(18, coordinate_width), nn.GELU(),
                                            nn.Linear(coordinate_width, coordinate_width))
        self.pool_queries = nn.Parameter(torch.randn(pool_queries, coordinate_width) * .02)
        self.pool_projection = nn.Linear(coordinate_width, global_width)
        self.trajectory_encoder = nn.Sequential(nn.Linear(trajectory_dim, global_width), nn.GELU(),
                                                nn.Linear(global_width, global_width))
        self.trajectory_scale_encoder = nn.Linear(2, global_width)
        self.latents = nn.Parameter(torch.randn(latents, global_width) * .02)
        self.cross = nn.MultiheadAttention(global_width, heads, dropout=0., batch_first=True)
        self.cross_norm = nn.LayerNorm(global_width)
        self.core = nn.ModuleList([nn.TransformerEncoderLayer(global_width, heads, ff_width,
                             dropout=0., activation='gelu', batch_first=True, norm_first=True)
                             for _ in range(blocks)])
        self.readout = nn.MultiheadAttention(global_width, heads, dropout=0., batch_first=True)
        self.row_decoder = nn.Sequential(nn.Linear(coordinate_width + global_width, coordinate_width),
                                         nn.GELU(), nn.Linear(coordinate_width, rank))
        self.column_decoder = nn.Sequential(nn.Linear(coordinate_width + global_width, coordinate_width),
                                            nn.GELU(), nn.Linear(coordinate_width, rank))
        self.vector_decoder = nn.Sequential(nn.Linear(coordinate_width + global_width, coordinate_width),
                                            nn.GELU(), nn.Linear(coordinate_width, 1))
        for decoder in (self.row_decoder, self.column_decoder, self.vector_decoder):
            nn.init.zeros_(decoder[-1].weight)
            nn.init.zeros_(decoder[-1].bias)

    def pool(self, h):
        attention = (self.pool_queries @ h.T / h.shape[-1]**.5).softmax(-1)
        return self.pool_projection(attention @ h)

    def forward(self, snapshot):
        # The GPT uses muoff's deterministic Flash-only policy. The auxiliary
        # functional is FP32, so select deterministic math attention locally.
        from torch.nn.attention import sdpa_kernel, SDPBackend
        with sdpa_kernel(SDPBackend.MATH):
            return self._forward(snapshot)

    def _forward(self, snapshot):
        local, tokens = {}, []
        for name, obs in snapshot.matrices.items():
            rows = self.coordinate_encoder(obs.rows)
            cols = self.coordinate_encoder(obs.columns)
            local[name] = (rows, cols)
            tokens.append(self.pool(torch.cat((rows, cols), 0)))
        vector_local = {}
        for name, obs in snapshot.vectors.items():
            h = self.vector_encoder(obs.coordinates)
            vector_local[name] = h
            # Explicitly accounted extra vector tokens: one pooled token per vector.
            tokens.append(self.pool(h).mean(0, keepdim=True))
        trajectory = snapshot.trajectories.reshape(snapshot.trajectories.shape[0], -1, 2, self.rank)
        scales = trajectory.square().mean((0, 1, 3)).sqrt().clamp_min(1e-12).detach()
        normalized = (trajectory / scales[None, None, :, None]).flatten(1)
        trajectory_tokens = self.trajectory_encoder(normalized) + self.trajectory_scale_encoder(scales.log())
        all_tokens = torch.cat(tokens + [trajectory_tokens], 0)[None]
        latent = self.latents[None]
        latent = self.cross_norm(latent + self.cross(latent, all_tokens, all_tokens, need_weights=False)[0])
        for block in self.core:
            latent = block(latent)
        queries = torch.stack([t.mean(0) for t in tokens])[None]
        contexts = self.readout(queries, latent, latent, need_weights=False)[0][0]
        matrices, vectors = {}, {}
        for i, (name, obs) in enumerate(snapshot.matrices.items()):
            rows, cols = local[name]
            c = contexts[i]
            p = self.row_decoder(torch.cat((rows, c.expand(rows.shape[0], -1)), -1))
            q = self.column_decoder(torch.cat((cols, c.expand(cols.shape[0], -1)), -1))
            matrices[name] = Factors.canonical(p, q, obs.anchors)
        for i, (name, obs) in enumerate(snapshot.vectors.items(), len(matrices)):
            h, c = vector_local[name], contexts[i]
            vectors[name] = self.vector_decoder(torch.cat((h, c.expand(h.shape[0], -1)), -1)).squeeze(-1)
        return RawAction(matrices, vectors)
