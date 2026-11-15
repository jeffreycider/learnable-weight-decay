"""Batched execution of the unchanged globally coupled Chisel functional.

This is a fused/batched *multi-kernel* backend, not a whole-training megakernel.
The original actor owns every parameter and remains the checkpoint authority.
No observation, pooling bank, attention token or gradient is omitted.
"""
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from .algebra import Factors, RawAction


class PackedGlobalRewriter(nn.Module):
    """Share an existing GlobalRewriter's parameters, batching coordinate work.

    Construct once: ``fast = PackedGlobalRewriter(actor)``. Optimize and save
    ``actor`` as before; ``fast(snapshot)`` returns the ordinary RawAction.
    Optional torch.compile is a separate optimization, never a kernel claim.
    """
    def __init__(self, actor, *, split_decoder=False):
        super().__init__()
        self.actor = actor
        self.split_decoder = split_decoder

    def _pool_banks(self, banks):
        # Different lengths must have independent softmax denominators. Batch
        # only equal-length banks; this has five groups on full-size PR323.
        groups = defaultdict(list)
        for i, bank in enumerate(banks):
            groups[bank.shape[0]].append(i)
        result = [None] * len(banks)
        actor = self.actor
        for indices in groups.values():
            bank = torch.stack([banks[i] for i in indices])
            weights = (actor.pool_queries @ bank.transpose(-1, -2)
                       / bank.shape[-1] ** .5).softmax(-1)
            pooled = actor.pool_projection(weights @ bank)
            for j, i in enumerate(indices):
                result[i] = pooled[j]
        return result

    def _decode(self, decoder, local, contexts, sizes):
        if not self.split_decoder:
            # Keep the reference's single reduction across [local;context].
            # Splitting it is algebraically exact but changes rounding before
            # the GPT's BF16 parameter cast, which can amplify tiny differences.
            broadcast = torch.cat([contexts[i].expand(size, -1) for i, size in enumerate(sizes)])
            return decoder(torch.cat((local, broadcast), -1))
        # W[h;c]+b = W_h h + W_c c + b. Project each context once before
        # expansion, rather than materializing a 192-wide concatenation for
        # every coordinate. Both summands remain differentiable.
        width = local.shape[-1]
        first = decoder[0]
        local_projected = F.linear(local, first.weight[:, :width], first.bias)
        projected_context = F.linear(contexts, first.weight[:, width:])
        broadcast = torch.cat([projected_context[i].expand(size, -1)
                               for i, size in enumerate(sizes)])
        return decoder[2](decoder[1](local_projected + broadcast))

    def forward(self, snapshot):
        with sdpa_kernel(SDPBackend.MATH):
            return self._forward(snapshot)

    def _forward(self, snapshot):
        actor = self.actor
        matrix_items = tuple(snapshot.matrices.items())
        vector_items = tuple(snapshot.vectors.items())
        row_sizes = [obs.rows.shape[0] for _, obs in matrix_items]
        col_sizes = [obs.columns.shape[0] for _, obs in matrix_items]
        vector_sizes = [obs.coordinates.shape[0] for _, obs in vector_items]
        matrix_count = len(matrix_items)
        encoded = actor.coordinate_encoder(torch.cat(
            [obs.rows for _, obs in matrix_items] + [obs.columns for _, obs in matrix_items]))
        row_total = sum(row_sizes)
        rows_flat, cols_flat = encoded[:row_total], encoded[row_total:]
        rows = rows_flat.split(row_sizes)
        cols = cols_flat.split(col_sizes)
        matrix_tokens = self._pool_banks([torch.cat((r, c)) for r, c in zip(rows, cols)])
        if vector_items:
            vectors_flat = actor.vector_encoder(torch.cat([obs.coordinates for _, obs in vector_items]))
            vector_local = vectors_flat.split(vector_sizes)
            vector_tokens = [token.mean(0, keepdim=True) for token in self._pool_banks(vector_local)]
        else:
            vector_tokens = []
        tokens = matrix_tokens + vector_tokens
        trajectory = snapshot.trajectories.reshape(snapshot.trajectories.shape[0], -1, 2, actor.rank)
        scales = trajectory.square().mean((0, 1, 3)).sqrt().clamp_min(1e-12).detach()
        normalized = (trajectory / scales[None, None, :, None]).flatten(1)
        trajectory_tokens = actor.trajectory_encoder(normalized) + actor.trajectory_scale_encoder(scales.log())
        all_tokens = torch.cat(tokens + [trajectory_tokens], 0)[None]
        latent = actor.latents[None]
        latent = actor.cross_norm(latent + actor.cross(latent, all_tokens, all_tokens, need_weights=False)[0])
        for block in actor.core:
            latent = block(latent)
        queries = torch.stack([token.mean(0) for token in tokens])[None]
        contexts = actor.readout(queries, latent, latent, need_weights=False)[0][0]
        p = self._decode(actor.row_decoder, rows_flat, contexts[:matrix_count], row_sizes).split(row_sizes)
        q = self._decode(actor.column_decoder, cols_flat, contexts[:matrix_count], col_sizes).split(col_sizes)
        # The exact anchored P projection has one small reduction per matrix.
        # Batch compatible row counts; do not replace it by an approximate or
        # detached projection, and do not form a dense projector.
        groups = defaultdict(list)
        for i, size in enumerate(row_sizes):
            groups[size].append(i)
        canonical = [None] * matrix_count
        for indices in groups.values():
            anchor = torch.stack([matrix_items[i][1].anchors.A for i in indices])
            raw = torch.stack([p[i] for i in indices])
            projected = raw - anchor @ (anchor.transpose(-1, -2) @ raw)
            for j, i in enumerate(indices):
                canonical[i] = projected[j]
        matrices = {name: Factors(canonical[i], q[i], obs.anchors)
                    for i, (name, obs) in enumerate(matrix_items)}
        vectors = {}
        if vector_items:
            output = self._decode(actor.vector_decoder, vectors_flat,
                                  contexts[matrix_count:], vector_sizes).squeeze(-1).split(vector_sizes)
            vectors = {name: output[i] for i, (name, _) in enumerate(vector_items)}
        return RawAction(matrices, vectors)
