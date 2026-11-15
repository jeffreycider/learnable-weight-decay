"""Chisel with the unchanged NanoGPT residual block and linear input/output layers.

Observations are deterministically split into patches of coordinates. All read
tokens precede all write tokens in one causal sequence. Each write position can
therefore attend to every GPT observation, including later GPT layers. No
pooling, latent bottleneck, cross-attention module, or coordinate MLP is used.
"""
import torch
from torch import nn
from torch.nn import functional as F

from baseline.model import Block, RMSNorm, initialize_parameters_like_record
from .algebra import Factors, RawAction


class TransformerChisel(nn.Module):
    architecture = "nanogpt_blocks_patch_io_v1"

    def __init__(self, rank=8, trajectory_dim=400, model_dim=256,
                 num_layers=4, patch_size=32, trajectory_mask_features=False):
        super().__init__()
        if model_dim < 128 or model_dim % 128:
            raise ValueError("unchanged harness Block requires width divisible by head_dim=128")
        if rank < 1 or num_layers < 1 or patch_size < 1:
            raise ValueError("rank, layer count, and patch size must be positive")
        if patch_size * rank > model_dim:
            raise ValueError("matrix output patch must fit residual width: patch_size * rank <= model_dim")
        if trajectory_dim % (2 * rank):
            raise ValueError("trajectory must contain aligned activation/adjoint pairs")
        self.trajectory_mask_features=bool(trajectory_mask_features)
        self.rank, self.trajectory_dim = rank, trajectory_dim
        self.model_dim, self.num_layers, self.patch_size = model_dim, num_layers, patch_size
        self.matrix_input = nn.Linear(patch_size * (5 * rank + 15), model_dim)
        self.vector_input = nn.Linear(patch_size * 18, model_dim)
        # Two absolute log scales accompany the normalized trajectories.
        self.trajectory_input = nn.Linear(trajectory_dim + 2 + (trajectory_dim//(2*rank) if self.trajectory_mask_features else 0), model_dim)
        # Read row/column/vector/trajectory; write row/column/vector.
        self.token_type = nn.Embedding(7, model_dim)
        self.norm1 = RMSNorm(model_dim)
        self.blocks = nn.ModuleList([Block(model_dim) for _ in range(num_layers)])
        self.norm2 = RMSNorm(model_dim)
        self.matrix_output = nn.Linear(model_dim, patch_size * rank)
        self.vector_output = nn.Linear(model_dim, patch_size)
        # Exactly the harness's residual-branch initialization. The input and
        # output layers are Chisel-specific; output zero recovers identity edits.
        initialize_parameters_like_record(self.blocks)
        nn.init.normal_(self.token_type.weight, std=.02)
        for head in self.output_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    @property
    def output_heads(self):
        return self.matrix_output, self.vector_output

    def configuration(self):
        return dict(architecture=self.architecture, rank=self.rank,
                    trajectory_dim=self.trajectory_dim, model_dim=self.model_dim,
                    num_layers=self.num_layers, patch_size=self.patch_size,
                    head_dim=128, attention_heads=self.model_dim // 128,
                    ff_width=4 * self.model_dim, causal=True,
                    cuda_residual_dtype="bfloat16", parameter_dtype="float32")

    def _patches(self, coordinates):
        padding = (-coordinates.shape[0]) % self.patch_size
        return F.pad(coordinates.float(), (0, 0, 0, padding)).reshape(
            -1, self.patch_size * coordinates.shape[-1])

    def _inputs(self, snapshot):
        patches, kinds, sizes = [], [], []
        for obs in snapshot.matrices.values():
            for kind, values in ((0, obs.rows), (1, obs.columns)):
                patch = self._patches(values)
                patches.append(patch)
                sizes.append((patch.shape[0], values.shape[0]))
                kinds.append(torch.full((patch.shape[0],), kind, device=patch.device,
                                        dtype=torch.long))
        if not patches:
            raise ValueError("Chisel requires at least one matrix observation")
        matrix = self.matrix_input(torch.cat(patches))
        matrix_kinds = torch.cat(kinds)
        reads = [matrix + self.token_type(matrix_kinds)]
        writes = [matrix + self.token_type(matrix_kinds + 4)]
        vector_sizes = []
        if snapshot.vectors:
            vector_patches = []
            for obs in snapshot.vectors.values():
                patch = self._patches(obs.coordinates)
                vector_patches.append(patch)
                vector_sizes.append((patch.shape[0], obs.coordinates.shape[0]))
            vector = self.vector_input(torch.cat(vector_patches))
            reads.append(vector + self.token_type.weight[2])
            writes.append(vector + self.token_type.weight[6])
        trajectory = snapshot.trajectories.reshape(
            snapshot.trajectories.shape[0], -1, 2, self.rank).float()
        if self.trajectory_mask_features:
            boundaries=self.trajectory_dim//(2*self.rank)
            mask=getattr(snapshot,'trajectory_mask',None)
            if mask is None:mask=trajectory.new_ones(boundaries)  # legacy GPT has all boundaries
            if tuple(mask.shape)!=(boundaries,):raise ValueError('trajectory boundary mask shape differs')
            mask=mask.to(trajectory)
            # Root's domain adapter validates finite binary nonempty masks at
            # snapshot creation; this compiled path keeps normalization tensor-only.
            masked=torch.where(mask[None,:,None,None].bool(),trajectory,0.)
            denom=(mask.sum()*trajectory.shape[0]*self.rank).clamp_min(1)
            scales=(masked.square().sum((0,1,3))/denom).sqrt().clamp_min(1e-12).detach()
            normalized=(masked/scales[None,None,:,None]).flatten(1)
            features=torch.cat((normalized,scales.log().expand(normalized.shape[0],-1),mask.expand(normalized.shape[0],-1)),dim=-1)
        else:
            if getattr(snapshot,'trajectory_mask',None) is not None:
                raise ValueError('masked trajectories require explicit new actor architecture')
            scales = trajectory.square().mean((0, 1, 3)).sqrt().clamp_min(1e-12).detach()
            normalized = (trajectory / scales[None, None, :, None]).flatten(1)
            features=torch.cat((normalized, scales.log().expand(normalized.shape[0], -1)),dim=-1)
        trajectory=self.trajectory_input(features)
        reads.append(trajectory + self.token_type.weight[3])
        read = torch.cat(reads)
        write = torch.cat(writes)
        return torch.cat((read, write))[None], read.shape[0], matrix.shape[0], sizes, vector_sizes

    def token_counts(self, snapshot):
        matrix = sum((x.shape[0] + self.patch_size - 1) // self.patch_size
                     for obs in snapshot.matrices.values() for x in (obs.rows, obs.columns))
        vectors = sum((obs.coordinates.shape[0] + self.patch_size - 1) // self.patch_size
                      for obs in snapshot.vectors.values())
        writes = matrix + vectors
        reads = writes + snapshot.trajectories.shape[0]
        return dict(observation_tokens=reads, write_tokens=writes, total_tokens=reads + writes)

    def forward(self, snapshot):
        x, read_count, matrix_count, sizes, vector_sizes = self._inputs(snapshot)
        # The harness uses BF16 residual activations with FP32 master parameters.
        # Keep CPU correctness tests FP32; production can use ordinary Flash SDPA.
        if x.is_cuda:
            x = x.bfloat16()
        x = self.norm1(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm2(x[:, read_count:]).float()[0]
        matrix_values = self.matrix_output(x[:matrix_count])
        pieces = matrix_values.split([count for count, _ in sizes])
        factors = [part.reshape(-1, self.rank)[:length]
                   for part, (_, length) in zip(pieces, sizes)]
        matrices = {name: Factors.canonical(factors[2*i], factors[2*i+1], obs.anchors)
                    for i, (name, obs) in enumerate(snapshot.matrices.items())}
        vectors = {}
        if vector_sizes:
            values = self.vector_output(x[matrix_count:])
            pieces = values.split([count for count, _ in vector_sizes])
            vectors = {name: part.flatten()[:length]
                       for (name, _), part, (_, length) in
                       zip(snapshot.vectors.items(), pieces, vector_sizes)}
        return RawAction(matrices, vectors)
