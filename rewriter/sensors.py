"""Ordinary-backward sequence sketches, in sequence-mean CE units.

No per-sequence dense parameter gradients or extra model evaluations are used.
``adjoint_to_sequence_loss`` removes the trainer's loss multiplier (including
AMP/accumulation scaling) exactly once. DDP averaging does not affect local
activation adjoints: projected local sums are explicitly SUM-reduced here.
Only fixed-length, equally weighted, computationally separate sequences are
supported. Embeddings use the transpose of their stored table by default.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.distributed as dist
import torch.nn.functional as F


@dataclass(frozen=True)
class MatrixObservation:
    train: tuple[Tensor, Tensor]
    audit: tuple[Tensor, Tensor]
    ones: tuple[Tensor, Tensor]
    input_rms: Tensor
    output_rms: Tensor


@dataclass(frozen=True)
class VectorObservation:
    train: Tensor
    audit: Tensor
    ones: Tensor


@dataclass(frozen=True)
class SensorSnapshot:
    matrix: dict[str, MatrixObservation]
    vectors: dict[str, VectorObservation]
    trajectories: Tensor
    trajectory_positions: Tensor
    num_sequences: int
    sequence_length: int
    loss_units: str = "sequence mean CE in nats/token"


def sequence_signs(sequence_ids: Tensor, probes: int, seed: int, step: int = 0) -> Tensor:
    """Stateless integer hash: identical for a sequence on every rank and use.

    Seeds for train/audit are disjoint; these are pseudorandom Rademacher signs,
    with no consumption of the GPT or controller initialization RNG streams.
    """
    ids = sequence_ids.to(dtype=torch.int64)
    k = torch.arange(probes, device=ids.device, dtype=torch.int64)[:, None]
    # Prime-field hashes keep intermediates below signed-int64 overflow.
    modulus = 2147483647
    x = (ids[None, :] + (int(seed) % modulus) + 104729 * (step + 1)
         + 130363 * (k + 1)) % modulus
    x = (x * 48271 + 12345) % modulus
    x = (x * x + 16807 * (k + 1)) % modulus
    return ((x % 2) * 2 - 1).to(torch.float32)


class SensorCollector:
    """Hooks are dormant outside ``set_microbatch`` / ``finalize`` boundaries.

    Construct once, call reset before each step (anchors then become immutable),
    set_microbatch before each ordinary forward/backward, and finalize afterward.
    Linear/embedding shared parameters accumulate into one canonical entry.
    Unsupported editable parameters fail at installation rather than silently
    acquiring missing preservation measurements. All 1D gains/biases are measured.
    """

    def __init__(self, model: nn.Module, anchors: Mapping[str, tuple[Tensor, Tensor]],
                 num_sequences: int, sequence_length: int, signs_seed: int = 1729,
                 trajectory_rank: int | None = None, trajectory_positions: int = 64,
                 train_probes: int = 4, audit_probes: int = 4,
                 max_float_elements: int = 16777216):
        if num_sequences < 1 or sequence_length < 1:
            raise ValueError("positive fixed sequence counts/lengths required")
        self.model, self.N, self.T = model, num_sequences, sequence_length
        self.seed, self.nt, self.na = signs_seed, train_probes, audit_probes
        self.K = self.nt + self.na + 1
        self.max_float_elements = max_float_elements
        self.params = dict(model.named_parameters())
        self.names = {id(p): n for n, p in self.params.items()}
        # The first registered owner determines the canonical parameter name.
        # Shapes cannot distinguish a square embedding table from its transpose,
        # especially when that same Parameter is also used as a linear head.
        self.transpose_storage = {
            n: isinstance(model.get_submodule(n.rsplit(".", 1)[0]) if "." in n else model, nn.Embedding)
            for n, p in self.params.items() if p.ndim == 2
        }
        self.handles = []
        self.active = False
        self.ctx = None
        self._matrix_names = set()
        self._vector_names = set()
        self.boundaries = []
        if hasattr(model, "blocks") and hasattr(model, "norm1"):
            self.boundaries.append((model.norm1, False))
            for block in model.blocks:
                self.boundaries.extend([(block.norm2, True), (block, False)])
        device = next(model.parameters()).device
        self.omega = None
        if self.boundaries:
            dim = model.embed.embedding_dim
            r = trajectory_rank or next(iter(anchors.values()))[0].shape[1]
            gen = torch.Generator(device="cpu").manual_seed(signs_seed + 41041)
            self.omega = torch.linalg.qr(torch.randn(dim, r, generator=gen)).Q.to(device)
        gen = torch.Generator(device="cpu").manual_seed(signs_seed + 72817)
        count = min(trajectory_positions, self.N * self.T)
        self.positions = torch.randperm(self.N * self.T, generator=gen)[:count].sort().values
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                self._matrix_names.add(self.names[id(module.weight)])
                if isinstance(module, nn.Linear) and module.bias is not None:
                    self._vector_names.add(self.names[id(module.bias)])
                self.handles.append(module.register_forward_hook(self._matrix_forward))
            elif isinstance(module, nn.RMSNorm) or (
                    module.__class__.__name__ == "RMSNorm" and hasattr(module, "gains")):
                p = module.gains if hasattr(module, "gains") else module.weight
                if p is not None:
                    self._vector_names.add(self.names[id(p)])
                    self.handles.append(module.register_forward_hook(self._norm_forward))
        missing = set(self.params) - self._matrix_names - self._vector_names
        if missing:
            self.close()
            raise ValueError(f"no signed-gradient collector for parameters: {sorted(missing)}")
        for q, (module, pre) in enumerate(self.boundaries):
            if pre:
                self.handles.append(module.register_forward_pre_hook(
                    lambda mod, args, q=q: self._trajectory(q, args[0])))
            else:
                self.handles.append(module.register_forward_hook(
                    lambda mod, args, out, q=q: self._trajectory(q, out)))
        self.reset(anchors, 0)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.active = False
        self.ctx = None

    def reset(self, anchors: Mapping[str, tuple[Tensor, Tensor]], step: int = 0):
        if self.ctx is not None:
            raise RuntimeError("finalize previous step before resetting sensor buffers")
        if set(anchors) != self._matrix_names:
            raise ValueError("anchors must cover every unique matrix exactly once")
        self.step = step
        self.anchors = {n: (a.detach().clone().float(), b.detach().clone().float())
                        for n, (a, b) in anchors.items()}
        self.banks, self.activity = {}, {}
        for n, (a, b) in self.anchors.items():
            p = self.params[n]
            expected = tuple(p.shape[::-1]) if self.transpose_storage[n] else tuple(p.shape)
            if expected != (a.shape[0], b.shape[0]):
                raise ValueError(f"anchor dimensions disagree with computational orientation of {n}")
            self.banks[n] = [a.new_zeros(self.K, *a.shape), b.new_zeros(self.K, *b.shape)]
            self.activity[n] = [b.new_zeros(b.shape[0]), a.new_zeros(a.shape[0]), a.new_zeros(2)]
        self.vbanks = {n: p.new_zeros((self.K, *p.shape), dtype=torch.float32)
                       for n, p in self.params.items() if n in self._vector_names}
        device = next(self.model.parameters()).device
        r = self.omega.shape[1] if self.omega is not None else 0
        self.z = torch.zeros(len(self.positions), len(self.boundaries), 2, r, device=device)
        self.seen = set()
        self.active = True

    def set_microbatch(self, sequence_ids: Tensor, inputs: Tensor | None = None,
                       adjoint_to_sequence_loss: float | None = None):
        if not self.active:
            raise RuntimeError("reset collectors before beginning a new step")
        ids = sequence_ids.detach().to(device=next(self.model.parameters()).device, dtype=torch.int64)
        if ids.ndim != 1 or ids.numel() == 0 or bool(((ids < 0) | (ids >= self.N)).any()):
            raise ValueError("sequence IDs must be a nonempty vector in [0,N)")
        cpuids = ids.cpu().tolist()
        if len(set(cpuids)) != len(cpuids) or self.seen.intersection(cpuids):
            raise ValueError("duplicate sequence IDs within a training iteration")
        self.seen.update(cpuids)
        if inputs is not None and tuple(inputs.shape[:2]) != (ids.numel(), self.T):
            raise ValueError("fixed-length batch shape differs from declared sequence IDs/length")
        scale = 1 / self.T if adjoint_to_sequence_loss is None else adjoint_to_sequence_loss
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("adjoint scaling must be finite and positive")
        train = sequence_signs(ids, self.nt, self.seed, self.step)
        audit = sequence_signs(ids, self.na, self.seed + 1000003, self.step)
        signs = torch.cat((train, audit, torch.ones(1, len(ids), device=ids.device)))
        self.ctx = (ids, signs / math.sqrt(self.N), float(scale))

    def _check_tensor(self, x: Tensor):
        if x.ndim != 3 or x.shape[0] != len(self.ctx[0]) or x.shape[1] != self.T:
            raise ValueError("sensors require explicit (sequence, token, coordinate) axes")

    @torch.no_grad()
    def _matrix_forward(self, module, args, output):
        if not self.active or self.ctx is None:
            return
        if not output.requires_grad:
            return
        self._check_tensor(output)
        name = self.names[id(module.weight)]
        a, b = self.anchors[name]
        x = args[0].detach()
        ctx = self.ctx
        embedding = isinstance(module, nn.Embedding)
        # Embedding's physical operator is table.T; linear's is stored weight.
        transposed = embedding != self.transpose_storage[name]
        if transposed:
            a, b = b, a
        if embedding:
            if module.padding_idx is not None or module.scale_grad_by_freq or module.sparse or module.max_norm is not None:
                raise ValueError("embedding padding/frequency/sparse/max_norm variants need explicit sensor rules")
            xb = b[x.long()]
            activity_in = torch.bincount(x.long().flatten(), minlength=b.shape[0]).float()
        else:
            self._check_tensor(x)
            xf = x.float()
            xb = xf @ b
            activity_in = xf.square().sum((0, 1))
        activity = self.activity[name]
        in_idx, out_idx = (1, 0) if transposed else (0, 1)
        activity[in_idx].add_(activity_in)
        activity[2][in_idx].add_(x.shape[0] * x.shape[1])

        @torch.no_grad()
        def backward(delta):
            _, signs, scale = ctx
            gb, gta = self.banks[name]
            if transposed:
                gb, gta = gta, gb
            # Token chunks bound float temporaries even for the vocabulary head.
            # GEMM packing avoids einsum paths that expand signs over logits.
            df = delta.flatten(0, 1)
            xbf = xb.flatten(0, 1)
            xf = x.flatten(0, 1)
            n, rank = a.shape
            token_chunk = max(1, self.max_float_elements // max(n, b.shape[0] if not embedding else 1))
            for start in range(0, df.shape[0], token_chunk):
                stop = min(start + token_chunk, df.shape[0])
                d = df[start:stop].float() * scale
                si = torch.arange(start, stop, device=d.device) // self.T
                ss = signs[:, si].T
                signed_xb = (ss[:, :, None] * xbf[start:stop, None, :]).flatten(1)
                gb.add_((d.T @ signed_xb).view(n, self.K, rank).permute(1, 0, 2))
                da = d @ a
                signed_da = ss[:, :, None] * da[:, None, :]
                activity[out_idx].add_((d / self.N).square().sum(0))
                if not embedding and module.bias is not None:
                    self.vbanks[self.names[id(module.bias)]].add_(ss.T @ d)
                if embedding:
                    flat = xf[start:stop].long()
                    for k in range(self.K):
                        gta[k].index_add_(0, flat, signed_da[:, k])
                else:
                    result = xf[start:stop].float().T @ signed_da.flatten(1)
                    gta.add_(result.view(b.shape[0], self.K, rank).permute(1, 0, 2))
            activity[2][out_idx].add_(delta.shape[0] * delta.shape[1])
        output.register_hook(backward)

    @torch.no_grad()
    def _norm_forward(self, module, args, output, *, epsilon=None):
        if not self.active or self.ctx is None or not output.requires_grad:
            return
        self._check_tensor(output)
        p = module.gains if hasattr(module, "gains") else module.weight
        name = self.names[id(p)]
        x = args[0].detach()
        if p.ndim != 1 or p.numel() != x.shape[-1]:
            raise ValueError("only coordinate RMSNorm gains are supported")
        eps = getattr(module, "eps", None) if epsilon is None else epsilon
        ctx = self.ctx

        @torch.no_grad()
        def backward(delta):
            _, signs, scale = ctx
            xf, df = x.flatten(0, 1), delta.flatten(0, 1)
            chunk = max(1, self.max_float_elements // x.shape[-1])
            for start in range(0, len(xf), chunk):
                stop = min(start + chunk, len(xf))
                normalized = F.rms_norm(xf[start:stop], (x.shape[-1],), eps=eps).float()
                si = torch.arange(start, stop, device=x.device) // self.T
                local = df[start:stop].float() * normalized * scale
                self.vbanks[name].add_(signs[:, si] @ local)
        output.register_hook(backward)

    def observe_explicit_norm(self, module, inputs, output, *, epsilon):
        """Measure a norm evaluated outside Module.forward, with its actual eps.

        The FP32 GPT path evaluates RMSNorm explicitly to retain the original
        epsilon. Its linear modules still use the ordinary registered hooks.
        Call this only for a norm whose module forward was bypassed.
        """
        parameter = module.gains if hasattr(module, "gains") else module.weight
        if self.names.get(id(parameter)) not in self._vector_names:
            raise ValueError("explicit norm must belong to the collector model")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("explicit norm epsilon must be positive and finite")
        self._norm_forward(module, (inputs,), output, epsilon=epsilon)

    def observe_residual_boundary(self, index, output):
        """Measure a residual boundary when its module hook was bypassed."""
        if not 0 <= index < len(self.boundaries):
            raise ValueError("unknown residual boundary")
        self._trajectory(index, output)

    @torch.no_grad()
    def _trajectory(self, q: int, x: Tensor):
        if not self.active or self.ctx is None or not x.requires_grad:
            return
        self._check_tensor(x)
        ids, _, scale = self.ctx
        selected = self.positions.to(ids.device)
        match = (selected[:, None] // self.T == ids[None, :]).nonzero()
        if not len(match):
            return
        zi, bi = match[:, 0], match[:, 1]
        ti = selected[zi] % self.T
        self.z[zi, q, 0] = x.detach()[bi, ti].float() @ self.omega

        @torch.no_grad()
        def backward(delta):
            self.z[zi, q, 1] = (delta[bi, ti].float() * (scale / self.N)) @ self.omega
        x.register_hook(backward)

    @torch.no_grad()
    def finalize(self, distributed: bool = False) -> SensorSnapshot:
        if not self.active or self.ctx is None:
            raise RuntimeError("no active measurements to finalize")
        device = next(self.model.parameters()).device
        coverage = torch.zeros(self.N, device=device)
        coverage[list(self.seen)] = 1
        arrays = [coverage, self.z]
        arrays += [x for values in self.banks.values() for x in values]
        arrays += [x for values in self.activity.values() for x in values]
        arrays += list(self.vbanks.values())
        if distributed:
            if not dist.is_initialized():
                raise RuntimeError("distributed sensors require an initialized process group")
            for x in arrays:
                dist.all_reduce(x, op=dist.ReduceOp.SUM)
        if not bool((coverage == 1).all()):
            raise ValueError("every global sequence must be observed exactly once across ranks/microbatches")
        matrix = {}
        for n, (gb, gta) in self.banks.items():
            ai, ao, counts = self.activity[n]
            if bool((counts == 0).any()):
                raise RuntimeError(f"missing ordinary backward measurement for {n}")
            matrix[n] = MatrixObservation(
                (gb[:self.nt].clone(), gta[:self.nt].clone()),
                (gb[self.nt:self.nt + self.na].clone(), gta[self.nt:self.nt + self.na].clone()),
                (gb[-1].clone(), gta[-1].clone()),
                (ai / counts[0]).sqrt(), (ao / counts[1]).sqrt())
        vectors = {n: VectorObservation(v[:self.nt].clone(), v[self.nt:self.nt+self.na].clone(), v[-1].clone())
                   for n, v in self.vbanks.items()}
        self.active = False
        self.ctx = None
        return SensorSnapshot(matrix, vectors, self.z.flatten(1).clone(), self.positions.clone(), self.N, self.T)
