"""One CUDA graph launch per packed Chisel forward/backward, multiple kernels.

This is deliberately named a graph backend, not a single CUDA megakernel.
Original actor parameters remain the optimizer/checkpoint authority. One actor
forward may be outstanding at a time; finish its backward before another call.
"""
import dataclasses

import torch
from torch import nn

from .accelerated_controller import PackedGlobalRewriter
from .algebra import Anchors, Factors, RawAction


class _SnapshotCodec:
    def __init__(self, snapshot):
        self.template = snapshot
        self.matrix_names = tuple(snapshot.matrices)
        self.vector_names = tuple(snapshot.vectors)
        tensors = self.tensors(snapshot)
        self.shapes = tuple(t.shape for t in tensors)
        self.sizes = tuple(t.numel() for t in tensors)
        self.device = tensors[0].device

    def tensors(self, snapshot):
        return [tensor for name in self.matrix_names for tensor in
                (snapshot.matrices[name].rows, snapshot.matrices[name].columns,
                 snapshot.matrices[name].anchors.A, snapshot.matrices[name].anchors.B)] + [
                snapshot.vectors[name].coordinates for name in self.vector_names] + [snapshot.trajectories]

    def pack(self, snapshot):
        if tuple(snapshot.matrices) != self.matrix_names or tuple(snapshot.vectors) != self.vector_names:
            raise ValueError('graphed Chisel snapshot names/order changed')
        tensors = self.tensors(snapshot)
        if any(t.shape != shape or t.dtype != torch.float32 or t.device != self.device or t.requires_grad
               for t, shape in zip(tensors, self.shapes)):
            raise ValueError('graphed Chisel requires fixed-shape detached FP32 CUDA observations')
        return torch.cat([t.reshape(-1) for t in tensors])

    def unpack(self, packed):
        tensors = iter(value.view(shape) for value, shape in zip(packed.split(self.sizes), self.shapes))
        matrices = {}
        for name in self.matrix_names:
            rows, cols, a, b = next(tensors), next(tensors), next(tensors), next(tensors)
            matrices[name] = dataclasses.replace(self.template.matrices[name], rows=rows, columns=cols,
                                                anchors=Anchors(a, b))
        vectors = {name: dataclasses.replace(self.template.vectors[name], coordinates=next(tensors))
                   for name in self.vector_names}
        return dataclasses.replace(self.template, matrices=matrices, vectors=vectors,
                                   trajectories=next(tensors))


class _FlatActor(nn.Module):
    def __init__(self, actor, codec):
        super().__init__()
        self.actor = PackedGlobalRewriter(actor)
        self.codec = codec

    def forward(self, packed):
        action = self.actor(self.codec.unpack(packed))
        return torch.cat([v.flatten() for factor in action.matrices.values() for v in (factor.P, factor.Q)]
                         + [v.flatten() for v in action.vectors.values()])


class GraphedGlobalRewriter(nn.Module):
    """Capture once; replay new same-shaped snapshots and updated actor weights.

    Forward and backward each replay a CUDA graph. They still contain multiple
    kernels. Capture itself is cold setup and must be timed separately.
    Only first-order differentiation and sequential forward/backward are supported.
    """
    def __init__(self, actor, sample_snapshot, *, warmup=3):
        super().__init__()
        if next(actor.parameters()).device.type != 'cuda':
            raise ValueError('GraphedGlobalRewriter requires CUDA')
        self.codec = _SnapshotCodec(sample_snapshot)
        self.actor = actor
        self.adapter = _FlatActor(actor, self.codec)
        sample = self.codec.pack(sample_snapshot)
        self.graphed = torch.cuda.make_graphed_callables(self.adapter, (sample,),
                                                        num_warmup_iters=warmup)
        # Replay owns and overwrites its backward buffers. AccumulateGrad may
        # otherwise adopt one as parameter.grad when grad starts at None, so a
        # later checkpoint replay would overwrite the earlier contribution.
        # Hooks run outside the captured graph and return ordinary owned grads.
        self._gradient_ownership_hooks = [p.register_hook(lambda gradient: gradient.clone())
                                          for p in actor.parameters()]
        self.output_shapes = tuple(shape for obs in sample_snapshot.matrices.values()
                                   for shape in (obs.anchors.A.shape, obs.anchors.B.shape)) + tuple(
                                   obs.weight.shape for obs in sample_snapshot.vectors.values())
        self.output_sizes = tuple(shape.numel() for shape in self.output_shapes)

    def forward(self, snapshot):
        flat = self.graphed(self.codec.pack(snapshot))
        tensors = iter(value.view(shape) for value, shape in zip(flat.split(self.output_sizes), self.output_shapes))
        matrices = {name: Factors(next(tensors), next(tensors), obs.anchors)
                    for name, obs in snapshot.matrices.items()}
        vectors = {name: next(tensors) for name in snapshot.vectors}
        return RawAction(matrices, vectors)
