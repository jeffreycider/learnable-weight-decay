"""Stride-exact static storage for Chisel observations, refillable per checkpoint.

A CUDA graph captures tensor addresses. Per-checkpoint rebinding of the actor's
observation (ControllerSnapshot + training Measurements, 892 leaves at
production shapes) therefore needs ONE set of device buffers that keeps every
leaf's shape AND strides fixed and only changes values. The owner's rebind pilot
showed why strides matter: cloning the snapshot contiguously changed the GEMM
layouts seen by the compiled actor (QR anchors are column-major, several leaves
are stride-0 broadcasts) and moved the raw gradient by 1.8e-4 relative versus
the eager reference.

This store allocates each leaf exactly as ``prefix_cache._restore_layout`` does
(zero-filled dense storage, ``as_strided`` to the recorded shape/stride, storage
offset 0) and refills it from any record with the same layout signature by
copying only the unique slice of broadcast axes (so overlapping destinations are
never written twice). Values, strides and storage offsets are tested equal to a
fresh unpack; the actor output on the store is tested bit-identical to the
actor output on the fresh snapshot. Nothing learned is stored: only the
immutable bank observation of the currently bound checkpoint.
"""
from __future__ import annotations
import dataclasses
import torch
from .controller import ControllerSnapshot, MatrixObservation, VectorObservation
from .algebra import Anchors, Measurements


def _leaves(snapshot, train):
    """Deterministic (name, tensor) order shared by allocate and refill."""
    out = []
    for name, obs in snapshot.matrices.items():
        out += [(f'{name}.anchors.A', obs.anchors.A), (f'{name}.anchors.B', obs.anchors.B),
                (f'{name}.rows', obs.rows), (f'{name}.columns', obs.columns)]
        out += [(f'{name}.weight_probes.{i}', t) for i, t in enumerate(obs.weight_probes)]
        out += [(f'{name}.train.{i}', t) for i, t in enumerate(train.matrices[name])]
    for name, obs in snapshot.vectors.items():
        out += [(f'{name}.coordinates', obs.coordinates), (f'{name}.weight', obs.weight), (f'{name}.train', train.vectors[name])]
    out += [('trajectories', snapshot.trajectories), ('weight_norm', snapshot.weight_norm), ('radius', snapshot.radius)]
    if snapshot.trajectory_mask is not None:
        out.append(('trajectory_mask', snapshot.trajectory_mask))
    return out


def layout_signature(snapshot, train):
    return tuple((name, tuple(t.shape), tuple(t.stride()), t.dtype) for name, t in _leaves(snapshot, train))


def _unique_view(tensor):
    view = tensor
    for axis, (dimension, step) in enumerate(zip(tensor.shape, tensor.stride())):
        if dimension > 1 and step == 0:
            view = view.narrow(axis, 0, 1)
    return view


def _allocate_like(tensor, device):
    shape, stride = tuple(tensor.shape), tuple(tensor.stride())
    size = 0 if any(d == 0 for d in shape) else 1 + sum((d - 1) * s for d, s in zip(shape, stride))
    storage = torch.zeros(size, dtype=tensor.dtype, device=device)
    return storage.as_strided(shape, stride)


class StaticSnapshotStore:
    """Allocate once from a template record; refill in place for later records."""

    def __init__(self, snapshot, train, *, device):
        self.device = torch.device(device)
        self.signature = layout_signature(snapshot, train)
        self._buffers = {name: _allocate_like(t, self.device) for name, t in _leaves(snapshot, train)}
        self.generation = 0
        self.bound = False
        self.refill(snapshot, train)

    def _check(self, snapshot, train):
        if layout_signature(snapshot, train) != self.signature:
            raise ValueError('observation layout (names/shapes/strides/dtypes) differs from the captured store; '
                             'reject this checkpoint or capture a separate graph')

    @torch.no_grad()
    def refill(self, snapshot, train, *, non_blocking=False):
        """Copy a new checkpoint's observation values into the fixed buffers.

        Any failure, INCLUDING a layout rejection before any copy, leaves the
        store marked unbound (``bound=False``) so a graph replay cannot proceed
        on stale or half-refilled buffers; a later successful refill re-binds
        it. (Astra's cycle-8 reproduction: the first version ran the layout
        check before clearing ``bound``, so a rejected checkpoint left the
        previous values marked bound.) Strides of the sources are irrelevant
        (values are read logically); strides of the destinations never change.
        """
        self.bound = False
        self._check(snapshot, train)
        for name, source in _leaves(snapshot, train):
            destination = self._buffers[name]
            _unique_view(destination).copy_(_unique_view(source), non_blocking=non_blocking)
        self.generation += 1
        self.bound = True
        return self.generation

    def addresses(self):
        return {name: (t.data_ptr(), t.storage_offset()) for name, t in self._buffers.items()}

    def tensor(self, name):
        return self._buffers[name]

    def snapshot(self, template_snapshot, template_train):
        """Views over the static buffers with the same dataclass structure.

        Returned objects alias the store: they are valid across refills (values
        change in place) and must be the ONLY observation handed to a captured
        actor. ``bound`` must be checked before every replay.
        """
        self._check(template_snapshot, template_train)
        b = self._buffers
        matrices, vectors, train_m, train_v = {}, {}, {}, {}
        for name, obs in template_snapshot.matrices.items():
            matrices[name] = MatrixObservation(Anchors(b[f'{name}.anchors.A'], b[f'{name}.anchors.B']),
                                               b[f'{name}.rows'], b[f'{name}.columns'],
                                               tuple(b[f'{name}.weight_probes.{i}'] for i in range(len(obs.weight_probes))))
            train_m[name] = tuple(b[f'{name}.train.{i}'] for i in range(len(template_train.matrices[name])))
        for name in template_snapshot.vectors:
            vectors[name] = VectorObservation(b[f'{name}.coordinates'], b[f'{name}.weight'])
            train_v[name] = b[f'{name}.train']
        return (ControllerSnapshot(matrices, vectors, b['trajectories'], b['weight_norm'], b['radius'],
                                   trajectory_mask=b.get('trajectory_mask')),
                Measurements(train_m, train_v))
