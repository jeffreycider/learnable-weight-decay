"""Graph-capture-safe pointer tables for the fused grouped-norm objective.

``rewriter.accelerated_grouped_norm`` builds its raw-pointer tables with
``torch.tensor([t.data_ptr() ...], device=cuda)`` inside BOTH the forward and
the backward of ``_MultiTensorNorm``. That is a pageable host->device copy with
a stream synchronization: illegal during CUDA stream capture, and even a pinned
copy would replay the capture-time pointer values. This isolated wrapper moves
every table to persistent device storage prepared OUTSIDE capture:

  * replacement tables are keyed by the tuple of replacement data pointers; a
    graph replays with stable pool addresses, so the table prepared at capture
    time stays valid for that graph and is rejected for any other pointer set;
  * backward gradients are written into persistent buffers allocated once, with
    a persistent address table; gradients of the replacement intermediates are
    consumed within the same backward, so buffer reuse across replays is safe
    (never aliased into leaf ``.grad``);
  * host->device staging of a new table uses pinned memory and a non-blocking
    copy on the current stream, and ``prepare`` refuses to run while a capture
    is active.

Arithmetic, kernels, reference setup, zero-group policy and version guards of
the fused objective are unchanged; only where the pointer tables live changes.
Does not import Triton at module import time, so the table logic is testable
on CPU; the fused subclass requires CUDA + Triton like its parent.
"""
from __future__ import annotations
import torch
from torch.autograd.function import once_differentiable


def _capturing():
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


class StaticPointerTables:
    """Persistent int64 device tables keyed by the pointer tuple they describe."""

    def __init__(self, device):
        self.device = torch.device(device)
        self._tables = {}

    @staticmethod
    def key(tensors):
        return tuple(int(t.data_ptr()) for t in tensors)

    def prepare(self, tensors, *, allow_capture=False):
        """Build (once) and return the device table for these tensors.

        Outside capture this is an ordinary pinned, non-blocking host->device
        copy. Inside capture (``allow_capture=True``) the copy is recorded as a
        memcpy node whose pinned source is kept alive by this object, so every
        replay re-copies the SAME capture-time addresses; that is exactly right
        because the replacement tensors live in the graph's private pool and
        keep their addresses across replays. Eager-run tables (different
        addresses) stay cached separately and are never confused with them.
        """
        key = self.key(tensors)
        table = self._tables.get(key)
        if table is None:
            if _capturing() and not allow_capture:
                raise RuntimeError('pointer table must be prepared outside CUDA graph capture, or pass allow_capture=True to record a pinned memcpy node')
            host = torch.tensor(key, dtype=torch.int64)
            if self.device.type == 'cuda':
                host = host.pin_memory()
                table = torch.empty_like(host, device=self.device)
                table.copy_(host, non_blocking=True)
                # Pinned sources must outlive every replay of a captured memcpy node.
                self._host_keepalive = getattr(self, '_host_keepalive', []) + [host]
            else:
                table = host.clone()
            self._tables[key] = table
        return table

    def lookup(self, tensors):
        """Return the prepared table or fail loudly; never allocates (replay-safe)."""
        table = self._tables.get(self.key(tensors))
        if table is None:
            raise RuntimeError('replacement pointers were not prepared; call prepare() outside capture')
        return table

    def __len__(self):
        return len(self._tables)


class _StaticMultiTensorNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, metadata, value_table, gradient_buffers, gradient_table, *values):
        from .accelerated_grouped_norm import _partials, _groups
        device = values[0].device
        partials = torch.empty((2, metadata.blocks), dtype=torch.float64, device=device)
        delta = torch.empty(len(metadata.groups), dtype=torch.float64, device=device)
        edit = torch.empty_like(delta)
        _partials[(metadata.blocks,)](value_table, metadata.base_addresses, metadata.sizes,
            metadata.bf16, metadata.tensor_ids, metadata.offsets, partials, metadata.blocks,
            metadata.block_size, enable_fp_fusion=False)
        _groups[(len(metadata.groups),)](partials, metadata.starts, metadata.ends, delta, edit,
            metadata.blocks, metadata.reduction_size, enable_fp_fusion=False)
        ctx.norm_metadata = metadata
        ctx.value_table = value_table
        ctx.gradient_buffers = gradient_buffers
        ctx.gradient_table = gradient_table
        ctx.save_for_backward(*values, *metadata.bases)
        ctx.value_count = len(values)
        return delta, edit

    @staticmethod
    @once_differentiable
    def backward(ctx, delta_cotangent, edit_cotangent):
        from .accelerated_grouped_norm import _backward
        meta = ctx.norm_metadata
        _backward[(meta.blocks,)](ctx.value_table, meta.base_addresses, ctx.gradient_table, meta.sizes,
            meta.bf16, meta.tensor_ids, meta.offsets, meta.group_ids,
            delta_cotangent.contiguous(), edit_cotangent.contiguous(), meta.block_size,
            enable_fp_fusion=False)
        return (None, None, None, None, *ctx.gradient_buffers)


def build_static_grouped_norm(parameters, *, zero_reference_policy='freeze', **kwargs):
    """Fused grouped objective whose pointer tables live in persistent device storage."""
    from .accelerated_grouped_norm import AcceleratedGroupedNormObjective, _RelativeGroups
    from .grouped_norm import GroupedNormResult

    class StaticGroupedNormObjective(AcceleratedGroupedNormObjective):
        def __init__(self, parameters, **inner):
            super().__init__(parameters, **inner)
            device = self.bases[0].device
            self.tables = StaticPointerTables(device)
            self.gradient_buffers = tuple(torch.empty_like(b) for b in self.bases)
            self.gradient_table = self.tables.prepare(self.gradient_buffers)

        def prepare(self, replacements, *, allow_capture=False):
            """Stage the pointer table for this replacement bank. Call it in the
            capture pass too (allow_capture=True): the graph-pool addresses of the
            edited parameters are only known during capture (Astra review)."""
            return self.tables.prepare(tuple(replacements[name] for name in self.names), allow_capture=allow_capture)

        def evaluate(self, replacements):
            if any(value._version != self._reference_versions[name] for name, value in self.parameters.items()):
                raise RuntimeError('frozen grouped norm reference was modified in place')
            if replacements.keys() != self.parameters.keys():
                raise ValueError('replacement mapping differs from frozen reference')
            values = tuple(replacements[name] for name in self.names)
            for name, value in zip(self.names, values):
                base = self.parameters[name]
                if (value.shape != base.shape or value.dtype != base.dtype or value.device != base.device
                        or not value.is_contiguous()):
                    raise ValueError('replacement shape/dtype/device/contiguity mismatch: ' + name)
            for name in self.frozen_parameter_names:
                value, base = replacements[name], self.parameters[name]
                if value.data_ptr() != base.data_ptr() and not torch.equal(value, base):
                    raise ValueError('zero-reference group must remain frozen: ' + name)
            table = self.tables.lookup(values)  # replay-safe: no allocation, no host copy
            delta, edit = _StaticMultiTensorNorm.apply(self, table, self.gradient_buffers, self.gradient_table, *values)
            relative = _RelativeGroups.apply(delta, self.reference_vector)
            norms = (self.reference_vector + delta).clamp_min(0).sqrt().detach()
            return GroupedNormResult(relative.mean(), dict(zip(self.groups, relative.unbind())),
                                     dict(zip(self.groups, norms.unbind())), delta.sum(), edit.sum())

    return StaticGroupedNormObjective(parameters, zero_reference_policy=zero_reference_policy, **kwargs)
