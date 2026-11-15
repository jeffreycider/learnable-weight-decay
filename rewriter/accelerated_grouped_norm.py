"""Batched Triton stored-parameter norms, with exact first-order cast semantics.

Three kernel launches cover all editable tensors: partial FP64 reductions,
group reduction, and (during backward) all dense parameter cotangents. This is
a component of a multi-kernel step, not a whole-step persistent megakernel.
"""
import torch
from torch.autograd.function import once_differentiable

from .grouped_norm import GroupedNormObjective, GroupedNormResult

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:
    @triton.jit
    def _partials(Values, Bases, Sizes, BF16, TensorIds, Offsets, Out,
                  BLOCKS: tl.constexpr, BLOCK: tl.constexpr):
        block = tl.program_id(0)
        tensor = tl.load(TensorIds + block)
        indices = tl.load(Offsets + block) + tl.arange(0, BLOCK)
        valid = indices < tl.load(Sizes + tensor)
        vp, bp = tl.load(Values + tensor), tl.load(Bases + tensor)
        if tl.load(BF16 + tensor):
            new = tl.load(vp.to(tl.pointer_type(tl.bfloat16)) + indices, valid, 0).to(tl.float64)
            old = tl.load(bp.to(tl.pointer_type(tl.bfloat16)) + indices, valid, 0).to(tl.float64)
        else:
            new = tl.load(vp.to(tl.pointer_type(tl.float32)) + indices, valid, 0).to(tl.float64)
            old = tl.load(bp.to(tl.pointer_type(tl.float32)) + indices, valid, 0).to(tl.float64)
        delta = new - old
        edit = delta * delta
        change = 2. * (old * delta) + edit
        tl.store(Out + block, tl.sum(change, 0))
        tl.store(Out + BLOCKS + block, tl.sum(edit, 0))

    @triton.jit
    def _groups(Partials, Starts, Ends, Delta, Edit, BLOCKS: tl.constexpr, BLOCK: tl.constexpr):
        group = tl.program_id(0)
        start, end = tl.load(Starts + group), tl.load(Ends + group)
        index = start + tl.arange(0, BLOCK)
        change = tl.load(Partials + index, index < end, 0)
        edit = tl.load(Partials + BLOCKS + index, index < end, 0)
        tl.store(Delta + group, tl.sum(change, 0))
        tl.store(Edit + group, tl.sum(edit, 0))

    @triton.jit
    def _backward(Values, Bases, Gradients, Sizes, BF16, TensorIds, Offsets,
                  GroupIds, DeltaCotangent, EditCotangent, BLOCK: tl.constexpr):
        block = tl.program_id(0)
        tensor, group = tl.load(TensorIds + block), tl.load(GroupIds + block)
        indices = tl.load(Offsets + block) + tl.arange(0, BLOCK)
        valid = indices < tl.load(Sizes + tensor)
        vp, bp, gp = tl.load(Values + tensor), tl.load(Bases + tensor), tl.load(Gradients + tensor)
        if tl.load(BF16 + tensor):
            new = tl.load(vp.to(tl.pointer_type(tl.bfloat16)) + indices, valid, 0).to(tl.float64)
            old = tl.load(bp.to(tl.pointer_type(tl.bfloat16)) + indices, valid, 0).to(tl.float64)
        else:
            new = tl.load(vp.to(tl.pointer_type(tl.float32)) + indices, valid, 0).to(tl.float64)
            old = tl.load(bp.to(tl.pointer_type(tl.float32)) + indices, valid, 0).to(tl.float64)
        dg, eg = tl.load(DeltaCotangent + group), tl.load(EditCotangent + group)
        gradient = 2. * dg * new + 2. * eg * (new - old)
        if tl.load(BF16 + tensor):
            tl.store(gp.to(tl.pointer_type(tl.bfloat16)) + indices, gradient, valid)
        else:
            tl.store(gp.to(tl.pointer_type(tl.float32)) + indices, gradient, valid)


def _addresses(values, device):
    return torch.tensor([value.data_ptr() for value in values], dtype=torch.int64, device=device)


class _MultiTensorNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, metadata, *values):
        device = values[0].device
        addresses = getattr(metadata, "address_table", _addresses)(values, device)
        partials = torch.empty((2, metadata.blocks), dtype=torch.float64, device=device)
        delta = torch.empty(len(metadata.groups), dtype=torch.float64, device=device)
        edit = torch.empty_like(delta)
        _partials[(metadata.blocks,)](addresses, metadata.base_addresses, metadata.sizes,
            metadata.bf16, metadata.tensor_ids, metadata.offsets, partials, metadata.blocks,
            metadata.block_size, enable_fp_fusion=False)
        _groups[(len(metadata.groups),)](partials, metadata.starts, metadata.ends, delta, edit,
            metadata.blocks, metadata.reduction_size, enable_fp_fusion=False)
        ctx.norm_metadata = metadata
        ctx.addresses = addresses
        # Tensor version checking also protects the raw-pointer kernel path.
        ctx.save_for_backward(*values, *metadata.bases)
        ctx.value_count = len(values)
        return delta, edit

    @staticmethod
    @once_differentiable
    def backward(ctx, delta_cotangent, edit_cotangent):
        values = ctx.saved_tensors[:ctx.value_count]
        meta = ctx.norm_metadata
        gradients = tuple(torch.empty_like(value) for value in values)
        addresses = getattr(meta, "address_table", _addresses)(gradients, values[0].device)
        _backward[(meta.blocks,)](ctx.addresses, meta.base_addresses, addresses, meta.sizes,
            meta.bf16, meta.tensor_ids, meta.offsets, meta.group_ids,
            delta_cotangent.contiguous(), edit_cotangent.contiguous(), meta.block_size,
            enable_fp_fusion=False)
        return (None, *gradients)


class _RelativeGroups(torch.autograd.Function):
    @staticmethod
    def forward(ctx, delta, reference):
        squared = reference + delta
        torch._assert_async(torch.all(squared >= -1e-10 * reference),
                            'negative realized group squared norm beyond numerical tolerance')
        new = squared.clamp_min(0).sqrt()
        old = reference.sqrt()
        ctx.save_for_backward(old, new)
        effective = torch.where(squared >= 0, delta, -reference)
        denominator = old * (new + old)
        # Exact-zero groups are separately required to stay frozen. Preserve
        # their zero contribution to the same fifteen-group denominator.
        return torch.where(old > 0, effective / denominator.clamp_min(torch.finfo(old.dtype).tiny), 0.)

    @staticmethod
    def backward(ctx, cotangent):
        old, new = ctx.saved_tensors
        denominator = (2 * old * new).clamp_min(torch.finfo(old.dtype).tiny)
        derivative = torch.where((old > 0) & (new > 0), 1 / denominator, 0.)
        return cotangent * derivative, None


class AcceleratedGroupedNormObjective(GroupedNormObjective):
    """Drop-in GPU GroupedNormObjective; same groups, rounding and zero policy.

    Inputs must be contiguous FP32/BF16. Reference setup/validation is performed
    once per checkpoint. CPU reference remains available in GroupedNormObjective.
    Raw pointers are rebuilt for each new replacement bank, so values and storage
    may change between calls. The pointer-table setup is not torch.compile-able.
    """
    block_size = 4096

    def __init__(self, parameters, groups=None, *, chunk_rows=512, zero_reference_policy='error'):
        if triton is None:
            raise RuntimeError('AcceleratedGroupedNormObjective requires Triton')
        super().__init__(parameters, groups, chunk_rows=chunk_rows,
                         zero_reference_policy=zero_reference_policy)
        self.names = tuple(name for names in self.groups.values() for name in names)
        self.bases = tuple(parameters[name] for name in self.names)
        device = self.bases[0].device
        if device.type != 'cuda' or any(v.device != device or not v.is_contiguous()
            or v.dtype not in (torch.float32, torch.bfloat16) for v in self.bases):
            raise ValueError('accelerated grouped norm requires contiguous CUDA FP32/BF16 references')
        tensor_ids, offsets, group_ids, starts, ends = [], [], [], [], []
        tensor = 0
        for group, names in enumerate(self.groups.values()):
            starts.append(len(tensor_ids))
            for name in names:
                count = triton.cdiv(parameters[name].numel(), self.block_size)
                tensor_ids.extend([tensor] * count)
                offsets.extend(i * self.block_size for i in range(count))
                group_ids.extend([group] * count)
                tensor += 1
            ends.append(len(tensor_ids))
        self.blocks = len(tensor_ids)
        self.reduction_size = triton.next_power_of_2(max(b-a for a, b in zip(starts, ends)))
        self.base_addresses = _addresses(self.bases, device)
        for name, value in dict(tensor_ids=tensor_ids, offsets=offsets, group_ids=group_ids,
            starts=starts, ends=ends, sizes=[v.numel() for v in self.bases],
            bf16=[v.dtype == torch.bfloat16 for v in self.bases]).items():
            setattr(self, name, torch.tensor(value, dtype=torch.int32, device=device))
        self.reference_vector = torch.stack(tuple(self.reference_squared.values()))

    def evaluate(self, replacements, *, edit_masks=None):
        if any(value._version != self._reference_versions[name] for name, value in self.parameters.items()):
            raise RuntimeError('frozen grouped norm reference was modified in place')
        if replacements.keys() != self.parameters.keys():
            raise ValueError('replacement mapping differs from frozen reference')
        for name in self.names:
            value, base = replacements[name], self.parameters[name]
            if (value.shape != base.shape or value.dtype != base.dtype or value.device != base.device
                or not value.is_contiguous()):
                raise ValueError('replacement shape/dtype/device/contiguity mismatch: '+name)
        if edit_masks is None:
            for name in self.frozen_parameter_names:
                value, base = replacements[name], self.parameters[name]
                if value.data_ptr() != base.data_ptr():
                    if torch.cuda.is_current_stream_capturing():
                        torch._assert_async(torch.all(value == base),'zero-reference group must remain frozen: '+name)
                    elif not torch.equal(value, base):
                        raise ValueError('zero-reference group must remain frozen: '+name)
        else:
            # Optional graph mode: freeze/edit status is live resident data.
            # Capturing the initial Python frozen-name list would incorrectly
            # assert after a zero group becomes nonzero on a later checkpoint.
            if set(edit_masks)!=set(self.names):raise ValueError('dynamic masks must cover editable norm parameters')
            consistent=[]
            for index,names in enumerate(self.groups.values()):
                active=self.reference_vector[index]>0
                for name in names:
                    mask=edit_masks[name]
                    consistent.append(mask==active)
                    torch._assert_async(mask | torch.all(replacements[name]==self.parameters[name]),
                                        'dynamic frozen group must retain exact base values: '+name)
            torch._assert_async(torch.stack(consistent).all(),'dynamic editable mask differs from current reference')
        delta, edit = _MultiTensorNorm.apply(self, *(replacements[name] for name in self.names))
        relative = _RelativeGroups.apply(delta, self.reference_vector)
        norms = (self.reference_vector + delta).clamp_min(0).sqrt().detach()
        return GroupedNormResult(relative.mean(), dict(zip(self.groups, relative.unbind())),
            dict(zip(self.groups, norms.unbind())), delta.sum(), edit.sum())
