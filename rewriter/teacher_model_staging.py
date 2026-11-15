"""Optional CPU staging for a strict FP32 frozen teacher loader.

The existing loader promotes floating parameters with ``.float()`` before its
device copy. Doing that after pinning the raw checkpoint can allocate a new,
pageable FP32 tensor. This helper performs the same conversion directly into
the worker's destination allocation. Buffers retain their checkpoint dtype.

No checkpoint, provenance, engine, or learned observation is modified here.
The verified-reader owner must validate the source record before calling and
issue the usual copied-tensor provenance after staging. This helper is not
enabled in a trainer or graph stager by importing it.
"""
import torch


@torch.no_grad()
def stage_fp32_teacher_model(model, *, parameter_names, pin_memory=False):
    """Copy CPU model tensors, promoting exactly the declared parameters.

    ``parameter_names`` must come from the strict FP32 destination engine, not
    a heuristic over names or tensor ranks. The engine's original validation
    and verified teacher/head binding must still run on consumption.
    """
    names=frozenset(parameter_names)
    if not names or not names.issubset(model):
        raise ValueError('explicit nonempty parameter names must exist in the model')
    # Validate every source before allocation/copy; never submit CUDA work from
    # the CPU staging thread or accidentally detach a live training graph.
    for name,value in model.items():
        if not isinstance(value,torch.Tensor) or value.device.type!='cpu':
            raise ValueError('teacher staging requires CPU tensors')
        if value.layout!=torch.strided or value.requires_grad:
            raise ValueError('teacher staging requires detached strided tensors')
        if name in names and not value.is_floating_point():
            raise ValueError('teacher parameters must be floating point')
    copied={}
    for name,value in model.items():
        dtype=torch.float32 if name in names else value.dtype
        target=torch.empty_like(value,dtype=dtype,device='cpu',
                                pin_memory=pin_memory,memory_format=torch.preserve_format)
        target.copy_(value)
        copied[name]=target
    return copied
