"""Optional batched host reads of already-computed scalar tensors.

Keep every original reduction and its dtype/order unchanged; only defer the
host reads until all scalar results exist. This is useful for observer metadata
that currently calls float(tensor) once per parameter. No cached/learned values
and no runtime improvement claim: GPU qualification belongs at the caller.
"""
from collections import defaultdict
import torch


@torch.no_grad()
def scalar_dict_to_host(values):
    """Return the same Python floats as individual float(tensor) calls.

    One transfer per (device,dtype), preserving input key order. The caller must
    retain original dependency/order of the reductions that produced values.
    This helper does not reschedule or recompute them. Values must be scalar
    real tensors; NaN/Inf and signed zero are preserved, as by float(tensor).
    """
    groups=defaultdict(list)
    for key,value in values.items():
        if not isinstance(value,torch.Tensor) or value.numel()!=1 or value.is_complex():
            raise ValueError('one-element real tensors required')
        groups[(value.device,value.dtype)].append((key,value.detach().reshape(())))
    result={}
    for entries in groups.values():
        scalars=torch.stack([v for _,v in entries]).cpu().tolist()
        result.update((key,float(value)) for (key,_),value in zip(entries,scalars))
    return {key:result[key] for key in values}
