"""Experimental owned CPU snapshots with one completion wait per CUDA stream.

Returns only after every asynchronous D2H copy has finished. Serialization,
validation, hashing and commit durability remain the caller's synchronous work.
No tensor aliases the source; metadata, shapes and preserve-format layouts match
the ordinary recursive ``detach().to('cpu', copy=True)`` path.
"""
import copy
import torch


@torch.no_grad()
def owned_cpu_snapshot(value):
    streams = {}
    def transfer(item):
        if isinstance(item, torch.Tensor):
            item = item.detach()
            if item.device.type == 'cpu':
                return item.to(device='cpu', copy=True)
            if item.device.type != 'cuda':
                raise ValueError('snapshot supports CPU and CUDA tensors only')
            stream = torch.cuda.current_stream(item.device)
            streams[(item.device.index, stream.cuda_stream)] = stream
            target = torch.empty_like(item, device='cpu', pin_memory=True, memory_format=torch.preserve_format)
            target.copy_(item, non_blocking=True)
            return target
        if isinstance(item, dict): return {key:transfer(child) for key,child in item.items()}
        if isinstance(item, list): return [transfer(child) for child in item]
        if isinstance(item, tuple): return tuple(transfer(child) for child in item)
        return copy.deepcopy(item)
    try:
        result = transfer(value)
    finally:
        # Also complete outstanding copies before propagating a recursive error.
        for stream in streams.values(): stream.synchronize()
    return result
