"""Occasional full raw-byte hashes; never evaluates the model."""
import hashlib
import json
import torch


@torch.no_grad()
def model_fingerprint(model):
    digest = hashlib.sha256()
    tensors, total_bytes = {}, 0
    # Enumerate aliases once, matching model parameter inventory/norm accounting.
    named = [('parameter', name, value) for name, value in model.named_parameters()]
    named += [('buffer', name, value) for name, value in model.named_buffers()]
    for kind, name, value in named:
        metadata = json.dumps([kind, name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode()
        # NumPy cannot directly expose BF16; a byte view preserves every bit.
        cpu = value.detach().contiguous().cpu()
        raw = memoryview(cpu.reshape(-1).view(torch.uint8).numpy())
        tensor_digest = hashlib.sha256(raw).hexdigest()
        digest.update(len(metadata).to_bytes(8, 'little'))
        digest.update(metadata)
        digest.update(raw)
        tensors[f'{kind}:{name}'] = tensor_digest
        total_bytes += raw.nbytes
    return dict(sha256=digest.hexdigest(), tensor_sha256=tensors, bytes=total_bytes)
