"""Replay past minibatches without consuming model or data RNG streams."""
import hashlib
import torch


def training_batch_step(config, identity, update, recorded_step):
    policy = config.get('preservation_data', {'policy': 'recorded'})
    if policy['policy'] == 'recorded':
        return None
    if policy['policy'] != 'uniform_past':
        raise ValueError('unknown preservation data policy')
    payload = f"{policy['seed']}:{identity}:{update}".encode()
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little')
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(recorded_step + 1, (), generator=generator).item()


def cursor_at_step(stream, cursor, step):
    if 'step' in cursor:
        return cursor | dict(step=step)
    remaining = step
    for index, path in enumerate(stream.files):
        header = torch.from_file(str(path), False, 256, dtype=torch.int32)
        if header[0] != 20240520 or header[1] != 1:
            raise ValueError('invalid FineWeb shard header')
        # Upstream switches shards when pos + batch + 1 >= token_count.
        count = (int(header[2]) - 2) // stream.batch_size
        if remaining < count:
            return cursor | dict(file_index=index, pos=remaining * stream.batch_size, batches=step)
        remaining -= count
    raise ValueError('preservation batch exceeds available dataset')
