"""PR323 shard order and boundary behavior with an explicit restart cursor."""
import glob
from pathlib import Path
import torch
import torch.distributed as dist


def _load_data_shard(file: Path, pin_memory=True):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, 'magic number mismatch in the data .bin file'
    assert header[1] == 1, 'unsupported version'
    num_tokens = int(header[2])
    with file.open('rb', buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=pin_memory)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, 'number of tokens read does not match header'
    return tokens


class FinewebStream:
    """A next-batch cursor; loading skips no GPT passes and reads only its shard.

    `pos` advances before returning a batch, exactly as the upstream generator.
    A cursor at the exhausted tail is valid: the next call opens the next shard.
    """
    def __init__(self, filename_pattern, batch_size, seq_len=1024, *, rank=None,
                 world_size=None, device='cuda'):
        self.files = sorted(Path(p) for p in glob.glob(filename_pattern))
        if not self.files:
            raise FileNotFoundError(filename_pattern)
        self.rank = dist.get_rank() if rank is None else rank
        self.world_size = dist.get_world_size() if world_size is None else world_size
        if batch_size % self.world_size or batch_size // self.world_size % seq_len:
            raise ValueError('batch must divide ranks into complete sequences')
        self.batch_size, self.seq_len = batch_size, seq_len
        self.local_batch_size = batch_size // self.world_size
        self.device = torch.device(device)
        self.file_index = self.pos = self.batches = 0
        self.tokens = None

    def __iter__(self):
        return self

    def _open(self):
        if self.file_index >= len(self.files):
            raise StopIteration
        self.tokens = _load_data_shard(self.files[self.file_index], pin_memory=self.device.type == 'cuda')

    def __next__(self):
        if self.tokens is None:
            self._open()
        if self.pos + self.batch_size + 1 >= len(self.tokens):
            self.file_index += 1
            self.pos = 0
            self._open()
        buf = self.tokens[self.pos+self.rank*self.local_batch_size:][:self.local_batch_size+1]
        if len(buf) != self.local_batch_size+1:
            raise ValueError('shard too small for the prescribed batch')
        inputs = buf[:-1].to(device=self.device, dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device=self.device, dtype=torch.int64, non_blocking=True)
        self.pos += self.batch_size
        self.batches += 1
        return inputs.view(-1, self.seq_len), targets.view(-1, self.seq_len)

    def state_dict(self):
        return dict(file_names=[p.name for p in self.files], batch_size=self.batch_size,
                    seq_len=self.seq_len, rank=self.rank, world_size=self.world_size,
                    file_index=self.file_index, pos=self.pos, batches=self.batches)

    def load_state_dict(self, state):
        expected = self.state_dict()
        for key in ('file_names', 'batch_size', 'seq_len', 'rank', 'world_size'):
            if state[key] != expected[key]:
                raise ValueError(f'data cursor mismatch: {key}')
        index, pos, batches = state['file_index'], state['pos'], state['batches']
        if not 0 <= index < len(self.files) or pos < 0 or pos % self.batch_size or batches < 0:
            raise ValueError('invalid data cursor')
        self.file_index, self.pos, self.batches = index, pos, batches
        self.tokens = None


def distributed_data_generator(filename_pattern: str, batch_size: int, seq_len=1024):
    return FinewebStream(filename_pattern, batch_size, seq_len)
