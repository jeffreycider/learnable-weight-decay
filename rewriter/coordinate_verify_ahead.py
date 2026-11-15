"""Optional immutable GPT verification overlap for the owner-balanced trainer.

Only verification stamps are prefetched. Tensor loading, sampling, model math,
and checkpoint ownership remain with the ordinary trainer/readers.
"""
import time
from .verify_prefetch import VerifyAhead


class CoordinateVerifyAhead:
    def __init__(self, mode, contexts, schedule, max_updates):
        if mode not in ('off', 'depth1_v1'):
            raise ValueError('unsupported verification lookahead')
        self.mode, self.contexts, self.schedule = mode, contexts, schedule
        self.max_updates = max_updates
        self.worker = None
        self.next_batch = None

    def __enter__(self):
        return self

    def wait(self, batch, state_id):
        if self.mode == 'off':
            return 0.
        if self.worker is None:
            # Resume from the actual committed sampler clock, never batch zero.
            ids = [self.schedule.visit(i, 'gpt')[0].local_id
                   for i in range(batch, self.max_updates)]
            if not ids or ids[0] != state_id:
                raise ValueError('verification schedule differs from current visit')
            self.worker = VerifyAhead(self.contexts.reader, self.contexts.cache,
                                      roles=('train',), depth=1)
            self.worker.schedule(ids)
            self.next_batch = batch
        if batch != self.next_batch or self.schedule.visit(batch, 'gpt')[0].local_id != state_id:
            raise ValueError('verification consumer changed sampler order')
        start = time.perf_counter()
        self.worker.wait(state_id)
        self.next_batch += 1
        return time.perf_counter() - start

    def __exit__(self, *exc):
        if self.worker is not None:
            self.worker.close()
