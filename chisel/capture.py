"""One task-independent trajectory writer; CIFAR and text select it in YAML."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
from chisel.recording import cpu_copy, save_atomic


class TrajectoryWriter:
    def __init__(self, root):
        self.root = Path(root)
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pending = []

    def submit(self, step, record):
        # Bound host memory while storage catches up. Copies already own their buffers.
        if len(self.pending) >= 2:
            self.pending.pop(0).result()
        self.pending.append(self.pool.submit(self.write, step, record))

    def write(self, step, record):
        path = self.root / f'step{step:06d}.pt'
        if path.exists():
            raise FileExistsError(path)
        save_atomic(record, path)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 << 20), b''):
                digest.update(block)
        receipt = dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=digest.hexdigest(), step=step)
        path.with_suffix('.ready.json').write_text(json.dumps(receipt))

    def close(self):
        try:
            for future in self.pending:
                future.result()
        finally:
            self.pool.shutdown(wait=True)


def setup():
    def hook(config, state):
        if state.extras['rank'] == 0:
            state.extras['trajectory_writer'] = TrajectoryWriter(Path(config['output_dir']) / 'trajectory')
    return hook


def record(every):
    def hook(config, state):
        if state.extras['rank'] or state.step % every or state.step >= config['train_steps'] - 1:
            return
        if 'consolidation' in state.extras:
            raise ValueError('baseline trajectory capture expects C=0')
        value = cpu_copy(dict(schema='chisel-muonh-transition-v1', config=config, step=state.step,
                              weights=state.model.state_dict(),
                              gradients={name: p.grad for name, p in state.model.named_parameters()},
                              optimizer=state.optimizer.state_dict(), next_data=state.train_batches.state_dict(),
                              activity=state.extras.get('activity', {})))
        state.extras['trajectory_writer'].submit(state.step, value)
    return hook


def close():
    def hook(config, state):
        if state.extras['rank'] == 0:
            state.extras['trajectory_writer'].close()
    return hook
