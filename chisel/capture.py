"""One task-independent trajectory writer; CIFAR and text select it in YAML."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import torch
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
        path = self.root / f'{step:06d}' / 'state.pt'
        if path.exists():
            raise FileExistsError(path)
        save_atomic(record, path)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 << 20), b''):
                digest.update(block)
        receipt = dict(path=str(path.resolve()), file=path.name, bytes=path.stat().st_size, sha256=digest.hexdigest(), step=step)
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


def begin(every):
    def hook(config, state):
        if state.step % every == 0:
            state.extras['capture_before'] = cpu_copy(dict(
                data=state.train_batches.state_dict(), torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []))
    return hook


def record(every):
    def hook(config, state):
        if state.extras['rank'] or state.step % every:
            return
        if 'consolidation' in state.extras:
            raise ValueError('baseline trajectory capture expects C=0')
        if 'capture_before' not in state.extras:
            raise RuntimeError('capture.begin must run before the ordinary training pass')
        hidden = state.optimizer.groups['hidden']
        selected = {id(p) for group in hidden.param_groups for p in group['params']}
        value = cpu_copy(dict(schema='chisel-muonh-transition-v2', config=config, step=state.step,
                              phase='after_backward_before_optimizer',
                              gradient_units=config['gradient_units'],
                              raw_radii={name: p.detach().norm() for name, p in state.model.named_parameters() if id(p) in selected},
                              current=state.extras['capture_before'],
                              source_hashes={name: digest for name, digest in state.extras['source_hashes'].items()
                                             if name in ('baseline/model.py', 'baseline/data.py', 'chisel/models.py', 'chisel/muonh.py', 'chisel/cifar.py')},
                              has_next_training_step=state.step + 1 < config['train_steps'],
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
