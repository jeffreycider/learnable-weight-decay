"""One task-independent trajectory writer; CIFAR and text select it in YAML."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import torch
import random
import numpy as np
import shutil
import time
import torch.distributed as dist
from chisel.recording import cpu_copy, save_atomic


def wait_for_space(minimum_free_bytes, poll_seconds=10):
    """Pause every training rank together while verified copies reclaim local space."""
    def hook(config, state):
        while True:
            available = state.extras['rank'] == 0 and shutil.disk_usage(config['output_dir']).free >= minimum_free_bytes
            ready = torch.tensor(int(available), device=state.extras['device'])
            if state.extras['world_size'] > 1:
                dist.broadcast(ready, src=0)
            if ready.item():
                return
            if state.extras['rank'] == 0:
                print('Waiting for verified recording copies to reclaim local space.', flush=True)
            time.sleep(poll_seconds)
    return hook


class TrajectoryWriter:
    def __init__(self, root, filename="state.pt"):
        self.root, self.filename = Path(root), filename
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pending = []

    def submit(self, step, record):
        # Bound host memory while storage catches up. Copies already own their buffers.
        if len(self.pending) >= 2:
            self.pending.pop(0).result()
        self.pending.append(self.pool.submit(self.write, step, record))

    def write(self, step, record):
        path = self.root / f'{step:06d}' / self.filename
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
        if 'activation_records' in state.extras:
            state.extras['activation_writer'] = TrajectoryWriter(Path(config['output_dir']) / 'trajectory', f'activation_rank{state.extras["rank"]:04d}.pt')
        if state.extras['rank'] == 0:
            state.extras['trajectory_writer'] = TrajectoryWriter(Path(config['output_dir']) / 'trajectory')
    return hook


def begin(every):
    def hook(config, state):
        if 'activation_records' in state.extras:
            state.extras['activation_records'].begin(state.step % every == 0)
        if state.step % every == 0:
            state.extras['capture_before'] = cpu_copy(dict(
                data=state.train_batches.state_dict(), torch_rng=torch.get_rng_state(),
                python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []))
    return hook


def record(every):
    def hook(config, state):
        if state.step % every:
            return
        if 'activation_records' in state.extras:
            samples = state.extras['activation_records'].snapshot(state.extras['rank'], state.extras['world_size'])
            samples['current'] = state.extras['capture_before']
            samples['gradient_units'] = config['gradient_units']
            samples['adjoint_reduction'] = 'local_batch_mean' if config['gradient_units'] == 'mean' else 'sum'
            state.extras['activation_writer'].submit(state.step, samples)
        if state.extras['rank']:
            return
        if 'consolidation' in state.extras:
            raise ValueError('baseline trajectory capture expects C=0')
        if 'capture_before' not in state.extras:
            raise RuntimeError('capture.begin must run before the ordinary training pass')
        hidden = state.optimizer.groups['hidden']
        selected = {id(p) for group in hidden.param_groups for p in group['params']}
        value = cpu_copy(dict(schema='chisel-muonh-transition-v2', config=config, step=state.step,
                              phase='after_backward_before_optimizer',
                              activation_shards=state.extras['world_size'] if 'activation_records' in state.extras else 0,
                              gradient_units=config['gradient_units'],
                              raw_radii={name: p.detach().norm() for name, p in state.model.named_parameters() if id(p) in selected},
                              current=state.extras['capture_before'],
                              source_hashes={name: digest for name, digest in state.extras['source_hashes'].items()
                                             if name in ('baseline/model.py', 'baseline/data.py', 'chisel/models.py', 'chisel/muonh.py', 'chisel/cifar.py', 'chisel/data_identity.py', 'chisel/fused_relu_square.py')},
                              has_next_training_step=state.step + 1 < config['train_steps'],
                              weights=state.model.state_dict(),
                              gradients={name: p.grad for name, p in state.model.named_parameters()},
                              optimizer=state.optimizer.state_dict(), next_data=state.train_batches.state_dict(),
                              activity=state.extras.get('activity', {})))
        state.extras['trajectory_writer'].submit(state.step, value)
    return hook


def close():
    def hook(config, state):
        if 'activation_writer' in state.extras:
            state.extras['activation_writer'].close()
            state.extras['activation_records'].close()
        if state.extras['rank'] == 0:
            state.extras['trajectory_writer'].close()
    return hook
