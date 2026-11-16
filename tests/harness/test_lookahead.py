import random
import numpy as np
import pytest
import torch
from chisel.harness import RunState
from chisel.lookahead import next_gradient


class RandomStream:
    def __init__(self):
        self.step = 0
    def __next__(self):
        self.step += 1
        return torch.randn(6, 4) + random.random() + np.random.rand(), torch.randint(3, (6,))
    def state_dict(self):
        return {'step': self.step}
    def load_state_dict(self, state):
        self.step = state['step']


def config():
    return dict(microbatch_size=2, gradient_units='mean', runtime=dict(autocast=None),
                loss=dict(name='torch.nn.functional.cross_entropy', kwargs={}))


def test_preview_replays_rng_batch_and_does_not_touch_metrics():
    from chisel.training import backward
    torch.manual_seed(1)
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Dropout(.1))
    stream = RandomStream()
    state = RunState(model=model, train_batches=stream, extras=dict(
        device=torch.device('cpu'), world_size=1, forward=model, training_forwards=21, train_loss=123))
    q, metrics = next_gradient(config(), state, dict(model.named_parameters()))
    assert stream.step == 0
    assert state.extras['training_forwards'] == 21
    assert state.extras['train_loss'] == 123
    assert all(p.grad is None for p in model.parameters())
    backward(config(), state)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(q[name], parameter.grad, rtol=0, atol=0)
    assert metrics['lookahead_loss'] == state.extras['train_loss']
    assert stream.step == 1


def test_exception_rewinds_cursor_and_rng():
    model = torch.nn.Linear(4, 3)
    stream = RandomStream()
    rng = torch.get_rng_state().clone()
    state = RunState(model=model, train_batches=stream, extras=dict(
        device=torch.device('cpu'), world_size=1, forward=model))
    cfg = config()
    cfg['loss']['name'] = 'not_a_real_package.loss'
    with pytest.raises(ModuleNotFoundError):
        next_gradient(cfg, state, dict(model.named_parameters()))
    assert stream.step == 0
    assert torch.equal(rng, torch.get_rng_state())


def test_rewinding_fineweb_reuses_resident_shard(tmp_path):
    from baseline.data import FinewebStream
    file = tmp_path / 'data.bin'
    header = np.zeros(256, np.int32)
    header[:3] = [20240520, 1, 100]
    with file.open('wb') as f:
        f.write(header.tobytes())
        f.write(np.arange(100, dtype=np.uint16).tobytes())
    stream = FinewebStream(str(file), 8, 4, rank=0, world_size=1, device='cpu')
    next(stream)
    cursor = stream.state_dict()
    expected = next(stream)
    storage = stream.tokens
    stream.load_state_dict(cursor)
    assert stream.tokens is storage
    replayed = next(stream)
    for a, b in zip(expected, replayed):
        assert torch.equal(a, b)
