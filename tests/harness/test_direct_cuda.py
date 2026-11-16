import pytest
import torch
from chisel.harness import run
from tests.harness.test_direct_transfer import tiny_config, assert_tree_equal

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA qualification')


def compilation_count():
    def hook(config, state):
        from torch._dynamo.utils import counters
        state.extras.setdefault('compile_counts', []).append(counters['stats']['unique_graphs'])
    return hook


def cuda_config(path, recipe):
    config = tiny_config(path, recipe)
    config['runtime'].update(device='cuda', autocast='bfloat16')
    config['compile']['enabled'] = True
    config['optimizers']['hidden']['optimizer']['compile'] = True
    config['train_steps'] = 6
    config['post_optimizer'] = [dict(name=__name__ + '.compilation_count')]
    return config


def test_compiled_zero_control_matches_baseline_and_stops_recompiling(tmp_path):
    baseline = run(cuda_config(tmp_path / 'base', 'nanogpt_muonh_40gb.yaml'))
    control = run(cuda_config(tmp_path / 'zero', 'nanogpt_direct_zero.yaml'))
    assert_tree_equal(baseline.model.state_dict(), control.model.state_dict())
    assert_tree_equal(baseline.optimizer.state_dict(), control.optimizer.base.state_dict())
    counts = control.extras['compile_counts']
    assert counts[-3] == counts[-2] == counts[-1], counts


def test_compiled_nonzero_uses_one_stable_forward_graph(tmp_path):
    result = run(cuda_config(tmp_path, 'nanogpt_direct_transfer.yaml'))
    counts = result.extras['compile_counts']
    assert counts[-3] == counts[-2] == counts[-1], counts
    assert result.optimizer.counts['lookahead_batches'] == 6
    assert any(c.norm() > 0 for c in result.optimizer.consolidated.values())
