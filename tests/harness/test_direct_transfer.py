import copy
import torch
from chisel.harness import run, resolve
from chisel.lookahead import next_gradient
from tests.harness.test_integration import source_config


def tiny_config(tmp_path, recipe='nanogpt_direct_transfer.yaml'):
    source = source_config(tmp_path)
    config = resolve('recipes/' + recipe)
    for key in ('train_steps', 'runtime', 'compile', 'model', 'data', 'optimizers', 'microbatch_size'):
        config[key] = source[key]
    config['output_dir'] = str(tmp_path / 'run')
    config['post_optimizer'] = []
    return config


def assert_tree_equal(a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree_equal(x, y)
    else:
        assert a == b


def test_zero_transfer_is_bitwise_baseline_including_optimizer_and_rng(tmp_path):
    baseline = tiny_config(tmp_path / 'base', 'nanogpt_muonh_40gb.yaml')
    plain = run(baseline)
    rng = torch.get_rng_state().clone()
    config = tiny_config(tmp_path / 'direct', 'nanogpt_direct_zero.yaml')
    direct = run(config)
    assert_tree_equal(plain.model.state_dict(), direct.model.state_dict())
    assert_tree_equal(plain.optimizer.state_dict(), direct.optimizer.base.state_dict())
    assert_tree_equal(plain.train_batches.state_dict(), direct.train_batches.state_dict())
    assert torch.equal(torch.get_rng_state(), rng)
    assert direct.optimizer.counts['lookahead_batches'] == 5
    assert direct.extras['training_forwards'] == plain.extras['training_forwards'] == 10
    assert all(torch.count_nonzero(c) == 0 for c in direct.optimizer.consolidated.values())


def test_nonzero_transfer_and_terminal_identity(tmp_path):
    config = tiny_config(tmp_path)
    result = run(config)
    opt = result.optimizer
    assert opt.counts['ordinary_batches'] == 5
    assert opt.counts['lookahead_batches'] == 4
    assert result.train_batches.step == 5
    assert any(c.norm() > 0 for c in opt.consolidated.values())
    for name, p in opt.parameters.items():
        torch.testing.assert_close(p, opt.raw[name] + opt.consolidated[name], atol=0, rtol=0)
    for optimizer in opt.base.groups.values():
        for state in optimizer.state.values():
            if 'step' in state:
                assert state['step'] == 5
    metrics = result.extras['metrics']
    assert metrics['model_backward_microbatches'] == 18
    assert metrics['model_training_tokens'] == 72
    assert metrics['edit_size'] == 0
    assert metrics['lookahead_loss'] is None


def test_resume_preserves_raw_consolidated_optimizer_and_cursor(tmp_path):
    config = tiny_config(tmp_path / 'full')
    config['post_optimizer'] = [dict(name='chisel.recording.checkpoint', hyperparams=dict(every=2))]
    full = run(config)
    prefix = copy.deepcopy(config)
    prefix.update(stop_step=2, output_dir=str(tmp_path / 'prefix'))
    run(prefix)
    resume = copy.deepcopy(config)
    resume.update(resume=str(tmp_path / 'prefix/latest_rank0.pt'), output_dir=str(tmp_path / 'resumed'))
    restored = run(resume)
    assert_tree_equal(full.model.state_dict(), restored.model.state_dict())
    assert_tree_equal(full.optimizer.state_dict(), restored.optimizer.state_dict())
    assert_tree_equal(full.train_batches.state_dict(), restored.train_batches.state_dict())


def checked_lookahead(config, state, parameters):
    # The independent baseline stores complete successor states, including AdamW.
    expected = torch.load(config['expected_successors'], weights_only=False)[state.step]
    assert_tree_equal(expected, state.model.state_dict())
    return next_gradient(config, state, parameters)


def capture_successor():
    def hook(config, state):
        state.extras.setdefault('successors', []).append(copy.deepcopy(state.model.state_dict()))
    return hook


def test_preview_uses_complete_successor_not_only_hidden_matrices(tmp_path):
    baseline = tiny_config(tmp_path / 'base', 'nanogpt_muonh_40gb.yaml')
    baseline['post_optimizer'] = [dict(name=__name__ + '.capture_successor')]
    plain = run(baseline)
    path = tmp_path / 'successors.pt'
    torch.save(plain.extras['successors'], path)
    config = tiny_config(tmp_path / 'direct', 'nanogpt_direct_zero.yaml')
    config['expected_successors'] = str(path)
    config['direct_transfer']['lookahead'] = __name__ + '.checked_lookahead'
    run(config)
