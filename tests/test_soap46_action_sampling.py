"""CPU integration of the bounded live action tap, not historical-run evidence."""
import copy
import json
import sys
import types
from pathlib import Path

import pytest
import torch

from rewriter.soap46_online import Soap46Online
from rewriter.soap46_action_diagnostic import compare_basis_action, validate_action_samples
from tests.test_soap46_online import bare, initial_state


def adapter(model, output, monkeypatch, steps=(), names=(), **kwargs):
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 1)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 2)
    scope = bare(model).scope
    scope.update(SEED=0, optimizer2=types.SimpleNamespace(param_groups=[{'params': list(model.parameters())}]))
    return Soap46Online(scope, variant='outer-only', meta_lr=1e-4, output=output,
                        stop_step=5, action_diagnostic_steps=steps,
                        action_diagnostic_names=names, **kwargs)


@pytest.mark.parametrize('variant', ['inner-outer', 'outer-only'])
def test_exact_sample_precedes_refresh_and_does_not_change_action(tmp_path, monkeypatch, variant):
    torch.manual_seed(136)
    model = torch.nn.Sequential(torch.nn.Linear(7, 4, bias=False), torch.nn.Linear(4, 3, bias=False))
    control = copy.deepcopy(model)
    def no_forward(*args, **kwargs):
        raise AssertionError('diagnostics must never call GPT')
    model.forward = control.forward = no_forward
    h = adapter(model, tmp_path/'on', monkeypatch, [1], ['0.weight'])
    off = adapter(control, tmp_path/'off', monkeypatch)
    h.variant = off.variant = variant
    states = {p: initial_state(p) for p in model.parameters()}
    controls = {p: initial_state(p) for p in control.parameters()}
    group = dict(lr=.0375, mu=.85)
    calls = []
    import rewriter.soap46_action_diagnostic as diagnostic
    def sample(r, xl, xr, *, parameter_name):
        p = model[0].weight
        assert parameter_name == '0.weight'
        assert torch.equal(r.weight, p)
        assert torch.equal(r.q_left, states[p]['q_row'])
        assert torch.equal(r.q_right, states[p]['q_col'])
        assert torch.equal(r.variance, states[p]['exp_avg_sq'])
        assert xl is h.entries[p]['xl'] and xr is h.entries[p]['xr']
        out = compare_basis_action(r, xl, xr, parameter_name=parameter_name)
        calls.append((r, xl.detach().clone(), xr.detach().clone(), out))
        return out
    monkeypatch.setattr(diagnostic, 'compare_basis_action', sample)
    for step in (0, 1):
        for p, c in zip(model.parameters(), control.parameters(), strict=True):
            p.grad = torch.arange(p.numel()).reshape_as(p).float().add(1+step).div(p.numel())
            c.grad = p.grad.clone()
            rng = torch.get_rng_state().clone()
            h.propose(p, states[p], group, step, True, True)
            assert torch.equal(torch.get_rng_state(), rng)
            off.propose(c, controls[c], group, step, True, True)
            assert torch.equal(p, c)
            assert torch.equal(p.grad, c.grad)
            for key, value in states[p].items():
                other = controls[c][key]
                assert torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other
            for key in ('xl', 'xr'):
                assert torch.equal(h.entries[p][key], off.entries[c][key])
                assert h.entries[p][key].grad is off.entries[c][key].grad is None
            if step == 0:
                with torch.no_grad():
                    for key in ('xl', 'xr'):
                        h.entries[p][key].fill_(.01)
                        off.entries[c][key].fill_(.01)
    assert len(calls) == 1  # no implicit all-matrix pass, nor unscheduled step
    r, xl, xr, expected = calls[0]
    assert r is h.entries[model[0].weight]['record']  # exact pending action record
    assert r.q_left.data_ptr() != states[model[0].weight]['q_row'].data_ptr()
    states[model[0].weight]['exp_avg_sq'].zero_()
    assert torch.count_nonzero(r.variance) > 0  # immutable pre-refresh snapshot
    row = json.loads((tmp_path/'on/action_diagnostics_rank1.jsonl').read_text())
    assert row['comparisons'] == expected['comparisons']
    assert row['residuals']['left_x_norm'] == xl.double().norm().item()
    assert row['rank'] == 1 and row['world_size'] == 2 and row['action_step'] == 1
    assert row['gpt_forwards'] == row['gpt_backwards'] == 0
    assert not list((tmp_path/'off').glob('action_diagnostics*'))


def test_basis_free_selection_is_explicitly_skipped(tmp_path, monkeypatch):
    model = torch.nn.Linear(7, 4, bias=False)
    h = adapter(model, tmp_path, monkeypatch, [0], ['weight'])
    p = model.weight
    p.grad = torch.ones_like(p)
    import rewriter.soap46_action_diagnostic as diagnostic
    monkeypatch.setattr(diagnostic, 'compare_basis_action', lambda *a, **k: pytest.fail('basis-free comparison'))
    h.propose(p, initial_state(p), dict(lr=.0375, mu=.85), 0, False, False)
    row = json.loads((tmp_path/'action_diagnostics_rank1.jsonl').read_text())
    assert row['status'] == 'skipped_basis_free'


@pytest.mark.parametrize('steps,names', [([1], []), ([], ['weight']), ([5], ['weight']),
    ([-1], ['weight']), ([1, 1], ['weight']), ([1], ['weight', 'weight']),
    (list(range(33)), ['weight']), ([1], [str(i) for i in range(9)]),
    (list(range(17)), [str(i) for i in range(8)])])
def test_selection_bounds(steps, names):
    with pytest.raises(ValueError):
        validate_action_samples(steps, names, stop_step=5, residual_frame='basis', compile_map=False)


@pytest.mark.parametrize('kwargs', [dict(residual_frame='parameter'), dict(compile_map=True)])
def test_reject_unsupported_execution(tmp_path, monkeypatch, kwargs):
    with pytest.raises(ValueError, match='original eager basis-frame'):
        adapter(torch.nn.Linear(7, 4, bias=False), tmp_path, monkeypatch, [1], ['weight'], **kwargs)


def test_exact_parameter_name_required(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='exact Muon matrix'):
        adapter(torch.nn.Linear(7, 4, bias=False), tmp_path, monkeypatch, [1], ['.*'])


def test_generated_source_and_archive_include_measurement(tmp_path, monkeypatch):
    from scripts import train_soap46
    monkeypatch.setattr(sys, 'argv', ['train_soap46', '--variant', 'outer-only', '--steps', '5',
        '--output', str(tmp_path), '--generate-only', '--action-diagnostic-steps', '1', '3',
        '--action-diagnostic-names', 'blocks.0.attn.q.weight'])
    monkeypatch.setenv('RANK', '0')
    train_soap46.main()
    source = (tmp_path/'executed_record46_rank0.py').read_text()
    compile(source, '<sampled record46>', 'exec')
    assert 'action_diagnostic_steps=(1, 3)' in source
    assert "action_diagnostic_names=('blocks.0.attn.q.weight',)" in source
    original = Path('baseline/record46/upstream.py').read_text()
    assert source.count('loss.backward()') == original.count('loss.backward()') == 1
    assert source.count('model(') == original.count('model(')
    config = json.loads((tmp_path/'config.json').read_text())
    assert config['action_diagnostic_steps'] == [1, 3]
    assert config['action_diagnostic_names'] == ['blocks.0.attn.q.weight']
    for name in ('rewriter/soap46_action_diagnostic.py', 'rewriter/soap46_online.py', 'scripts/train_soap46.py'):
        assert (tmp_path/'source_record'/name).read_bytes() == Path(name).read_bytes()
    from rewriter.soap46_checkpoint import identity
    assert config['implementation_sha256'] == identity()


def test_checkpoint_identity_pins_diagnostic_source(monkeypatch):
    from rewriter.soap46_checkpoint import identity
    before = identity()
    read_bytes = Path.read_bytes
    def edited_helper(path):
        data = read_bytes(path)
        return data + b'\n# changed diagnostic\n' if path.name == 'soap46_action_diagnostic.py' else data
    monkeypatch.setattr(Path, 'read_bytes', edited_helper)
    assert identity() != before
