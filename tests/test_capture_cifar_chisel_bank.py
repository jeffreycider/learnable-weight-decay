"""CPU tests for scripts/capture_cifar_chisel_bank.py: source pin/patch inventory, independent seeds and
fresh optimizers via toy injection, capture schema, frozen artifacts, fail-closed manifest/resume."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch

from scripts import capture_cifar_chisel_bank as cap

UPSTREAM_CLONE = Path('/tmp/chisel-cifar-airbench/airbench94_muon.py')


# ----------------------------------------------------------------------------- source / patch inventory
def test_vendored_source_license_and_patch_inventory():
    text = cap.pristine_source()
    assert hashlib.sha256(text.encode()).hexdigest() == cap.PRISTINE_SHA256
    pin = json.loads((cap.VENDOR / 'PIN.json').read_text())
    assert pin['upstream_commit'] == cap.UPSTREAM_COMMIT == '4c1b6d1e3889b037efadcfd5c0ea65b246592362'
    for name in ('airbench94_muon.py', 'LICENSE', 'return_optimizers.patch'):
        assert cap.sha256_file(cap.VENDOR / name) == pin['files'][name]['sha256'], name
    license_text = (cap.VENDOR / 'LICENSE').read_text()
    assert 'Keller Jordan' in license_text and 'Permission is hereby granted, free of charge' in license_text
    if UPSTREAM_CLONE.exists():
        assert UPSTREAM_CLONE.read_bytes() == (cap.VENDOR / 'airbench94_muon.py').read_bytes()
    patched = cap.patched_source()
    assert hashlib.sha256(patched.encode()).hexdigest() == pin['files']['return_optimizers.patch']['patched_source_sha256']
    original, changed = text.splitlines(), patched.splitlines()
    assert len(original) == len(changed)
    diff = [(a, b) for a, b in zip(original, changed) if a != b]
    assert diff == [('    return tta_val_acc', '    return tta_val_acc, optimizers')]
    compile(patched, 'airbench94_muon_patched.py', 'exec')  # syntax
    tree = ast.parse(patched)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    returns = [n for n in ast.walk(main) if isinstance(n, ast.Return)]
    assert len(returns) == 1 and isinstance(returns[0].value, ast.Tuple) and [e.id for e in returns[0].value.elts] == ['tta_val_acc', 'optimizers']
    assert 'if __name__ == "__main__":' in patched and 'for run in range(200)' in patched  # trials only under __main__
    inventory = cap.source_inventory()
    assert inventory['pristine_sha256'] == cap.PRISTINE_SHA256 and inventory['patched_sha256'] != inventory['pristine_sha256']
    assert inventory['upstream_commit'] == cap.UPSTREAM_COMMIT and inventory['pin_sha256'] == cap.sha256_file(cap.VENDOR / 'PIN.json')


def test_patch_fails_closed_on_tampered_source_or_patch(monkeypatch, tmp_path):
    vendor = tmp_path / 'vendor'
    vendor.mkdir()
    for name in ('airbench94_muon.py', 'LICENSE', 'return_optimizers.patch'):
        (vendor / name).write_bytes((cap.VENDOR / name).read_bytes())
    monkeypatch.setattr(cap, 'VENDOR', vendor)
    cap.patched_source()
    (vendor / 'return_optimizers.patch').write_text('--- a\n+++ b\n@@ -1 +1 @@\n-    return tta_val_acc\n+    return tta_val_acc, model\n')
    with pytest.raises(RuntimeError, match='expected hunk'):
        cap.patched_source()
    (vendor / 'airbench94_muon.py').write_text((vendor / 'airbench94_muon.py').read_text() + '\n# tamper\n')
    with pytest.raises(RuntimeError, match='pinned upstream hash'):
        cap.pristine_source()


def test_data_files_required_and_never_downloaded(tmp_path):
    with pytest.raises(FileNotFoundError, match='nothing is downloaded'):
        cap.verify_data_files(tmp_path)
    (tmp_path / 'train.pt').write_bytes(b'x'); (tmp_path / 'test.pt').write_bytes(b'yy')
    info = cap.verify_data_files(tmp_path)
    assert set(info) == {'train.pt', 'test.pt'} and info['test.pt']['bytes'] == 2


@pytest.mark.skipif(importlib.util.find_spec('torchvision') is None, reason='upstream imports torchvision')
def test_patched_module_loads_without_running_trials(tmp_path):
    (tmp_path / 'train.pt').write_bytes(b'x'); (tmp_path / 'test.pt').write_bytes(b'y')
    module = cap.load_airbench(tmp_path)
    assert module.CifarLoader.__name__ == 'PinnedCifarLoader' and callable(module.main) and hasattr(module, 'CifarNet')


# ----------------------------------------------------------------------------- toy injection
class ToyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, bias=False)
        self.norm = torch.nn.BatchNorm2d(4)
        self.head = torch.nn.Linear(4, 10, bias=False)

    def reset(self):
        for m in self.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()


class ToyRunner:
    """Mimics the patched upstream contract: reset model, fresh optimizers, train, return (acc, optimizers)."""
    def __init__(self, accuracy=lambda seed: .9, reuse=False):
        self.calls = []; self.accuracy = accuracy; self.reuse = reuse; self.last = None

    def __call__(self, run, model):
        self.calls.append(run)
        model.reset()
        if self.reuse and self.last is not None:
            optimizers = self.last
        else:
            optimizers = [torch.optim.SGD([model.conv.weight, model.norm.bias], lr=.1, momentum=.85, nesterov=True),
                          torch.optim.SGD([model.head.weight], lr=.5, momentum=.6)]
        x = torch.randn(8, 3, 8, 8)
        loss = model.head(model.norm(model.conv(x)).mean((2, 3))).logsumexp(-1).mean()
        loss.backward()
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)
        acc = self.accuracy(run)
        print(f'| {run} | toy epoch | tta {"" if acc is None else format(acc, ".4f")} |')
        self.last = optimizers
        return acc, optimizers


def _header_inputs(deferred=False):
    data = {'train.pt': dict(sha256='a' * 64, bytes=1)}
    if not deferred:
        data['test.pt'] = dict(sha256='b' * 64, bytes=1)
    return dict(data=data, source=cap.source_inventory(deferred), evaluation_deferred=deferred)


# ----------------------------------------------------------------------------- deferred test readouts (opt-in)
def test_defer_patch_inventory_source_ast_and_default_unchanged():
    pin = json.loads((cap.VENDOR / 'PIN.json').read_text())
    entry = pin['files']['defer_test_readouts.patch']
    assert cap.sha256_file(cap.VENDOR / 'defer_test_readouts.patch') == entry['sha256']
    base, deferred = cap.patched_source(False), cap.patched_source(True)
    assert hashlib.sha256(deferred.encode()).hexdigest() == entry['patched_source_sha256']
    assert cap.patched_source() == base  # default path unchanged by the option
    diff = [(a, b) for a, b in zip(base.splitlines(), deferred.splitlines()) if a != b]
    assert len(base.splitlines()) == len(deferred.splitlines()) and diff == [
        ('    test_loader = CifarLoader("cifar10", train=False, batch_size=2000)', '    test_loader = None'),
        ('        val_acc = evaluate(model, test_loader, tta_level=0)', '        val_acc = None'),
        ('    tta_val_acc = evaluate(model, test_loader, tta_level=2)', '    tta_val_acc = None')]
    compile(deferred, 'airbench94_muon_deferred.py', 'exec')
    tree = ast.parse(deferred)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    calls = [n.func.id for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert 'evaluate' not in calls and calls.count('CifarLoader') == 1  # only the train loader is built
    train_loader_call = next(n for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'CifarLoader')
    assert any(k.arg == 'train' and k.value.value is True for k in train_loader_call.keywords)
    # Training trajectory untouched: everything in main except the three lines is byte-identical.
    assert 'F.cross_entropy(outputs, labels, label_smoothing=0.2, reduction="sum").backward()' in deferred
    assert 'model.reset()' in deferred and 'optimizers = [optimizer1, optimizer2]' in deferred
    inventory = cap.source_inventory(True)
    assert inventory['defer_test_readouts'] is True and inventory['defer_patch_sha256'] == entry['sha256']
    assert inventory['patched_sha256'] == entry['patched_source_sha256'] and inventory['return_patched_sha256'] == cap.source_inventory(False)['patched_sha256']
    assert cap.source_inventory(False)['defer_test_readouts'] is False and 'defer_patch_sha256' not in cap.source_inventory(False)


def test_defer_mode_train_only_data_and_never_downloads(tmp_path):
    (tmp_path / 'train.pt').write_bytes(b'x')
    with pytest.raises(FileNotFoundError, match='test.pt'):
        cap.verify_data_files(tmp_path)
    info = cap.verify_data_files(tmp_path, require_test=False)
    assert set(info) == {'train.pt'}


def test_defer_mode_capture_preserves_none_and_resume_modes_fail_closed(tmp_path):
    out = tmp_path / 'defer'
    runner = ToyRunner(accuracy=lambda seed: None)
    rows = cap.run_capture(output=out, seeds=[0, 1], model=ToyNet(), runner=runner, device='cpu', **_header_inputs(deferred=True))
    assert [r['tta_val_acc'] for r in rows] == [None, None]
    record = torch.load(out / 'seed_0001.pt', weights_only=False)
    assert record['tta_val_acc'] is None and record['metrics']['tta_val_acc'] is None and record['evaluation_deferred'] is True
    assert record['evaluation']['deferred'] is True and 'test.pt' not in record['data'] and record['data']['train.pt']['sha256'] == 'a' * 64
    assert record['source']['defer_test_readouts'] is True and 'tta ' in record['metrics']['training_log']
    assert set(record['momentum']) == {'conv.weight', 'norm.bias', 'head.weight'} and not os.access(out / 'seed_0000.pt', os.W_OK)
    header = json.loads((out / 'capture.json').read_text())
    assert header['evaluation']['deferred'] is True and 'test.pt' not in header['data']
    assert json.loads((out / 'manifest.json').read_text())['rows'][0]['tta_val_acc'] is None
    # Seed independence holds in deferred mode too: seed 1 alone == seed 1 after seed 0.
    cap.run_capture(output=tmp_path / 'solo', seeds=[1], model=ToyNet(), runner=ToyRunner(accuracy=lambda s: None), device='cpu', **_header_inputs(deferred=True))
    solo = torch.load(tmp_path / 'solo' / 'seed_0001.pt', weights_only=False)
    for key in record['state_dict']:
        torch.testing.assert_close(solo['state_dict'][key], record['state_dict'][key], rtol=0, atol=0)
    # Default (non-deferred) resume of a deferred directory refuses, and vice versa.
    with pytest.raises(RuntimeError, match='resume configuration changed'):
        cap.run_capture(output=out, seeds=[0, 1], model=ToyNet(), runner=ToyRunner(), device='cpu', **_header_inputs(deferred=False))
    plain = tmp_path / 'plain'
    cap.run_capture(output=plain, seeds=[0], model=ToyNet(), runner=ToyRunner(), device='cpu', **_header_inputs(deferred=False))
    with pytest.raises(RuntimeError, match='resume configuration changed'):
        cap.run_capture(output=plain, seeds=[0], model=ToyNet(), runner=ToyRunner(accuracy=lambda s: None), device='cpu', **_header_inputs(deferred=True))


def test_defer_mode_guards_against_fake_or_missing_accuracy(tmp_path):
    # Deferred header but the runner still produced an accuracy: the patch was not in effect.
    with pytest.raises(RuntimeError, match='deferral patch was not in effect'):
        cap.run_capture(output=tmp_path / 'g1', seeds=[0], model=ToyNet(), runner=ToyRunner(), device='cpu', **_header_inputs(deferred=True))
    assert not (tmp_path / 'g1' / 'seed_0000.pt').exists()
    # Default mode must have a finite accuracy: None and NaN both refuse.
    with pytest.raises(RuntimeError, match='missing or nonfinite'):
        cap.run_capture(output=tmp_path / 'g2', seeds=[0], model=ToyNet(), runner=ToyRunner(accuracy=lambda s: None), device='cpu', **_header_inputs())
    with pytest.raises(RuntimeError, match='missing or nonfinite'):
        cap.run_capture(output=tmp_path / 'g3', seeds=[0], model=ToyNet(), runner=ToyRunner(accuracy=lambda s: float('nan')), device='cpu', **_header_inputs())
    # Header/data consistency: deferred headers carry no test hash; default headers require one.
    bad = _header_inputs(deferred=True); bad['data']['test.pt'] = dict(sha256='b' * 64, bytes=1)
    with pytest.raises(RuntimeError, match='must not carry a test hash'):
        cap.run_capture(output=tmp_path / 'g4', seeds=[0], model=ToyNet(), runner=ToyRunner(accuracy=lambda s: None), device='cpu', **bad)
    missing = _header_inputs(); del missing['data']['test.pt']
    with pytest.raises(RuntimeError, match='test.pt hash required'):
        cap.run_capture(output=tmp_path / 'g5', seeds=[0], model=ToyNet(), runner=ToyRunner(), device='cpu', **missing)


def test_independent_seeds_fresh_optimizers_schema_and_no_selection(tmp_path):
    model = ToyNet()
    runner = ToyRunner(accuracy=lambda seed: .01 if seed == 1 else .9)
    rows = cap.run_capture(output=tmp_path / 'a', seeds=[0, 1, 2], model=model, runner=runner, device='cpu', **_header_inputs())
    assert runner.calls == [0, 1, 2] and [r['seed'] for r in rows] == [0, 1, 2]
    records = {s: torch.load(tmp_path / 'a' / f'seed_{s:04d}.pt', weights_only=False) for s in (0, 1, 2)}
    # No accuracy-based selection: the .01 seed is frozen like the others.
    assert records[1]['tta_val_acc'] == .01 and rows[1]['tta_val_acc'] == .01
    # Independent initialization: different seeds give different parameters AND different momentum.
    assert not torch.equal(records[0]['state_dict']['conv.weight'], records[1]['state_dict']['conv.weight'])
    assert not torch.equal(records[0]['momentum']['conv.weight'], records[2]['momentum']['conv.weight'])
    # Seed independence from order: seed 1 captured alone equals seed 1 captured after seed 0.
    alone = cap.run_capture(output=tmp_path / 'b', seeds=[1], model=ToyNet(), runner=ToyRunner(), device='cpu', **_header_inputs())
    solo = torch.load(tmp_path / 'b' / 'seed_0001.pt', weights_only=False)
    for key in records[1]['state_dict']:
        torch.testing.assert_close(solo['state_dict'][key], records[1]['state_dict'][key], rtol=0, atol=0)
    for key in records[1]['momentum']:
        torch.testing.assert_close(solo['momentum'][key], records[1]['momentum'][key], rtol=0, atol=0)
    assert torch.equal(solo['rng_after']['torch_cpu'], records[1]['rng_after']['torch_cpu'])
    # Schema.
    r = records[2]
    assert r['schema'] == cap.SCHEMA and r['seed'] == 2 and set(r['state_dict']) == set(model.state_dict())
    assert all(v.device.type == 'cpu' for v in r['state_dict'].values())
    assert set(r['momentum']) == {'conv.weight', 'norm.bias', 'head.weight'}
    assert [g['optimizer_class'] for g in r['optimizer_groups']] == ['SGD', 'SGD'] and r['optimizer_groups'][0]['params'] == ['conv.weight', 'norm.bias']
    assert r['optimizer_groups'][0]['momentum'] == .85 and r['optimizer_groups'][1]['lr'] == .5
    assert r['upstream_config']['total_train_steps'] == 200 and r['upstream_config']['accuracy_selection'] is False
    assert r['source']['pristine_sha256'] == cap.PRISTINE_SHA256 and r['data']['train.pt']['sha256'] == 'a' * 64
    assert 'toy epoch' in r['metrics']['training_log'] and r['metrics']['wall_seconds'] >= 0 and r['environment']['torch'] == torch.__version__
    # Frozen files and manifests.
    for name in ('seed_0000.pt', 'capture.json', 'manifest.json'):
        assert not os.access(tmp_path / 'a' / name, os.W_OK)
    manifest = json.loads((tmp_path / 'a' / 'manifest.json').read_text())
    assert [row['seed'] for row in manifest['rows']] == [0, 1, 2] and manifest['header']['seeds'] == dict(start=0, end=2, count=3)
    for row in manifest['rows']:
        assert cap.sha256_file(tmp_path / 'a' / row['file']) == row['sha256']


def test_reused_optimizer_or_wrong_return_fails_closed(tmp_path):
    with pytest.raises(RuntimeError, match='fresh optimizers'):
        cap.run_capture(output=tmp_path / 'r', seeds=[0, 1], model=ToyNet(), runner=ToyRunner(reuse=True), device='cpu', **_header_inputs())
    def tta_only(run, model):
        return .9
    with pytest.raises(RuntimeError, match=r'\(tta_val_acc, optimizers\)'):
        cap.run_capture(output=tmp_path / 'w', seeds=[0], model=ToyNet(), runner=tta_only, device='cpu', **_header_inputs())
    assert not (tmp_path / 'w' / 'seed_0000.pt').exists()


def test_resume_skips_frozen_seeds_and_fails_closed_on_tamper_orphan_duplicate_or_config_change(tmp_path):
    out = tmp_path / 'c'
    first = ToyRunner()
    # Interrupt after three seeds: a runner that raises on seed 3.
    def interrupted(run, model):
        if run == 3:
            raise RuntimeError('simulated interruption')
        return first(run, model)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        cap.run_capture(output=out, seeds=list(range(5)), model=ToyNet(), runner=interrupted, device='cpu', **_header_inputs())
    assert first.calls == [0, 1, 2] and not (out / 'manifest.json').exists()
    second = ToyRunner()
    rows = cap.run_capture(output=out, seeds=list(range(5)), model=ToyNet(), runner=second, device='cpu', **_header_inputs())
    assert second.calls == [3, 4] and [r['seed'] for r in rows] == [0, 1, 2, 3, 4] and (out / 'manifest.json').exists()
    header = json.loads((out / 'capture.json').read_text())
    third = ToyRunner()
    assert cap.run_capture(output=out, seeds=list(range(5)), model=ToyNet(), runner=third, device='cpu', **_header_inputs()) and third.calls == []
    assert json.loads((out / 'capture.json').read_text()) == header  # created_utc preserved on resume
    # Range/config changes fail closed.
    with pytest.raises(RuntimeError, match='resume configuration changed'):
        cap.run_capture(output=out, seeds=list(range(6)), model=ToyNet(), runner=ToyRunner(), device='cpu', **_header_inputs())
    other = _header_inputs(); other['data']['train.pt']['sha256'] = 'c' * 64
    with pytest.raises(RuntimeError, match='resume configuration changed'):
        cap.run_capture(output=out, seeds=list(range(5)), model=ToyNet(), runner=ToyRunner(), device='cpu', **other)
    # Orphan artifact without a manifest line.
    orphan = tmp_path / 'd'
    cap.run_capture(output=orphan, seeds=[0, 1], model=ToyNet(), runner=ToyRunner(), device='cpu', **_header_inputs())
    lines = (orphan / 'manifest.jsonl').read_text().splitlines()
    (orphan / 'manifest.jsonl').write_text(lines[0] + '\n')
    with pytest.raises(RuntimeError, match='without a manifest entry'):
        cap.run_capture(output=orphan, seeds=[0, 1], model=ToyNet(), runner=ToyRunner(), device='cpu', **_header_inputs())
    # Duplicate manifest line.
    (orphan / 'manifest.jsonl').write_text(lines[0] + '\n' + lines[0] + '\n')
    with pytest.raises(RuntimeError, match='seed twice'):
        cap.run_capture(output=orphan, seeds=[0, 1], model=ToyNet(), runner=ToyRunner(), device='cpu', **_header_inputs())
    # Tampered frozen artifact.
    (orphan / 'manifest.jsonl').write_text('\n'.join(lines) + '\n')
    target = orphan / 'seed_0001.pt'
    os.chmod(target, 0o644)
    with open(target, 'ab') as f:
        f.write(b'!')
    with pytest.raises(RuntimeError, match='does not match'):
        cap.run_capture(output=orphan, seeds=[0, 1], model=ToyNet(), runner=ToyRunner(), device='cpu', **_header_inputs())


def test_seed_range_parsing_and_cli_refuses_cpu(tmp_path):
    assert cap.parse_seed_range('0-12', 99) == list(range(13)) and cap.parse_seed_range('99', 99) == [99]
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        cap.parse_seed_range('90-100', 99)
    with pytest.raises(argparse.ArgumentTypeError):
        cap.parse_seed_range('5-3', 99)
    (tmp_path / 'train.pt').write_bytes(b'x'); (tmp_path / 'test.pt').write_bytes(b'y')
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match='requires CUDA'):
            cap.main(['--data', str(tmp_path), '--seeds', '0-0', '--device', 'cpu', '--output', str(tmp_path / 'o')])


def test_dynamic_module_registered_for_compiler_guards(tmp_path,monkeypatch):
    import sys
    # Isolate registration from optional local torchvision availability.
    monkeypatch.setattr(cap,'verify_data_files',lambda *a,**kw:{})
    monkeypatch.setattr(cap,'patched_source',lambda *a,**kw:'class CifarLoader:\n    pass\n')
    module=cap.load_airbench(tmp_path)
    assert sys.modules[module.__name__] is module
    assert module.CifarLoader.__module__==cap.__name__  # Wrapper subclass.


def test_cli_completion_print_does_not_pass_flush_to_json():
    import ast,inspect
    tree=ast.parse(inspect.getsource(cap.main))
    dumps=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='dumps']
    assert dumps and all(k.arg!='flush' for n in dumps for k in n.keywords)
