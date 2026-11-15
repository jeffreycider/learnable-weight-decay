"""Capture independently initialized terminal CIFAR-10 airbench94_muon models for the Chisel bank.

One process owns one bounded seed range on one device. For every seed it sets the torch/CUDA
seeds, runs the vendored upstream ``main`` (8 epochs x 25 steps of batch 2000) on the SAME compiled
model (upstream resets the model and builds fresh optimizers inside ``main``), and freezes the
terminal state: parameters/buffers, named momentum buffers, RNG states, metrics, the training log,
config, environment and source hashes. Every seed is kept; nothing is selected by accuracy. The
only source change is one explicit line so ``main`` returns ``(tta_val_acc, optimizers)``; the
pristine upstream file, its LICENSE and the patch live in baseline/vendor/airbench. Upstream's
default 200 trials and warmup run are never executed, and nothing is downloaded.

Opt-in ``--defer-test-readouts`` (default off) applies a second pinned three-line patch that skips the
test-set readouts during training (test_loader=None, val_acc=None, tta_val_acc=None) so capture can run
before the official test set exists. The training loop, reset, optimizers, data order and RNG use are
unchanged; accuracies are stored as None with evaluation_deferred=true, never substituted or faked.
Terminal test evaluation then belongs to a later owner-run readout on the frozen artifacts.
"""
import argparse
import contextlib
import datetime
import hashlib
import io
import json
import math
import os
import platform
import subprocess
import sys
import time
import types
import weakref
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'baseline' / 'vendor' / 'airbench'
UPSTREAM_URL = 'https://github.com/KellerJordan/cifar10-airbench.git'
UPSTREAM_COMMIT = '4c1b6d1e3889b037efadcfd5c0ea65b246592362'
PRISTINE_SHA256 = '704e29251a5207d8aed8a35adf13f772ff386b90a712aaddc2a28854385b536e'
ORIGINAL_LINE = '    return tta_val_acc\n'
PATCHED_LINE = '    return tta_val_acc, optimizers\n'
DEFER_HUNKS = (('    test_loader = CifarLoader("cifar10", train=False, batch_size=2000)\n', '    test_loader = None\n'),
               ('        val_acc = evaluate(model, test_loader, tta_level=0)\n', '        val_acc = None\n'),
               ('    tta_val_acc = evaluate(model, test_loader, tta_level=2)\n', '    tta_val_acc = None\n'))
SCHEMA = 'cifar10-airbench94-muon-terminal-v1'
UPSTREAM_CONFIG = dict(epochs=8, batch_size=2000, train_images=50000, steps_per_epoch=25, total_train_steps=200,
                       whiten_bias_train_steps=75, label_smoothing=.2, optimizers=['SGD(nesterov, momentum .85)', 'Muon(momentum .6, nesterov)'],
                       tta_level=2, warmup_run=False, upstream_trials=False, accuracy_selection=False)
DATA_FILES = ('train.pt', 'test.pt')


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# ----------------------------------------------------------------------------- source provenance
def pristine_source():
    path = VENDOR / 'airbench94_muon.py'
    text = path.read_text()
    if sha256_text(text) != PRISTINE_SHA256:
        raise RuntimeError('vendored airbench94_muon.py does not match the pinned upstream hash')
    return text


def _apply_hunks(text, hunks, patch_name):
    """Apply explicit single-line hunks; fail closed unless each matches exactly once and the patch file agrees."""
    patch = (VENDOR / patch_name).read_text()
    if patch.count('\n@@') != len(hunks):
        raise RuntimeError(f'{patch_name} does not describe the {len(hunks)} expected hunk(s)')
    for original, replacement in hunks:
        if text.count(original) != 1:
            raise RuntimeError(f'upstream line not found exactly once; refusing to apply {patch_name}: {original.strip()}')
        if ('-' + original) not in patch or ('+' + replacement) not in patch:
            raise RuntimeError(f'{patch_name} does not contain the expected hunk for: {original.strip()}')
        text = text.replace(original, replacement)
    return text


def patched_source(defer_test_readouts=False):
    text = _apply_hunks(pristine_source(), ((ORIGINAL_LINE, PATCHED_LINE),), 'return_optimizers.patch')
    if defer_test_readouts:
        text = _apply_hunks(text, DEFER_HUNKS, 'defer_test_readouts.patch')
    return text


def source_inventory(defer_test_readouts=False):
    inventory = dict(upstream_url=UPSTREAM_URL, upstream_commit=UPSTREAM_COMMIT,
                     pristine_sha256=sha256_file(VENDOR / 'airbench94_muon.py'),
                     license_sha256=sha256_file(VENDOR / 'LICENSE'),
                     patch_sha256=sha256_file(VENDOR / 'return_optimizers.patch'),
                     return_patched_sha256=sha256_text(patched_source(False)),
                     patched_sha256=sha256_text(patched_source(defer_test_readouts)),
                     patch_description='main returns (tta_val_acc, optimizers) instead of tta_val_acc; no other change',
                     defer_test_readouts=bool(defer_test_readouts),
                     wrapper_sha256=sha256_file(Path(__file__).resolve()),
                     pin_sha256=sha256_file(VENDOR / 'PIN.json') if (VENDOR / 'PIN.json').exists() else None)
    if defer_test_readouts:
        inventory.update(defer_patch_sha256=sha256_file(VENDOR / 'defer_test_readouts.patch'),
                         defer_patch_description='test_loader=None; val_acc=None; tta_val_acc=None (three lines); training loop, reset, optimizers, data, RNG unchanged')
    return inventory


def load_airbench(data_dir, defer_test_readouts=False):
    """Execute the patched upstream module (never as __main__) with its loader pinned to data_dir."""
    verify_data_files(data_dir, require_test=not defer_test_readouts)
    module = types.ModuleType('airbench94_muon_patched')
    module.__file__ = str(VENDOR / 'airbench94_muon.py')
    # Dynamo resolves class/function globals through the module registry.
    sys.modules[module.__name__] = module
    argv0 = sys.argv[0]
    sys.argv[0] = module.__file__  # upstream reads sys.argv[0] at import to embed its own code
    try:
        exec(compile(patched_source(defer_test_readouts), module.__file__, 'exec'), module.__dict__)
    finally:
        sys.argv[0] = argv0
    upstream_loader = module.CifarLoader

    class PinnedCifarLoader(upstream_loader):
        """Same upstream loader contract (train.pt/test.pt dicts); the directory is pinned, never downloaded."""
        def __init__(self, path, *args, **kwargs):
            super().__init__(str(data_dir), *args, **kwargs)
    module.CifarLoader = PinnedCifarLoader
    return module


def verify_data_files(data_dir, require_test=True):
    """Hash the prepared tensors. With require_test=False (deferred readouts) only train.pt is required and
    recorded; no test hash is invented."""
    data_dir = Path(data_dir)
    names = DATA_FILES if require_test else DATA_FILES[:1]
    missing = [name for name in names if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f'prepared CIFAR tensors missing in {data_dir}: {missing}; conversion is owned elsewhere, nothing is downloaded')
    return {name: dict(sha256=sha256_file(data_dir / name), bytes=(data_dir / name).stat().st_size) for name in names}


# ----------------------------------------------------------------------------- per-seed capture
def seed_everything(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_states():
    states = dict(torch_cpu=torch.get_rng_state().clone())
    if torch.cuda.is_available():
        states['torch_cuda_all'] = [s.clone() for s in torch.cuda.get_rng_state_all()]
    return states


def named_momentum(model, optimizers):
    names = {id(p): n for n, p in model.named_parameters()}
    momentum, groups = {}, []
    for index, opt in enumerate(optimizers):
        for g_index, group in enumerate(opt.param_groups):
            hyper = {k: v for k, v in group.items() if k != 'params'}
            members = []
            for p in group['params']:
                name = names.get(id(p))
                if name is None:
                    raise RuntimeError('optimizer parameter is not a named model parameter')
                members.append(name)
                buffer = opt.state.get(p, {}).get('momentum_buffer')
                if buffer is None:
                    raise RuntimeError(f'no momentum buffer for {name}; optimizer did not step')
                if name in momentum:
                    raise RuntimeError(f'{name} appears in two optimizer groups')
                momentum[name] = buffer.detach().to('cpu').clone()
            groups.append(dict(optimizer=index, optimizer_class=type(opt).__name__, group=g_index, params=members, **hyper))
    return momentum, groups


def environment(device):
    info = dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda, hostname=platform.node(),
                device=str(device), cudnn_benchmark=bool(torch.backends.cudnn.benchmark))
    if device.type == 'cuda' and torch.cuda.is_available():
        info['device_name'] = torch.cuda.get_device_name(device)
    try:
        info['repo_commit'] = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, capture_output=True, timeout=10).stdout.strip()
    except Exception:
        info['repo_commit'] = None
    return info


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for stream in self.streams:
            stream.write(s)
        return len(s)
    def flush(self):
        for stream in self.streams:
            stream.flush()


def capture_seed(seed, model, runner, *, device, header, seen_optimizers):
    seed_everything(seed)
    log = io.StringIO()
    started = time.perf_counter()
    with contextlib.redirect_stdout(_Tee(sys.__stdout__, log)):
        result = runner(seed, model)
    wall = time.perf_counter() - started
    if not (isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], (list, tuple)) and result[1]):
        raise RuntimeError('patched main must return (tta_val_acc, optimizers)')
    deferred = bool(header['evaluation']['deferred'])
    if deferred:
        if result[0] is not None:
            raise RuntimeError('deferred-readout capture produced an accuracy; the deferral patch was not in effect')
        tta_val_acc = None  # preserved explicitly: no substitute, no NaN
    else:
        if result[0] is None or not math.isfinite(float(result[0])):
            raise RuntimeError('terminal tta_val_acc missing or nonfinite outside deferred mode')
        tta_val_acc = float(result[0])
    optimizers = list(result[1])
    for opt in optimizers:  # seen_optimizers is a WeakSet: identity of LIVE objects, no id recycling
        if opt in seen_optimizers:
            raise RuntimeError('optimizer object reused across seeds; fresh optimizers per seed are required')
        seen_optimizers.add(opt)
    momentum, groups = named_momentum(model, optimizers)
    state = {k: v.detach().to('cpu').clone() for k, v in model.state_dict().items()}
    return dict(schema=SCHEMA, seed=seed, tta_val_acc=tta_val_acc, evaluation_deferred=deferred, evaluation=dict(header['evaluation']),
                state_dict=state, momentum=momentum, optimizer_groups=groups,
                rng_after=rng_states(), metrics=dict(tta_val_acc=tta_val_acc, wall_seconds=wall, training_log=log.getvalue()),
                upstream_config=dict(UPSTREAM_CONFIG), source=header['source'], data=header['data'], environment=environment(device),
                completed_utc=utc_now())


# ----------------------------------------------------------------------------- output / manifest
def seed_path(output, seed):
    return Path(output) / f'seed_{seed:04d}.pt'


def freeze_write(path, save):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    if path.exists():
        raise FileExistsError(f'{path} already exists; frozen artifacts are never overwritten')
    save(tmp)
    os.replace(tmp, path)
    os.chmod(path, 0o444)
    return sha256_file(path)


def write_header(output, header):
    path = Path(output) / 'capture.json'
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != header:
            raise RuntimeError('resume configuration changed (capture.json differs: seeds, data, source or config)')
        return
    freeze_write(path, lambda tmp: tmp.write_text(json.dumps(header, indent=2, sort_keys=True) + '\n'))


def read_manifest(output):
    path = Path(output) / 'manifest.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    seeds = [r['seed'] for r in rows]
    if len(seeds) != len(set(seeds)):
        raise RuntimeError('manifest lists a seed twice; refusing to resume')
    for row in rows:
        file = Path(output) / row['file']
        if not file.is_file() or sha256_file(file) != row['sha256']:
            raise RuntimeError(f'manifest entry for seed {row["seed"]} does not match {file}; refusing to resume')
    return rows


def append_manifest(output, row):
    with open(Path(output) / 'manifest.jsonl', 'a') as f:
        f.write(json.dumps(row, sort_keys=True) + '\n')
        f.flush()
        os.fsync(f.fileno())


def evaluation_record(deferred):
    if deferred:
        return dict(deferred=True, test_readouts='omitted during training (reporting only); official 10K test set pending',
                    terminal_test_eval='deferred to a later owner-run readout on the frozen artifacts; training data is never used as test')
    return dict(deferred=False, test_readouts='upstream val_acc per epoch and terminal TTA on test.pt')


def run_capture(*, output, seeds, model, runner, device, data, source, extra_config=None, evaluation_deferred=False):
    """Capture every seed in ``seeds`` not already frozen. Returns the manifest rows."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    if evaluation_deferred and 'test.pt' in data:
        raise RuntimeError('deferred-readout header must not carry a test hash')
    if not evaluation_deferred and 'test.pt' not in data:
        raise RuntimeError('test.pt hash required unless --defer-test-readouts is set')
    header = dict(schema=SCHEMA, seeds=dict(start=min(seeds), end=max(seeds), count=len(seeds)), upstream_config=dict(UPSTREAM_CONFIG),
                  source=source, data=data, evaluation=evaluation_record(evaluation_deferred), config=extra_config or {}, created_utc=None)
    existing_header = output / 'capture.json'
    header['created_utc'] = json.loads(existing_header.read_text())['created_utc'] if existing_header.exists() else utc_now()
    write_header(output, header)
    rows = read_manifest(output)
    done = {r['seed'] for r in rows}
    for seed in sorted(seeds):
        if seed in done:
            continue
        if seed_path(output, seed).exists():
            raise RuntimeError(f'{seed_path(output, seed)} exists without a manifest entry (interrupted capture); remove it explicitly before resuming')
    unknown = done - set(seeds)
    if unknown:
        raise RuntimeError(f'manifest contains seeds outside this range: {sorted(unknown)}')
    seen = weakref.WeakSet()
    for seed in sorted(seeds):
        if seed in done:
            continue
        record = capture_seed(seed, model, runner, device=device, header=header, seen_optimizers=seen)
        path = seed_path(output, seed)
        digest = freeze_write(path, lambda tmp: torch.save(record, tmp))
        row = dict(seed=seed, file=path.name, sha256=digest, bytes=path.stat().st_size, tta_val_acc=record['tta_val_acc'],
                   wall_seconds=record['metrics']['wall_seconds'], completed_utc=record['completed_utc'])
        append_manifest(output, row)
        rows.append(row)
        print(json.dumps(dict(event='seed_frozen', **row)), flush=True)
    if {r['seed'] for r in rows} == set(seeds) and not (output / 'manifest.json').exists():
        summary = dict(schema=SCHEMA, header=header, rows=sorted(rows, key=lambda r: r['seed']), completed_utc=utc_now())
        freeze_write(output / 'manifest.json', lambda tmp: tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n'))
    return rows


# ----------------------------------------------------------------------------- CLI
def parse_seed_range(text, max_seed):
    start, _, end = text.partition('-')
    start, end = int(start), int(end if end else start)
    if not 0 <= start <= end <= max_seed:
        raise argparse.ArgumentTypeError(f'seed range must satisfy 0 <= start <= end <= {max_seed}')
    return list(range(start, end + 1))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--data', required=True, help='directory holding train.pt and test.pt in the upstream CifarLoader format (prepared elsewhere)')
    p.add_argument('--seeds', required=True, help='inclusive seed range, e.g. 0-12')
    p.add_argument('--max-seed', type=int, default=99)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output', required=True)
    p.add_argument('--compile-mode', default='max-autotune', help="upstream uses max-autotune; 'none' disables compilation")
    p.add_argument('--defer-test-readouts', action='store_true',
                   help='opt-in: skip per-epoch/terminal test readouts (stored as None, evaluation_deferred=true); test.pt not required')
    a = p.parse_args(argv)
    seeds = parse_seed_range(a.seeds, a.max_seed)
    device = torch.device(a.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('the upstream training loop requires CUDA (timers, loader map_location); CPU is for tests via injection only')
    torch.cuda.set_device(device)
    data = verify_data_files(a.data, require_test=not a.defer_test_readouts)
    module = load_airbench(a.data, defer_test_readouts=a.defer_test_readouts)
    model = module.CifarNet().cuda().to(memory_format=torch.channels_last)
    if a.compile_mode != 'none':
        model.compile(mode=a.compile_mode)
    module.print_columns(module.logging_columns_list, is_head=True)
    rows = run_capture(output=a.output, seeds=seeds, model=model, runner=module.main, device=device, data=data,
                       source=source_inventory(a.defer_test_readouts), evaluation_deferred=a.defer_test_readouts,
                       extra_config=dict(compile_mode=a.compile_mode, argv=sys.argv[1:], max_seed=a.max_seed, defer_test_readouts=a.defer_test_readouts))
    print(json.dumps(dict(event='range_complete', output=str(a.output), seeds=len(rows))), flush=True)


if __name__ == '__main__':
    main()
