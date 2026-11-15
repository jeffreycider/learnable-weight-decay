"""Actual continuation updates with unchanged math and coalesced observations.

This isolated qualifier substitutes only the GPT bank reader. It retains the
same file/SHA checks, local/global identities, cache binding, constant cache,
actor, both Adam owners, and finite student passes. No active trainer is patched.
"""
from contextlib import contextmanager
from pathlib import Path
import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from rewriter.coalesced_state_reader import CoalescedStateBankReader
from rewriter.coalesced_snapshot_store import BankedObservationReader
from rewriter.offline_systems import VerifiedStateConstants
from rewriter.multibank import MultiStateBankReader, MultiTeacherCacheReader
from scripts.qualify_coordinate_chisel import FixedContexts
from scripts.continue_coordinate_chisel import run
import scripts.continue_coordinate_chisel as continuation


def differences(a, b, path=''):
    if isinstance(a, torch.Tensor):
        return [] if isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b) else [path]
    if isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys(): return [path + ':keys']
        return [v for key in a for v in differences(a[key], b[key], path + '/' + str(key))]
    if isinstance(a, (tuple, list)):
        if type(a) is not type(b) or len(a) != len(b): return [path + ':sequence']
        return [v for i, (x, y) in enumerate(zip(a, b)) for v in differences(x, y, path + '/' + str(i))]
    return [] if a == b else [path]


@contextmanager
def transport(mode, timings, gpt_ids=()):
    if mode not in ('reference', 'coalesced', 'banked', 'banked_prefetch'): raise ValueError('unknown transport')
    if mode == 'banked_prefetch' and not gpt_ids: raise ValueError('explicit GPT sampler order required')
    originals = []
    adapters = []
    gradient_hashes = timings.setdefault('first_update_raw_gradient_sha256', [])
    def patch(owner, name, function):
        originals.append((owner, name, getattr(owner, name)))
        setattr(owner, name, function)
    def timed(name, function):
        def call(*args, **kwargs):
            start = time.perf_counter()
            try: return function(*args, **kwargs)
            finally: timings.setdefault(name, []).append(1000 * (time.perf_counter() - start))
        return call
    def hash_gradients(parameters):
        digest = hashlib.sha256()
        for parameter in parameters:
            value = parameter.grad
            digest.update(str((tuple(parameter.shape), value is None)).encode())
            if value is not None:
                value = value.detach().cpu().contiguous()
                digest.update(str(value.dtype).encode())
                digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()
    original_private_clip = continuation.clip_private_gradients
    original_shared_clip = continuation.clip_shared_continuation
    def private_clip(parameters, *args, **kwargs):
        parameters = list(parameters)
        if len(gradient_hashes) < 5:
            gradient_hashes.append(('private_unweighted', hash_gradients(parameters)))
        return original_private_clip(parameters, *args, **kwargs)
    def shared_clip(optimizer, *args, **kwargs):
        if len(gradient_hashes) < 5:
            gradient_hashes.append(('shared_weighted_sum', hash_gradients(p for group in optimizer.param_groups for p in group['params'])))
        return original_shared_clip(optimizer, *args, **kwargs)
    patch(continuation, 'clip_private_gradients', private_clip)
    patch(continuation, 'clip_shared_continuation', shared_clip)
    original_init = FixedContexts.__init__
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if mode == 'coalesced':
            for seed, old in list(self.reader.readers.items()):
                replacement = CoalescedStateBankReader(old.directory, verify_once=old.verify_once, mmap_model=old.mmap_model)
                # Initialization has not consumed any records. Do not invent a
                # verified stamp or bypass the ordinary first-visit hash.
                self.reader.readers[seed] = replacement
        elif mode in ('banked', 'banked_prefetch'):
            # Keep the original MultiStateBankReader for cache namespace
            # binding. Only the catalog's read entry point gains static views.
            adapter = BankedObservationReader(self.reader)
            self.catalog.readers['gpt'] = adapter
            adapters.append(adapter)
    patch(FixedContexts, '__init__', initialize)
    if mode == 'banked_prefetch':
        original_measure = FixedContexts.measure
        def measure(self, context, *args, **kwargs):
            if context['domain'] == 'gpt' and kwargs.get('learning'):
                position = gpt_ids.index(context['local_id'])
                self.catalog.readers['gpt'].prefetch(gpt_ids[(position + 1) % len(gpt_ids)])
            return original_measure(self, context, *args, **kwargs)
        patch(FixedContexts, 'measure', measure)
    patch(MultiStateBankReader, 'read', timed('bank_and_observation', MultiStateBankReader.read))
    patch(MultiTeacherCacheReader, 'read', timed('teacher_panel_read', MultiTeacherCacheReader.read))
    patch(VerifiedStateConstants, 'load', timed('engine_load', VerifiedStateConstants.load))
    patch(VerifiedStateConstants, 'grouped', timed('grouped_constants', VerifiedStateConstants.grouped))
    patch(VerifiedStateConstants, 'radius_squared', timed('radius_constants', VerifiedStateConstants.radius_squared))
    try: yield
    finally:
        for adapter in adapters: adapter.close()
        for owner, name, old in reversed(originals): setattr(owner, name, old)


def qualify(recipe, output, order):
    output.mkdir(parents=True, exist_ok=False)
    recipe = json.loads(Path(recipe).read_text())
    end = recipe['end_update']
    checkpoints = {}
    result = dict(scope='Disposable complete four-state coordinate updates; same frozen checkpoint, shared/private Adam and RNG per arm. Cached teacher body never replayed. Coalescing changes only observation transport. Initial verification/capture and final checkpoint reported separately.',
                  recipe=recipe, order=order, arms=[], comparisons=[])
    fields = ('actor', 'shared_optimizer', 'private', 'duals', 'rng', 'cuda_rng', 'update', 'compute')
    for index, mode in enumerate(order):
        arm = output / f'{index}_{mode}'
        current = dict(recipe, name=f'coordinate-staging-{index}-{mode}')
        path = output / f'{index}_{mode}.json'; path.write_text(json.dumps(current, indent=2))
        start = time.perf_counter()
        with (output / f'{index}_{mode}.log').open('w') as log:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--recipe', str(path),
                            '--output', str(arm), '--arm', mode], stdout=log, stderr=subprocess.STDOUT, check=True)
        elapsed = time.perf_counter() - start
        timings = json.loads((arm / 'staging_timings.json').read_text())
        checkpoint = arm / f'checkpoint_{end:06d}.pt'
        checkpoints[index] = checkpoint
        metrics = [json.loads(line) for line in (arm / 'metrics.jsonl').read_text().splitlines()]
        updates = [row['wall_seconds'] for row in metrics if row.get('event') == 'joint_update']
        result['arms'].append(dict(index=index, mode=mode, elapsed_seconds=elapsed,
            update_seconds=updates, warm_middle_median_seconds=statistics.median(updates[1:-1]) if len(updates)>2 else None,
            load_cpu_wall_ms=timings, checkpoint=str(checkpoint)))
        if index:
            reference = torch.load(checkpoints[0], map_location='cpu', weights_only=False)
            candidate = torch.load(checkpoint, map_location='cpu', weights_only=False)
            diff = [d for field in fields for d in differences(reference[field], candidate[field], field)]
            gradients_equal = timings['first_update_raw_gradient_sha256'] == result['arms'][0]['load_cpu_wall_ms']['first_update_raw_gradient_sha256']
            result['comparisons'].append(dict(reference_index=0, candidate_index=index,
                all_numeric_state_bitexact=not diff, first_update_raw_gradients_bitexact=gradients_equal,
                fields=fields, differences=diff))
            del reference, candidate
        result['all_gates'] = all(row['all_numeric_state_bitexact'] and row['first_update_raw_gradients_bitexact'] for row in result['comparisons'])
        (output / 'summary.json').write_text(json.dumps(result, indent=2))
        if not result['all_gates']: raise RuntimeError('staging changed complete numeric training state')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=['reference', 'coalesced', 'banked', 'banked_prefetch'])
    parser.add_argument('--order', nargs='+', choices=['reference', 'coalesced', 'banked', 'banked_prefetch'], default=['reference', 'coalesced', 'coalesced', 'reference'])
    args = parser.parse_args()
    if args.arm:
        recipe = json.loads(args.recipe.read_text())
        saved = torch.load(recipe['checkpoint'], map_location='cpu', weights_only=False)
        ids = tuple(uid for domain, uid in saved['private'] if domain == 'gpt')
        del saved
        timings = {}
        with transport(args.arm, timings, ids): run(args.recipe, args.output)
        (args.output / 'staging_timings.json').write_text(json.dumps(timings, indent=2))
    else:
        qualify(args.recipe, args.output, args.order)
