"""Read-only gradient decomposition on a completed CPU fit's exact training panel.

One ordinary offline student forward/backward per stored training sequence, plus
norm-only matrix/controller VJPs. Cached CPU teachers are required. No optimizer
or dual step, unseen examples, validation data, or online GPT run is involved.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from baseline.model import GPT
from rewriter.edit_scope import editable_names, editable_weight_norm_sq
from rewriter.experimental_factor_panel import FactorPanelCapture  # Pin optional sources before execution.
from rewriter.experimental_factor_norm import prepare_factor_panel
from rewriter.function_preservation import FunctionalPreservation
from rewriter.grouped_coordinate_chisel import GroupedCoordinateChisel
from rewriter.grouped_norm import GroupedNormObjective
from rewriter.prefix_cache import unpack_record
from rewriter.ranking_dual import RankingDual
from rewriter.tokenwise_reference import TokenwiseConfig
from scripts.diagnose_cpu_norm_preservation import (
    LocalCPUContexts, load_bound_state, pin_sources, teacher_cache_class, tensor_sha,
)
from scripts.evaluate_cpu_joint_unseen_tokens import restore_evaluation_dual, validate_seed_identity
from scripts.fit_cpu_edit_teacher import exact_state_equal, sha
from scripts.prepare_cpu_teacher_state import validate_train_panel


def fitted_inputs(panel, case):
    ids = panel['metadata']['sequence_ids']
    wanted = case['sequence_ids']
    if len(ids) != len(set(ids)) or len(wanted) != len(set(wanted)):
        raise ValueError('Ambiguous training sequence identities')
    positions = {uid: i for i, uid in enumerate(ids)}
    if not 1 <= len(wanted) <= 16 or any(uid not in positions for uid in wanted):
        raise ValueError('Missing or oversized fitting panel')
    inputs = panel['inputs'][[positions[uid] for uid in wanted]].clone()
    if tensor_sha(inputs) != case['input_sha256']:
        raise ValueError('Fitting input hash changed')
    return inputs


def validate_profile_endpoint(saved, training):
    step = saved['completed_updates']
    final = training['final']
    if type(step) is not int or step < 1 or training['completed_updates'] != step or final['completed_updates_before_measure'] != step:
        raise ValueError('Completed endpoint does not match checkpoint')
    if final['updates_after'] != step or final['learning'] or final['gradient_profile']:
        raise ValueError('Expected completed forward-only endpoint')
    meta = saved['metadata']
    if not meta.get('fixed_train_panels') or len(meta['cases']) != 2:
        raise ValueError('Diagnostic currently requires two fixed fitting panels')
    rows = {row['local_id']: row for row in final['rows']}
    if len(rows) != 2 or len(final['rows']) != 2 or set(rows) != {c['state_id'] for c in meta['cases']} or saved['dual']['updates'] != step:
        raise ValueError('Endpoint identities or dual clock differ')
    if meta['state_weights'] != [.5, .5]:
        raise ValueError('Expected equal checkpoint weights')
    for case in meta['cases']:
        if case['state_id'] not in rows or rows[case['state_id']]['context_ids'] != case['sequence_ids']:
            raise ValueError('Endpoint fitting sequence identities differ')


def validate_reproduced_measurement(row, endpoint):
    # A source/runtime difference must not silently become a gradient diagnosis
    # of another edited network. These are deterministic CPU measurements.
    keys = ('context_ids', 'norm_objective', 'mean_kl', 'token_kl_quantiles',
            'strict_topk_changes', 'ranking_penalty')
    if any(row[key] != endpoint[key] for key in keys):
        raise RuntimeError('Profile did not reproduce the saved endpoint measurement')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    started = time.perf_counter()
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    result_path = args.checkpoint.parent / 'complete.json'
    training = json.loads(result_path.read_text())
    validate_profile_endpoint(saved, training)
    if args.checkpoint.name != f"checkpoint_{saved['completed_updates']:04d}.pt":
        raise ValueError('Expected final checkpoint filename')
    meta = saved['metadata']
    if json.loads(json.dumps(meta)) != json.loads((args.checkpoint.parent / 'metadata.json').read_text()):
        raise ValueError('Completed run metadata differs')
    args.output.mkdir(parents=True, exist_ok=False)
    sources = pin_sources(args.output)
    checkpoint_sha = sha(args.checkpoint)
    config = TokenwiseConfig(**meta['objective'])
    ids = [case['state_id'] for case in meta['cases']]
    dual = restore_evaluation_dual(ids, config, meta, saved['dual'])
    dual_before = copy.deepcopy(dual.state_dict())
    actor = GroupedCoordinateChisel(rank=8, trajectory_dim=400, model_dim=256,
        num_layers=2, latent_tokens=8, fresh_weight_core=True,
        factor_precision='fp64', residual_precision='fp32')
    actor.load_state_dict(saved['actor'], strict=True)
    actor.eval()
    for name, p in actor.named_parameters():
        p.requires_grad_(meta['args']['policy'] == 'full' or name.startswith(('matrix_output.', 'vector_output.')))
    contexts = LocalCPUContexts('direct_streamed')
    cache_records = {c['state_id']: c for c in meta['teacher_cache_provenance']}
    endpoint_rows = {row['local_id']: row for row in training['final']['rows']}
    engine = None
    rows = []
    expected_sequences = 0
    for case in meta['cases']:
        paths = case['paths']
        for name, path in paths.items():
            if sha(path) != case['sha256'][name]:
                raise ValueError('Original case source changed: ' + name)
        state = torch.load(paths['state'], map_location='cpu', weights_only=False, mmap=True)
        if state['completed_steps'] != case['state_id']:
            raise ValueError('GPT checkpoint identity changed')
        panel = torch.load(paths['panel'], map_location='cpu', weights_only=False, mmap=True)
        panel_meta = json.loads(Path(paths['panel_metadata']).read_text())
        validate_train_panel(panel, panel_meta, case['sha256']['state'], case['sha256']['panel'])
        inputs = fitted_inputs(panel, case)
        del panel
        seed = torch.load(paths['checkpoint'], map_location='cpu', weights_only=False, mmap=True)['metadata']
        snapshot = unpack_record(state['observation'], device='cpu').snapshot
        private = actor.make_private(snapshot, state_key=('cpu-teaching', 0, case['state_id'], case['sha256']['state']),
                                     seed=seed['args']['private_seed'])
        private.load_state_dict(saved['private'][case['state_id']], strict=True)
        validate_seed_identity(case, seed, actor, private)
        if engine is None:
            base = state['model']
            depth = len({n.split('.')[1] for n in base if n.startswith('blocks.')})
            model = GPT(base['embed.weight'].shape[0], depth, base['embed.weight'].shape[1]).float().eval().requires_grad_(False)
            engine = FunctionalPreservation(model, microbatch_sequences=1, chunk_tokens=128,
                teacher_cache_device='cpu', kl_dtype=torch.float64, gpt_precision='fp32')
        load_bound_state(engine, state['model'], case['sha256']['state'])
        cache = cache_records[case['state_id']]
        if sha(Path(cache['path']) / 'manifest.json') != cache['manifest_sha256']:
            raise ValueError('Teacher cache manifest changed')
        provider = teacher_cache_class(cache.get('format', 'fp64_log_probs'))(
            cache['path'], engine, inputs, case['sequence_ids'], case['sha256']['state'])
        grouped = GroupedNormObjective(engine.parameters, zero_reference_policy='freeze')
        editable = editable_names(engine.parameters, 'grouped') - grouped.frozen_parameter_names
        context = dict(engine=engine, domain='gpt', local_id=case['state_id'], snapshot=snapshot,
            grouped=grouped, inputs=inputs, ids=case['sequence_ids'], teacher=provider,
            norm_squared=editable_weight_norm_sq(engine.parameters, editable))
        case_config = dual.config_for_state(case['state_id']) if isinstance(dual, RankingDual) else config
        row = contexts.measure(context, actor, private, case_config, dual,
                               learning=True, weight=.5, gradient_components=True)
        validate_reproduced_measurement(row, endpoint_rows[case['state_id']])
        rows.append(row)
        expected_sequences += len(inputs)
        if not exact_state_equal(private.state_dict(), saved['private'][case['state_id']]):
            raise RuntimeError('Private weights changed')
        print(json.dumps(row, allow_nan=False), flush=True)
        del context, provider, grouped, private, state, snapshot
    if not exact_state_equal(actor.state_dict(), saved['actor']) or not exact_state_equal(dual.state_dict(), dual_before):
        raise RuntimeError('Actor or dual changed')
    counts = contexts.compute
    if counts['student_forwards'] != expected_sequences or counts['student_backwards'] != expected_sequences or counts['teacher_forwards']:
        raise RuntimeError('Unexpected GPT evaluation count')
    if sha(args.checkpoint) != checkpoint_sha or any(sha(name) != digest for name, digest in sources.items()):
        raise RuntimeError('Input checkpoint or diagnostic source changed')
    result = dict(scope=__doc__, checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_sha,
        completed_training_updates=saved['completed_updates'], training_result_sha256=sha(result_path),
        source_sha256=sources, rows=rows, compute=counts, extra_norm_vjps=len(rows),
        optimizer_updates=0, dual_updates=0, actor_private_dual_unchanged=True,
        saved_endpoint_measurements_identical=True,
        elapsed_seconds=time.perf_counter() - started)
    (args.output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
