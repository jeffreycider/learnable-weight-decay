"""One norm-only anchored diagnostic, evaluated on a completed frozen run's audit.

This analytic control does not replace or initialize the learned policy. It
ignores preservation when choosing its action, then measures finite teacher KL.
Extra GPT forwards are explicit; there are no backwards or optimizer updates.
The continuous optimum assumes exact orthonormal anchors/unrounded weights;
its BF16/FP32 stored-parameter realization is not a rigorous norm lower bound.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline.execution import enable_deterministic_execution, execution_environment
from baseline.model import GPT
from baseline.train import source_record
from rewriter.algebra import Anchors, Factors, RawAction
from rewriter.function_preservation import FunctionalPreservation
from rewriter.replay_buffer import sample_past_sequences, synthetic_sequences
from rewriter.edit_scope import (editable_names,editable_weight_norm_sq,scoped_action,
    keep_readonly_parameters,readonly_hashes,verify_readonly)
from rewriter.grouped_norm import GroupedNormObjective
from rewriter.grouped_control import grouped_norm_only_action


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def norm_only_action(snapshot, editable=None):
    """Minimize ||W+a||² in the anchored subspace with one global radius.

    All vector coordinates are included. Coefficients use the saved weight
    probes; FP64 thin arithmetic limits additional cancellation/rounding, but
    cannot recover precision already lost when these probes were collected.
    """
    radius = snapshot.radius.double()
    if radius.ndim or not torch.isfinite(radius) or radius < 0:
        raise ValueError('finite nonnegative scalar radius required')
    matrices, vectors, orthogonality_error = {}, {}, 0.
    for name, obs in snapshot.matrices.items():
        anchors = Anchors(obs.anchors.A.double(), obs.anchors.B.double())
        for basis in (anchors.A, anchors.B):
            error = (basis.T @ basis - torch.eye(basis.shape[1], device=basis.device, dtype=basis.dtype)).abs().max()
            orthogonality_error = max(orthogonality_error, float(error))
        if orthogonality_error > 1e-5:
            raise ValueError('saved anchors are not orthonormal within the FP32 tolerance')
        wb, wta = (x.double() for x in obs.weight_probes)
        matrices[name] = Factors.canonical(wb, wta, anchors)
    for name, obs in snapshot.vectors.items():
        vectors[name] = obs.weight.double().clone()
    projected = RawAction(matrices, vectors)
    if editable is not None:
        projected=scoped_action(projected,editable)
        matrices,vectors=projected.matrices,projected.vectors
    projected_sq = projected.norm_sq()
    if not torch.isfinite(projected_sq):
        raise ValueError('nonfinite weight projection')
    projected_norm = projected_sq.sqrt()
    scale = torch.where(projected_norm > 0, torch.minimum(torch.ones_like(radius),
                        radius / projected_norm.clamp_min(torch.finfo(torch.float64).tiny)), radius.new_zeros(()))
    action = RawAction({n: Factors(-scale*f.P, -scale*f.Q, f.anchors) for n, f in matrices.items()},
                       {n: -scale*v for n, v in vectors.items()})
    return action, dict(radius=float(radius), projected_weight_norm=float(projected_norm),
        global_coefficient=float(scale), raw_edit_norm=float(action.norm_sq().sqrt()),
        ideal_continuous_squared_norm_change=float((scale.square()-2*scale)*projected_sq),
        cached_contraction_squared_norm_change=float(action.norm_change(snapshot.weights,snapshot.vector_weights)),
        anchor_orthogonality_max_error=orthogonality_error,
        interpretation='Norm-only continuous optimum for this fixed anchored subspace/radius; '
                       'rounded stored action and cached FP32 probes do not constitute a strict bound.')


def _normalized_provenance(rows):
    return [dict(row, shard=Path(row['shard']).name) for row in rows]


def matched_replay(source, run, training_files):
    """Reconstruct exactly the original sample IDs/order, allowing local staging paths."""
    config = source['base_config']
    replay = sample_past_sequences(training_files, source['checkpoint_step'],
        batch_tokens=config['batch_tokens'], seq_len=config['seq_len'], count=source['buffer_sequences'])
    original = json.loads((Path(run)/'replay_provenance.json').read_text())
    if original['metadata'] != replay.metadata:
        raise ValueError('replay metadata differs from source run')
    for split in ('train', 'audit'):
        if _normalized_provenance(original[split]) != _normalized_provenance(getattr(replay,split).provenance):
            raise ValueError(f'{split} replay IDs/order/shards differ from source run')
    return replay


def matching_norm_control(snapshot,parameters,source):
    """Choose the control for the recorded objective, never relabel another optimum."""
    scope=source.get('editable_scope','all')
    objective=source.get('norm_objective','global_squared')
    if objective not in ('global_squared','grouped_l2') or ((scope=='grouped')!=(objective=='grouped_l2')):
        raise ValueError('recorded edit scope and norm objective do not match')
    if objective=='grouped_l2':
        groups=GroupedNormObjective(parameters)
        action,algebra=grouped_norm_only_action(snapshot,parameters,groups.groups)
        return action,algebra,groups
    action,algebra=norm_only_action(snapshot,editable_names(parameters,scope))
    return action,algebra,None


@torch.no_grad()
def evaluate(engine, parameters, replay, source):
    config = source['base_config']
    groups = {'historical_audit': (replay.audit, 'past_training')}
    for i, kind in enumerate(('uniform', 'repeated', 'alternating')):
        groups[kind] = (synthetic_sequences(kind, source['synthetic_audit_sequences'], config['seq_len'],
            config['model']['vocab_size'], seed=600001+i), 'synthetic')
    distributions = {}
    for name, (batch, role) in groups.items():
        # Match the source runner's full=True convention, including its cap.
        count = min(len(batch.inputs), source['final_audit_sequences'])
        values = []
        for index in range(count):
            teacher = engine.cache_teacher(batch.inputs[index:index+1], data_role=role)[0]
            values.append(float(engine.student_parameter_loss(parameters, teacher)))
        tensor = torch.tensor(values, dtype=torch.float64)
        if not torch.isfinite(tensor).all():
            raise ValueError(f'nonfinite measured KL in {name}')
        digest = hashlib.sha256(batch.inputs[:count].contiguous().numpy().tobytes()).hexdigest()
        distributions[name] = dict(sequences=count, mean_kl=float(tensor.mean()),
            p95_sequence_mean_kl=float(tensor.quantile(.95)), max_sequence_mean_kl=float(tensor.max()),
            sequence_mean_kl=values, input_int32_sha256=digest)
    return distributions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--source-run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--train-data', default='/tmp/fineweb10B/fineweb_train_*.bin')
    parser.add_argument('--editable-scope',choices=('grouped','hidden','all'),
                        help='must match source run; absent historical setting means all')
    args = parser.parse_args(argv)
    source = json.loads((args.source_run/'config.json').read_text())
    scope=source.get('editable_scope','all')
    objective=source.get('norm_objective','global_squared')
    if objective not in ('global_squared','grouped_l2') or ((scope=='grouped')!=(objective=='grouped_l2')):
        raise ValueError('recorded edit scope and norm objective do not match')
    if args.editable_scope is not None and args.editable_scope!=scope:
        raise ValueError('norm control scope must match the source run')
    events = [json.loads(line) for line in (args.source_run/'metrics.jsonl').read_text().splitlines() if line]
    if not any(row.get('event') == 'finished' for row in events):
        raise ValueError('source frozen experiment must have completed')
    checkpoint_hash = sha256(args.checkpoint)
    if checkpoint_hash != source['checkpoint_sha256']:
        raise ValueError('checkpoint differs from the source frozen run')
    args.output.mkdir(parents=True, exist_ok=False)
    enable_deterministic_execution()
    device = torch.device('cuda:0'); torch.cuda.set_device(device)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    if state['controller'] is not None or state['config']['mode'] != 'baseline':
        raise ValueError('requires baseline checkpoint without transferred controller state')
    if state['config'] != source['base_config'] or state['step'] != source['checkpoint_step']:
        raise ValueError('source checkpoint configuration/clock mismatch')
    observation_path = args.source_run/'observation.pt'
    observation = torch.load(observation_path, map_location=device, weights_only=False)
    snapshot = observation['snapshot']
    model = GPT(**state['config']['model']).to(device)
    model.load_state_dict(state['model']); model.requires_grad_(False)
    del state
    replay = matched_replay(source, args.source_run, args.train_data)
    metadata = dict(experiment='standalone norm-only anchored diagnostic control', benchmark_compliant=False,
        editable_scope=scope,norm_objective=objective,
        norm_relative_change_definition=('mean_functional_group_relative_L2_change' if objective=='grouped_l2'
                                         else 'editable_squared_norm_change_over_reference'),
        controller_replacement=False, controller_state_loaded=False, extra_gpt_backwards=0,
        source_run=str(args.source_run.resolve()), source_config=source,
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_hash,
        observation_sha256=sha256(observation_path), source_run_config_sha256=sha256(args.source_run/'config.json'),
        script_sha256=sha256(__file__), kl_arithmetic='float64 from float32 GPT logits',
        replay_matching='same sampled IDs/order and shard basenames; original sampler did not hash shard contents',
        execution=execution_environment())
    (args.output/'config.json').write_text(json.dumps(metadata, indent=2)+'\n')
    source_record(metadata, args.output)
    started = time.perf_counter()
    engine = FunctionalPreservation(model, microbatch_sequences=1, teacher_cache_device='cuda', kl_dtype=torch.float64)
    editable=editable_names(engine.parameters,scope)
    editable_norm_sq=(engine.base_weight_norm_sq if scope=='all' else editable_weight_norm_sq(engine.parameters,editable))
    readonly=readonly_hashes(engine.parameters,editable)
    with torch.no_grad():
        action, algebra,group_objective = matching_norm_control(snapshot,engine.parameters,source)
        parameters = keep_readonly_parameters(engine.edited_parameters(action),engine.parameters,editable)
        grouped=group_objective.evaluate(parameters) if group_objective is not None else None
        if grouped is None:
            change, squared_edit = engine.realized_parameter_norm_change(parameters)
            relative_change=change/editable_norm_sq
        else:
            change,squared_edit=grouped.squared_norm_change,grouped.edit_norm_sq
            relative_change=grouped.mean_relative_change
        distributions = evaluate(engine, parameters, replay, source)
    compute = engine.compute.as_dict()
    assert compute['student_backwards'] == 0
    result = dict(event='finished', label=('norm_only_grouped_l2_anchored' if grouped is not None else 'norm_only_anchored'),
        norm_objective=objective,benchmark_compliant=False,
        elapsed_seconds=time.perf_counter()-started, algebra=algebra,
        actual_squared_norm_change=float(change), norm_relative_change=float(relative_change),
        editable_squared_norm_relative_change=float(change/editable_norm_sq),
        **(grouped.metrics() if grouped is not None else {}),
        total_norm_relative_change=float(change/engine.base_weight_norm_sq),editable_scope=scope,
        editable_weight_norm=float(editable_norm_sq.sqrt()),read_only_integrity=verify_readonly(parameters,editable,readonly),
        realized_edit_norm=float(squared_edit.sqrt()), base_weight_norm=float(engine.base_weight_norm_sq.sqrt()),
        distributions=distributions, compute=compute,
        calibration_passes_this_control=0, reused_source_calibration=observation['calibration'])
    (args.output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
