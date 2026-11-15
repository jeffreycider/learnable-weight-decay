"""Optimize one checkpoint's anchored factors as a diagnostic, not Chisel.

This bounded control uses the same finite-KL engine, group normalization, radius,
and primal-dual loss as production fitting. It never trains on withheld states,
never changes Chisel, and is not a generalizing or deployable policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline.execution import enable_deterministic_execution
from baseline.model import GPT
from baseline.train import source_record
from rewriter.diagnostic_action import DiagnosticAnchoredAction
from rewriter.edit_scope import editable_names, editable_weight_norm_sq, scoped_snapshot
from rewriter.function_preservation import FunctionalPreservation
from rewriter.gpt_precision import precision_metadata
from rewriter.primal_dual import PrimalDual, feasibility_score
from rewriter.state_bank import StateBankReader
from rewriter.teacher_cache import TeacherCacheReader
from scripts.train_multistate_chisel import build_proposal, evaluate_sequences, mean_kl_and_backward


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-bank', type=Path, required=True)
    parser.add_argument('--teacher-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--state-id', type=int, default=999)
    parser.add_argument('--updates', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--dual-lr', type=float, default=.1)
    parser.add_argument('--rho', type=float, default=.01)
    parser.add_argument('--kl-tolerance', type=float, default=1e-6)
    parser.add_argument('--radius-fraction', type=float, default=.01)
    parser.add_argument('--audit-every', type=int, default=25)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args(argv)
    if args.updates < 1 or args.updates > 1000 or args.audit_every < 1:
        parser.error('bounded diagnostic requires 1..1000 updates and positive audit frequency')
    if any(not math.isfinite(v) or v <= 0 for v in
           (args.lr, args.dual_lr, args.rho, args.kl_tolerance, args.radius_fraction)):
        parser.error('positive finite optimizer and constraint scales required')
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == 'cuda':
        enable_deterministic_execution()
    reader = StateBankReader(args.state_bank, verify_once=True, mmap_model=True)
    if args.state_id not in reader.manifest['split']['train']:
        parser.error('diagnostic optimization may use fitting checkpoints only')
    cache = TeacherCacheReader(args.teacher_cache, verify_once=True, mmap=True)
    record = reader.read(args.state_id, model_device='cpu', device=device)
    model = GPT(**reader.manifest['provenance']['model_config']).to(device).requires_grad_(False)
    engine = FunctionalPreservation(model, microbatch_sequences=16, teacher_cache_device=device,
                                   kl_dtype=torch.float64, gpt_precision='fp32')
    cache.load_engine(engine, record)
    if device.type == 'cuda':
        from rewriter.accelerated_grouped_norm import AcceleratedGroupedNormObjective as NormObjective
    else:
        from rewriter.grouped_norm import GroupedNormObjective as NormObjective
    grouped = NormObjective(engine.parameters, zero_reference_policy='freeze')
    editable = editable_names(engine.parameters, 'grouped') - grouped.frozen_parameter_names
    snapshot = scoped_snapshot(record.snapshot, editable_weight_norm_sq(engine.parameters, editable),
                               args.radius_fraction, scope='grouped')
    action = DiagnosticAnchoredAction(snapshot)
    optimizer = torch.optim.Adam(action.parameters(), lr=args.lr, betas=(.9, .99), eps=1e-8)
    dual = PrimalDual([args.state_id], tolerance=args.kl_tolerance, rho=args.rho, dual_lr=args.dual_lr)
    train = cache.read(args.state_id, 'train')
    audit = cache.read(args.state_id, 'audit')
    if len(train.inputs) != 128 or len(audit.inputs) != 64:
        raise ValueError('this control requires the full 128 fitting / 64 withheld panels')
    if set(train.sequence_ids) & set(audit.sequence_ids):
        raise ValueError('fitting and withheld tokens overlap')
    metadata = dict(options={k: str(v) if isinstance(v, Path) else v
                             for k, v in vars(args).items() if k != 'resume'},
                    architecture=action.architecture, diagnostic_only=True, deployable=False,
                    state_sha256=record.assert_verified(),
                    state_bank_manifest_sha256=hashlib.sha256((args.state_bank/'manifest.json').read_bytes()).hexdigest(),
                    teacher_manifest_sha256=hashlib.sha256((args.teacher_cache/'manifest.json').read_bytes()).hexdigest(),
                    gpt_precision=precision_metadata('fp32'), train_sequence_ids=train.sequence_ids,
                    audit_sequence_ids=audit.sequence_ids,
                    allocated_coordinates=sum(p.numel() for p in action.parameters()),
                    editable_coordinates=sum(p.numel()+q.numel() for n, p, q in
                        zip(action.matrix_names, action.rows, action.columns) if n in editable)
                        + sum(v.numel() for n, v in zip(action.vector_names, action.vectors) if n in editable),
                    selection='Best feasible shrinkage on the fixed fitting panel; withheld panel read only after fitting',
                    gradient_clip=1., adam_betas=[.9, .99], adam_epsilon=1e-8, adam_weight_decay=0.,
                    initialization='zero canonical factors', schedule='fixed learning rates, bounded update count')
    root = Path(__file__).resolve().parents[1]
    metadata['implementation_sha256'] = {
        name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
        ('scripts/diagnose_fixed_action_fit.py', 'scripts/train_multistate_chisel.py',
         'rewriter/diagnostic_action.py', 'rewriter/function_preservation.py',
         'rewriter/gpt_precision.py', 'rewriter/primal_dual.py')}
    previous = None
    if args.resume:
        previous = torch.load(args.resume, map_location=device, weights_only=False)
        if previous['metadata'] != metadata:
            raise ValueError('diagnostic resume configuration, source or data changed')
        if not (args.output/'action_best.pt').exists():
            raise ValueError('diagnostic resume requires the preserved best action')
        action.load_state_dict(previous['action'])
        optimizer.load_state_dict(previous['optimizer'])
        dual.load_state_dict(previous['primal_dual'])
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        source_record(metadata, args.output)
        (args.output/'config.json').write_text(json.dumps(metadata, indent=2)+'\n')
    wb = None
    if args.wandb:
        import wandb
        wb = wandb.init(project='chisel', mode='offline', dir=str(args.output), config=metadata)
    started = time.perf_counter() - (previous['elapsed_seconds'] if previous else 0.)
    log_file = (args.output/'metrics.jsonl').open('a' if previous else 'w')

    def compute_counts():
        counts = engine.compute.as_dict()
        counts['cached_teacher_head_forwards'] = getattr(engine, 'cached_teacher_head_forwards', 0)
        counts['cached_teacher_head_tokens'] = getattr(engine, 'cached_teacher_head_tokens', 0)
        if previous:
            counts = {k: v+previous['compute'][k] for k, v in counts.items()}
        return counts

    def log(event, **fields):
        row = dict(event=event, elapsed_seconds=time.perf_counter()-started, **fields)
        log_file.write(json.dumps(row)+'\n'); log_file.flush()
        if wb:
            wb.log({k: v for k, v in row.items() if isinstance(v, (int, float))})
        print(json.dumps(row), flush=True)

    def save(name, update):
        temp = args.output/(name+'.tmp')
        torch.save(dict(metadata=metadata, update=update, action=action.state_dict(),
                        optimizer=optimizer.state_dict(), primal_dual=dual.state_dict(),
                        best_score=best_score, best_update=best_update,
                        elapsed_seconds=time.perf_counter()-started, compute=compute_counts()), temp)
        temp.replace(args.output/name)

    def evaluate(panel, update, role):
        with torch.enable_grad():
            parameters, regularizer, stats, _, credit = build_proposal(
                action, record.snapshot, engine, grouped, radius_fraction=args.radius_fraction)
        kl, ce = evaluate_sequences(engine, parameters, panel.inputs,
            teacher_provider=lambda start, stop: panel.teacher_batch(engine, start, stop), return_ce=True)
        row = dict(update=update, role=role, mean_kl=kl, **stats, **ce)
        row['feasible'] = kl <= args.kl_tolerance
        log('audit', **row)
        return row

    if previous:
        best_score, best_update = previous['best_score'], previous['best_update']
        first_update = previous['update']+1
        log('resume', checkpoint=str(args.resume), next_update=first_update,
            compute_scope='Restored trajectory plus new work; abandoned attempt rows remain in raw log')
    else:
        initial = evaluate(train, 0, 'fitting')
        if initial['mean_kl'] != 0 or initial['realized_edit_norm'] != 0:
            raise AssertionError('zero factors must recover the exact cached FP32 teacher')
        best_score = feasibility_score([initial], args.kl_tolerance)
        best_update = 0
        first_update = 1
        save('action_best.pt', 0)
    for update in range(first_update, args.updates+1):
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        parameters, regularizer, stats, _, credit = build_proposal(
            action, record.snapshot, engine, grouped, radius_fraction=args.radius_fraction)
        measured = mean_kl_and_backward(engine, parameters, regularizer, train.inputs,
            kl_budget=args.kl_tolerance, primal_dual=dual, state_id=args.state_id, actor_credit=credit,
            teacher_provider=lambda start, stop: train.teacher_batch(engine, start, stop))
        gradient = torch.nn.utils.clip_grad_norm_(action.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        dual.step([(args.state_id, measured['mean_kl'])], expected_ids=[args.state_id])
        log('update', update=update, measurement_at='before_optimizer_update', **stats, **measured,
            gradient_norm=float(gradient), wall_seconds=time.perf_counter()-step_started,
            compute=compute_counts())
        del parameters, regularizer, credit
        if update % args.audit_every == 0 or update == args.updates:
            row = evaluate(train, update, 'fitting')
            score = feasibility_score([row], args.kl_tolerance)
            if score < best_score:
                best_score, best_update = score, update
                save('action_best.pt', update)
            save('action_latest.pt', update)
    saved = torch.load(args.output/'action_best.pt', map_location=device, weights_only=False)
    action.load_state_dict(saved['action'])
    selected_fit = evaluate(train, best_update, 'selected_fitting')
    selected_audit = evaluate(audit, best_update, 'selected_withheld_sequences')
    log('finished', updates=args.updates, best_update=best_update, compute=compute_counts(),
        selected_fitting=selected_fit, selected_withheld=selected_audit,
        stopping_reason='predeclared_bounded_control', convergence_claim=False, deployable=False)
    log_file.close()
    if wb:
        wb.finish()


if __name__ == '__main__':
    main()
