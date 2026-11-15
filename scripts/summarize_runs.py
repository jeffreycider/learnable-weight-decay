#!/usr/bin/env python3
"""Summarize recorded runs without promoting incomplete or unmatched results.

Examples:
  python scripts/summarize_runs.py runs/base0 runs/learned0 --output reports/pilot
  python scripts/summarize_runs.py --run base0 /a/metrics.jsonl /a/config.json /a/environment.json \
      --run learned0 /b/metrics.jsonl /b/config.json /b/environment.json --output reports/pilot

Only an explicitly supplied checkpoint can replace the configured endpoint.
Optional data_manifest.json is discovered beside config.json. Nothing trains or
selects an endpoint based on validation loss. matplotlib is optional (--plot).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
from datetime import datetime, timezone

VERSION = 2
OBSERVABLES = (
    'train_loss', 'kappa', 'action_norm', 'committed_norm', 'realized_edit_norm',
    'casting_error_norm', 'base_update_norm', 'weight_norm', 'norm_change',
    'train_proxy', 'audit_proxy', 'allones_relative_error', 'radius',
    'task_gradient_norm', 'auxiliary_gradient_norm', 'meta_updates',
    'auxiliary_evaluations', 'controller_parameters', 'actor_snapshot_elements',
    'matrix_count', 'vector_count', 'previous_edit_reversal_fraction',
    'previous_realized_edit_norm', 'base_update_previous_edit_dot',
    'base_update_previous_edit_cosine', 'diagnostics_extra_resident_bytes',
)
CONFIG_FIELDS = ('seed', 'max_steps', 'schedule_steps', 'model', 'batch_tokens',
                 'seq_len', 'microbatch_sequences', 'val_tokens', 'world_size', 'compile')
ENV_FIELDS = ('readout', 'gpu', 'torch', 'cuda', 'python', 'gradient_units')
# Only these documented observers are ignored; unknown settings remain scientific.
OBSERVATION_CONFIG_FIELDS = {'wandb', 'checkpoint_every', 'fingerprint_steps'}
OBSERVATION_REWRITER_FIELDS = {'diagnostics'}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def reject_constant(value):
    raise ValueError(f'nonfinite JSON constant: {value}')


def load_json(path, warnings):
    try:
        return json.loads(path.read_text(), parse_constant=reject_constant)
    except (OSError, ValueError) as exc:
        warnings.append(f'{path}: {exc}')
        return None


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:12]


def stats(values):
    values = [float(v) for v in values if finite(v)]
    if not values:
        return None
    return dict(count=len(values), first=values[0], last=values[-1],
                min=min(values), max=max(values), mean=statistics.fmean(values))


def dataset_identity(config, config_path, warnings):
    path = config_path.parent / 'data_manifest.json'
    if path.exists():
        manifest = load_json(path, warnings)
        if (isinstance(manifest, list) and manifest and all(
                isinstance(row, dict) and isinstance(row.get('sha256'), str) and len(row['sha256']) == 64
                and row.get('kind') in {'train', 'val'} for row in manifest)
                and {row['kind'] for row in manifest} == {'train', 'val'}):
            # Local staging directories can differ; order, contents, and consumption cannot.
            rows = [{key: row.get(key) for key in ('kind', 'sha256', 'tokens', 'bytes', 'consumed_batches')}
                    for row in manifest]
            return dict(verified=True, identity=rows, source=str(path))
        warnings.append('Dataset manifest missing content hashes; data identity remains unverified.')
    return dict(verified=False, identity={k: config.get(k) for k in ('train_data', 'val_data')},
                source='configured paths only')


def summarize(label, metrics_path, config_path, environment_path, checkpoint=None):
    warnings = []
    config = load_json(config_path, warnings) or {}
    environment = load_json(environment_path, warnings) or {}
    if not isinstance(config, dict):
        warnings.append('Configuration must be a JSON object.'); config = {}
    if not isinstance(environment, dict):
        warnings.append('Environment must be a JSON object.'); environment = {}
    records, bad_lines = [], 0
    try:
        with metrics_path.open() as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line, parse_constant=reject_constant)
                    if not isinstance(row, dict):
                        raise ValueError('record must be an object')
                    records.append(row)
                except ValueError as exc:
                    bad_lines += 1
                    warnings.append(f'metrics line {line_no}: {exc}')
    except OSError as exc:
        warnings.append(str(exc))
    # Last duplicate wins after a legitimate resume; duplicates remain disclosed.
    train, validation = {}, {}
    for row in records:
        if row.get('event') in {'train', 'validation'} and isinstance(row.get('step'), int):
            target = train if row['event'] == 'train' else validation
            if row['step'] in target:
                warnings.append(f'Duplicate {row["event"]} step {row["step"]}; using last record.')
            target[row['step']] = row
    steps = sorted(train)
    rows = [train[s] for s in steps]
    endpoint = config.get('max_steps')
    if not isinstance(endpoint, int) or isinstance(endpoint, bool) or endpoint < 1:
        warnings.append('Missing or invalid positive integer max_steps.')
        endpoint = None
    requested = endpoint if checkpoint is None else checkpoint
    final = next((r for r in reversed(records) if r.get('event') == 'finished'), {})
    complete = (isinstance(endpoint, int) and endpoint > 0 and final.get('steps') == endpoint
                and bool(steps) and steps[-1] == endpoint and not bad_lines)
    accounting = dict(expected_per_rank=None, forwards=None, backwards=None, valid=None)
    try:
        numerator = endpoint * config['batch_tokens']
        denominator = config['world_size'] * config['seq_len'] * config['microbatch_sequences']
        if denominator <= 0 or numerator % denominator:
            raise ValueError('nonintegral prescribed microbatch count')
        accounting['expected_per_rank'] = numerator // denominator
        for kind, key in [('forwards', 'training_forwards_per_rank'), ('backwards', 'training_backwards_per_rank')]:
            accounting[kind] = final.get(key, rows[-1].get(key) if rows else None)
        if accounting['forwards'] is not None and accounting['backwards'] is not None:
            accounting['valid'] = (accounting['forwards'] == accounting['backwards'] == accounting['expected_per_rank'])
            if complete and not accounting['valid']:
                warnings.append('Finished run has unexpected main-model computation counts.')
                complete = False
    except (KeyError, TypeError, ZeroDivisionError, ValueError) as exc:
        warnings.append(f'Cannot certify prescribed computation counts: {exc}')
    observed_keys = set(OBSERVABLES)
    observed_keys.update(k for row in rows for k in row if k.startswith((
        'allones_', 'realized_squared_norm_change', 'forward_', 'proposed_edit_norm',
        'realized_edit_norm_', 'casting_error_norm_')))
    measurements = {key: result for key in sorted(observed_keys)
                    if (result := stats([row.get(key) for row in rows])) is not None}
    runtime = max((r['elapsed_seconds'] for r in records if finite(r.get('elapsed_seconds'))), default=None)
    peak = max((r['peak_memory_bytes'] for r in records if finite(r.get('peak_memory_bytes'))), default=None)
    ratios = [r['realized_edit_norm'] / r['committed_norm'] for r in rows
              if finite(r.get('realized_edit_norm')) and finite(r.get('committed_norm')) and r['committed_norm'] > 0]
    rounding = [r['casting_error_norm'] / r['committed_norm'] for r in rows
                if finite(r.get('casting_error_norm')) and finite(r.get('committed_norm')) and r['committed_norm'] > 0]
    loss = validation.get(requested, {}).get('val_loss')
    if not finite(loss):
        loss = None
    if loss is None:
        warnings.append(f'No finite validation loss at requested checkpoint {requested}.')
    data = dataset_identity(config, config_path, warnings)
    return dict(label=label, mode=config.get('mode', 'unknown'), config=config, environment=environment,
        paths=dict(metrics=str(metrics_path), config=str(config_path), environment=str(environment_path)),
        status='completed' if complete else 'incomplete_or_failed', completed=complete,
        last_train_step=steps[-1] if steps else None, requested_checkpoint=requested,
        validation_loss_at_checkpoint=loss,
        validation=[dict(step=s, loss=validation[s].get('val_loss')) for s in sorted(validation)],
        last_train_loss=rows[-1].get('train_loss') if rows else None,
        elapsed_seconds=runtime, peak_memory_bytes=peak, measurements=measurements,
        realized_to_proposed_norm=stats(ratios), casting_error_relative_to_proposed_norm=stats(rounding),
        computation_accounting=accounting, dataset=data, warnings=warnings,
        parsed_records=len(records), invalid_metric_lines=bad_lines,
        terminal_identity=rows[-1].get('terminal_identity') if rows else None)



def scientific_config(config, *, retain_seed=True, retain_treatment=True):
    """Remove a narrow observer allowlist; preserve every other configuration key."""
    result = {k:v for k,v in config.items() if k not in
              OBSERVATION_CONFIG_FIELDS | {'train_data', 'val_data'}}
    result.setdefault('schedule_steps', result.get('max_steps'))
    result.setdefault('compile', True)
    result.setdefault('compile_loss', False)
    result.setdefault('deterministic', False)
    result.setdefault('unscored', False)
    if not retain_seed:
        result.pop('seed', None)
    if retain_treatment:
        rewrite = {k:v for k,v in result.get('rewriter', {}).items()
                   if k not in OBSERVATION_REWRITER_FIELDS}
        # Baseline mode never executes rewriter settings.
        result['rewriter'] = {} if result.get('mode') == 'baseline' else rewrite
    else:
        result.pop('mode', None)
        result.pop('rewriter', None)
    return result


def observation_settings(run):
    return dict(top_level={k:run['config'].get(k) for k in sorted(OBSERVATION_CONFIG_FIELDS)},
                rewriter={k:run['config'].get('rewriter', {}).get(k, False)
                          for k in sorted(OBSERVATION_REWRITER_FIELDS)})


def compare(control, treatment):
    mismatches, missing = {}, []
    for field in CONFIG_FIELDS:
        a, b = control['config'].get(field), treatment['config'].get(field)
        if field == 'schedule_steps':
            a = control['config'].get(field, control['config'].get('max_steps'))
            b = treatment['config'].get(field, treatment['config'].get('max_steps'))
        if a is None or b is None:
            missing.append(f'config.{field}')
        elif a != b:
            mismatches[f'config.{field}'] = [a, b]
    for field in ENV_FIELDS:
        a, b = control['environment'].get(field), treatment['environment'].get(field)
        if a is None or b is None:
            missing.append(f'environment.{field}')
        elif a != b:
            mismatches[f'environment.{field}'] = [a, b]
    a_cfg = scientific_config(control['config'], retain_treatment=False)
    b_cfg = scientific_config(treatment['config'], retain_treatment=False)
    for key in a_cfg.keys() | b_cfg.keys():
        if a_cfg.get(key) != b_cfg.get(key):
            mismatches[f'config.{key}'] = [a_cfg.get(key), b_cfg.get(key)]
    # Execution controls include SDPA, TF32, launch variables, and deterministic flags.
    a_execution, b_execution = control['environment'].get('execution'), treatment['environment'].get('execution')
    if a_execution != b_execution:
        mismatches['environment.execution'] = [a_execution, b_execution]
    if (a_cfg.get('deterministic') or b_cfg.get('deterministic')) and (a_execution is None or b_execution is None):
        missing.append('recorded deterministic execution environment')
    if control['dataset']['verified'] != treatment['dataset']['verified']:
        missing.append('matching verified dataset identities')
    elif control['dataset']['identity'] != treatment['dataset']['identity']:
        mismatches['dataset'] = [control['dataset']['identity'], treatment['dataset']['identity']]
    elif not control['dataset']['verified']:
        missing.append('content-verified dataset identity')
    if control['requested_checkpoint'] != treatment['requested_checkpoint']:
        mismatches['checkpoint'] = [control['requested_checkpoint'], treatment['requested_checkpoint']]
    losses = [control['validation_loss_at_checkpoint'], treatment['validation_loss_at_checkpoint']]
    if any(v is None for v in losses):
        missing.append('finite common-checkpoint validation losses')
    if not control['completed'] or not treatment['completed']:
        missing.append('both runs completed')
    if any(r['computation_accounting']['valid'] is not True for r in (control, treatment)):
        missing.append('verified prescribed computation counts')
    eligible = not mismatches and not missing
    return dict(control=control['label'], treatment=treatment['label'],
                seed=treatment['config'].get('seed'), checkpoint=treatment['requested_checkpoint'],
                eligible=eligible, mismatches=mismatches, missing=missing,
                observation_settings_differ=observation_settings(control) != observation_settings(treatment),
                observation_settings=dict(control=observation_settings(control), treatment=observation_settings(treatment)),
                runtime_comparison_caveat='Runtime and memory include enabled observers; observer differences are not training-algorithm changes.',
                treatment_minus_control=(losses[1]-losses[0]) if eligible else None,
                interpretation=('Descriptive paired difference only; one pair does not establish statistical improvement.'
                                if eligible else 'No matched improvement comparison: missing or mismatched evidence.'))


def acceptance_group(runs):
    checkpoints = {r['requested_checkpoint'] for r in runs}
    seeds = [r['config'].get('seed') for r in runs]
    reasons = []
    if len(checkpoints) != 1 or None in checkpoints:
        reasons.append('No single common predeclared checkpoint.')
    if not all(r['completed'] for r in runs):
        reasons.append('Not all runs completed.')
    if not all(r['validation_loss_at_checkpoint'] is not None for r in runs):
        reasons.append('Missing finite validation loss at common checkpoint.')
    if None in seeds or len(set(seeds)) != len(seeds):
        reasons.append('Independent distinct seeds are not established; duplicate seeds are not counted as independent trials.')
    if not all(r['dataset']['verified'] for r in runs):
        reasons.append('Dataset identities are not content verified.')
    if not all(r['computation_accounting']['valid'] is True for r in runs):
        reasons.append('Main-model computation counts are not verified.')
    if any(r['config'].get('unscored', False) for r in runs):
        reasons.append('At least one run is explicitly unscored.')
    result = dict(runs=[r['label'] for r in runs], n=len(runs), eligible=not reasons,
                  reasons=reasons, checkpoint=next(iter(checkpoints)) if len(checkpoints)==1 else None,
                  mean_validation_loss=None, threshold_margin_times_sqrt_n=None, formula_satisfied=None,
                  interpretation='Arithmetic rule check only; not benchmark certification or improvement over a matched baseline.')
    if not reasons:
        mean = statistics.fmean(r['validation_loss_at_checkpoint'] for r in runs)
        margin = (3.28-mean) * math.sqrt(len(runs))
        result.update(mean_validation_loss=mean, threshold_margin_times_sqrt_n=margin,
                      formula_satisfied=(mean < 3.28 and margin >= .004))
    return result


def group_key(run):
    # Seeds and a narrow allowlist of passive observers may differ. Algorithm,
    # execution policy, model/data/readout and all unknown settings remain matched.
    cfg = scientific_config(run['config'], retain_seed=False)
    return digest(dict(config=cfg, environment=run['environment'], dataset=run['dataset']['identity'],
                       requested_checkpoint=run['requested_checkpoint']))


def aggregate_pairs(comparisons, runs):
    by_label = {run['label']:run for run in runs}
    groups = defaultdict(list)
    for pair in comparisons:
        key = (group_key(by_label[pair['control']]), group_key(by_label[pair['treatment']]))
        groups[key].append(pair)
    results = []
    for pairs in groups.values():
        seeds = [pair['seed'] for pair in pairs]
        distinct = None not in seeds and len(set(seeds)) == len(seeds)
        eligible = all(pair['eligible'] for pair in pairs) and distinct
        differences = [pair['treatment_minus_control'] for pair in pairs] if eligible else []
        result = dict(pairs=[dict(control=p['control'], treatment=p['treatment'], seed=p['seed']) for p in pairs],
            n_pairs=len(pairs), seeds=seeds, all_pairs_eligible=eligible,
            completed_eligible_pairs=sum(pair['eligible'] for pair in pairs),
            duplicate_or_unknown_seed=not distinct, mean_paired_difference=None,
            sample_stddev=None, standard_error=None, all_differences=None,
            interpretation=('Descriptive independent paired-seed summary; two pairs remain limited evidence. '
                            'No significance test or robust-improvement claim is implied.'))
        if eligible:
            spread = statistics.stdev(differences) if len(differences)>1 else None
            result.update(mean_paired_difference=statistics.fmean(differences), sample_stddev=spread,
                          standard_error=spread/math.sqrt(len(differences)) if spread is not None else None,
                          all_differences=differences)
        results.append(result)
    return results


def number(value, digits=6):
    return 'unavailable' if value is None else f'{value:.{digits}f}'


def markdown(report):
    lines = ['# Recorded experiment summary', '',
        f'Generated {report["generated_at"]}; summarizer version {VERSION}.', '',
        'This report uses recorded artifacts only. Missing or unfinished runs are retained. '
        'No statistical improvement is inferred from a single paired seed, and a threshold calculation is not benchmark certification.', '',
        '| Run | Status | Steps | Checkpoint | Validation loss | Runtime (s) | Peak GPU memory (GiB) |',
        '|---|---|---:|---:|---:|---:|---:|']
    for run in report['runs']:
        label = run['label'].replace('|', '\\|')
        memory = run['peak_memory_bytes'] / 2**30 if run['peak_memory_bytes'] is not None else None
        lines.append(f'| {label} | {run["status"]} | {run["last_train_step"]} | {run["requested_checkpoint"]} | '
                     f'{number(run["validation_loss_at_checkpoint"])} | {number(run["elapsed_seconds"], 1)} | {number(memory, 2)} |')
    lines += ['', '## Matched paired comparisons', '']
    if not report['comparisons']:
        lines.append('No baseline/treatment pairs were available.')
    for pair in report['comparisons']:
        lines.append(f'- **{pair["treatment"]} versus {pair["control"]}**, seed {pair["seed"]}: ' +
            (f'validation difference {pair["treatment_minus_control"]:+.6f} nats/token at checkpoint {pair["checkpoint"]}. '
             'This is a descriptive paired difference.' if pair['eligible'] else
             f'not eligible; mismatches: {list(pair["mismatches"])}; missing: {pair["missing"]}.'))
    lines += ['', '## Independent paired-seed summaries', '']
    for group in report.get('paired_seed_groups', []):
        if group['all_pairs_eligible']:
            lines.append(f'- Seeds {group["seeds"]}: n={group["n_pairs"]}, mean paired difference '
                         f'{group["mean_paired_difference"]:+.6f}, sample SD {number(group["sample_stddev"])}, '
                         f'SE {number(group["standard_error"])} nats/token. All differences: {group["all_differences"]}.')
        else:
            lines.append(f'- Seeds {group["seeds"]}: aggregate withheld; only {group["completed_eligible_pairs"]} '
                         f'of {group["n_pairs"]} pairs are eligible, or seeds are duplicated/unknown.')
    lines += ['', 'Two paired seeds remain limited evidence; these are descriptive statistics, not a significance claim. '
              'Passive diagnostic differences are disclosed in the JSON and affect runtime/memory comparisons.', '',
              '## Common-checkpoint acceptance arithmetic', '']
    for group in report['acceptance_groups']:
        names = ', '.join(group['runs'])
        if group['eligible']:
            lines.append(f'- {names}: n={group["n"]}, mean={group["mean_validation_loss"]:.6f}, '
                         f'(3.28−mean)√n={group["threshold_margin_times_sqrt_n"]:.6f}; '
                         f'formula satisfied: {group["formula_satisfied"]}.')
        else:
            lines.append(f'- {names}: not evaluated. {" ".join(group["reasons"])}')
    lines += ['', '## Controller and accounting observations', '']
    for run in report['runs']:
        lines += [f'### {run["label"]}', '', f'Last ordinary training loss: {number(run["last_train_loss"])}.', '',
                  '| Measurement | Count | First | Last | Min | Max | Mean |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for key, values in run['measurements'].items():
            lines.append('| '+key+' | '+str(values['count'])+' | '+' | '.join(number(values[k]) for k in ('first','last','min','max','mean'))+' |')
        lines += ['', f'Computation accounting: `{json.dumps(run["computation_accounting"], sort_keys=True)}`.',
                  f'Realized/proposed norm ratio: `{json.dumps(run["realized_to_proposed_norm"], sort_keys=True)}`.',
                  f'Casting-error/proposed norm ratio: `{json.dumps(run["casting_error_relative_to_proposed_norm"], sort_keys=True)}`.',
                  f'Terminal identity recorded: `{run["terminal_identity"]}`.', '',
                  f'Artifacts: `{run["paths"]["metrics"]}`, `{run["paths"]["config"]}`, `{run["paths"]["environment"]}`.', '']
        if run['warnings']:
            lines += ['Warnings:', ''] + ['- '+w for w in run['warnings']] + ['']
    if report.get('plot') == 'validation.png':
        lines += ['## Recorded validation trajectories', '', '![Recorded validation trajectories](validation.png)', '']
    lines += ['`task_loss`, when present in raw logs, is a controller VJP contraction, not an additional task-loss evaluation. '
              'Norm changes are proposed changes unless explicitly logged as realized. Peak memory is the recorded CUDA allocation peak, not total device occupancy.', '']
    return '\n'.join(lines)


def plot_runs(runs, output):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return 'matplotlib is unavailable; no plot generated.'
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for run in runs:
        pts = [r for r in run['validation'] if finite(r['loss'])]
        if pts:
            ax.plot([r['step'] for r in pts], [r['loss'] for r in pts], marker='.', label=run['label'])
    ax.set(xlabel='Completed GPT iterations', ylabel='Validation loss (nats/token)', title='Recorded validation trajectories')
    if ax.lines:
        ax.legend()
    fig.tight_layout(); fig.savefig(output / 'validation.png', dpi=160); plt.close(fig)
    return 'validation.png'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run_dirs', nargs='*', type=Path, help='Directories containing metrics.jsonl, config.json, environment.json')
    parser.add_argument('--run', nargs=4, action='append', default=[], metavar=('LABEL', 'METRICS', 'CONFIG', 'ENVIRONMENT'))
    parser.add_argument('--output', type=Path, required=True, help='Explicit directory for summary.json and report.md')
    parser.add_argument('--checkpoint', type=int, help='Predeclared common checkpoint; default is each configured endpoint')
    parser.add_argument('--baseline', action='append', default=[], help='Control run label; otherwise config.mode=baseline identifies controls')
    parser.add_argument('--plot', action='store_true', help='Generate standalone validation.png when matplotlib is available')
    args = parser.parse_args(argv)
    inputs = [(p.name, p/'metrics.jsonl', p/'config.json', p/'environment.json') for p in args.run_dirs]
    inputs += [(label, Path(m), Path(c), Path(e)) for label,m,c,e in args.run]
    if not inputs:
        parser.error('provide at least one run directory or --run')
    if len({row[0] for row in inputs}) != len(inputs):
        parser.error('run labels must be unique; use --run for duplicate directory basenames')
    if args.checkpoint is not None and args.checkpoint < 0:
        parser.error('--checkpoint must be nonnegative')
    runs = [summarize(*row, checkpoint=args.checkpoint) for row in inputs]
    controls = [r for r in runs if r['label'] in args.baseline] if args.baseline else [r for r in runs if r['mode']=='baseline']
    unknown = set(args.baseline)-{r['label'] for r in runs}
    if unknown:
        parser.error(f'unknown baseline labels: {sorted(unknown)}')
    comparisons = []
    for treatment in runs:
        if treatment in controls:
            continue
        matches = [r for r in controls if r['config'].get('seed')==treatment['config'].get('seed')]
        for control in matches or controls:
            comparisons.append(compare(control, treatment))
    groups = defaultdict(list)
    for run in runs:
        groups[group_key(run)].append(run)
    report = dict(schema_version=VERSION, generated_at=datetime.now(timezone.utc).isoformat(),
                  runs=runs, comparisons=comparisons, paired_seed_groups=aggregate_pairs(comparisons, runs),
                  matching_policy=dict(ignored_top_level_observers=sorted(OBSERVATION_CONFIG_FIELDS),
                                       ignored_rewriter_observers=sorted(OBSERVATION_REWRITER_FIELDS),
                                       all_other_settings_preserved=True),
                  acceptance_groups=[acceptance_group(group) for group in groups.values()])
    args.output.mkdir(parents=True, exist_ok=True)
    if args.plot:
        report['plot'] = plot_runs(runs, args.output)
    (args.output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (args.output/'report.md').write_text(markdown(report))
    print(json.dumps(dict(summary=str(args.output/'summary.json'), report=str(args.output/'report.md'),
                          runs=len(runs), completed=sum(r['completed'] for r in runs)), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
