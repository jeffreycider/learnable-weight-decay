"""Read-only numerical triage of complete OR aborted full200 benchmarks.

This reports existing driver gates, never grants qualification or derives a speedup
from partial evidence. Checkpoint maxima are over200 unique states, not rank copies.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rewriter.qualification_metrics import metric_within


def diagnose(metrics, *, tolerance, dual_lr, objective_relative_limit=0., kl_abs_limit=None):
    if objective_relative_limit not in (0.,1e-3):raise ValueError('unknown objective gate policy')
    kl_abs_limit=1e-9 if kl_abs_limit is None else kl_abs_limit
    failures=[]
    for key,cosine in [('preclip',.99999),('clipped',.99999),('adam_delta',None)]:
        if not metric_within(metrics[key],.003,cosine):failures.append(key)
    for key in ('exp_avg','exp_avg_sq'):
        entry=metrics['adam'][key]
        for label,m in [('global',entry['global_metric']),*[(f'group{i}',m) for i,m in enumerate(entry['groups'])]]:
            if not metric_within(m,.003):failures.append(f'adam.{key}.{label}')
    maxima={}
    for key in ('training_rows','post_update_rows'):
        rows=metrics[key]
        if len(rows)!=200 or len({r['state_id'] for r in rows})!=200:
            raise ValueError('exactly200 unique checkpoint rows required')
        maxima[key]={}
        for field,limit in [('kl_difference',kl_abs_limit),('objective_difference',1e-4)]:
            if not all(math.isfinite(r[field]) for r in rows):raise ValueError('nonfinite checkpoint error')
            worst=max(rows,key=lambda r:abs(r[field]))
            def bound(row):
                if field=='objective_difference' and objective_relative_limit:
                    ref=row['reference_objective']
                    if not math.isfinite(ref):raise ValueError('nonfinite reference objective')
                    return limit+objective_relative_limit*abs(ref)
                return limit
            exceeded=[r['state_id'] for r in rows if abs(r[field])>bound(r)]
            maxima[key][field]=dict(state_id=worst['state_id'],signed_difference=worst[field],
                absolute_difference=abs(worst[field]),limit=bound(worst),absolute_floor=limit,relative_limit=objective_relative_limit if field=='objective_difference' else 0.,exceeded_count=len(exceeded),exceeded_state_ids=exceeded)
            if exceeded:failures.append(f'{key}.{field}')
    dual_limit=dual_lr*1e-9/tolerance+1e-12
    if not metrics['dual_finite'] or not metrics['dual_max_difference']<=dual_limit:failures.append('dual')
    if metrics.get('reference_repeat_bit_identical') is False:failures.append('reference_repeat')
    if bool(failures)==metrics['within_declared_global_gates']:
        raise ValueError('reconstructed gates contradict saved verdict')
    return dict(failed_gates=failures,maxima=maxima,dual_limit=dual_limit,
        dual_max_difference=metrics['dual_max_difference'],
        gradient_metrics={k:metrics[k] for k in ('preclip','clipped','adam_delta')})


def analyze(directory):
    root=Path(directory);startup=json.loads((root/'startup.json').read_text())
    options=startup['options'];result={}
    confirmation=startup.get('benchmark_objective',{}).get('independent_confirmation')
    kwargs={}
    if confirmation:
        if (confirmation['objective_abs_floor']!=1e-4 or confirmation['objective_relative_limit']!=1e-3
                or confirmation['kl_abs_limit']!=.1*options['kl_tolerance']):
            raise ValueError('unknown prospective gate policy')
        kwargs=dict(objective_relative_limit=1e-3,kl_abs_limit=confirmation['kl_abs_limit'])
    for path in sorted(root.glob('qualification_*_rank*.json')):
        metrics=json.loads(path.read_text())
        result[path.name]=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            **diagnose(metrics,tolerance=options['kl_tolerance'],dual_lr=options['dual_lr'],**kwargs))
    return dict(scope='per-rank gate diagnostics; partial output is not full qualification or timing evidence',
        independent_confirmation=confirmation,completed_summary_present=(root/'summary.json').exists(),rank_records=len(result),records=result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();a.output.write_text(json.dumps(analyze(a.directory),indent=2,allow_nan=False)+'\n')
