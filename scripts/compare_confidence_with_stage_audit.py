"""Read-only consistency check: confidence-audit rows vs the trainer's own audit.

The repair stage's fixed_training audit evaluates a deterministic set of 32
fitting checkpoints on all 128 seen sequences; the phase-1 confidence audit of
the same frozen policy evaluates every fitting checkpoint on the same cached
panels through an independent code path (plain engine, reference grouped norm,
per-token FP64 KL). Restricted to those 32 checkpoints, the mean of checkpoint
means and the maximum must agree up to floating-point summation order. This
validates the audit pipeline on the real policy; it selects nothing.
"""
import argparse,json
from pathlib import Path


def compare(rows,state_ids,expected_mean,expected_max,population='fitting_seen'):
    by={r['state_id']:r for r in rows if r['population']==population}
    missing=[s for s in state_ids if s not in by]
    if missing:raise ValueError(f'confidence rows lack states {missing[:5]}...')
    means=[by[s]['summary']['mean_kl'] for s in state_ids]
    mean=sum(means)/len(means);maximum=max(means)
    return dict(population=population,states=len(state_ids),confidence_mean_of_means=mean,stage_mean=expected_mean,
                mean_relative_difference=(mean-expected_mean)/expected_mean if expected_mean else None,
                confidence_max=maximum,stage_max=expected_max,
                max_relative_difference=(maximum-expected_max)/expected_max if expected_max else None,
                worst_state=max(state_ids,key=lambda s:by[s]['summary']['mean_kl']),
                shrink_mean=sum(by[s]['mean_group_l2_percentage_reduction'] for s in state_ids)/len(state_ids))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--states',type=Path,required=True,help='confidence audit states.jsonl')
    p.add_argument('--history',type=Path,required=True,help='repair audit history JSONL with state_ids (runs/chisel-fable/repair_v2_audit_history.jsonl)')
    p.add_argument('--update',type=int,required=True,help='repair update whose fixed_training audit to compare against (e.g. 8)')
    p.add_argument('--role',default='fixed_training')
    p.add_argument('--population',default='fitting_seen')
    a=p.parse_args(argv)
    rows=[json.loads(l) for l in a.states.read_text().splitlines() if l.strip()]
    audits=[json.loads(l) for l in a.history.read_text().splitlines() if l.strip()]
    stage=[x for x in audits if x['update']==a.update and x['role']==a.role and x.get('state_ids')]
    if not stage:raise ValueError('no stage audit row with state IDs for that update/role in the history file')
    stage=stage[-1]
    result=compare(rows,stage['state_ids'],stage['mean_kl'],stage.get('maximum_kl'),a.population)
    result.update(stage_update=a.update,stage_role=a.role,stage_shrink=stage.get('mean_group_l2_percentage_reduction'))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
