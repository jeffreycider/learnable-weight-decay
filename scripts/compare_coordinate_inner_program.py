"""Exact-state gate for tied frames, private fitting, and staged restart.

This compares a bounded execution fixture; zero scientific rank changes are
reported independently and are not inferred from backend equivalence.
"""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.qualify_coordinate_fullbank import read_joint
from rewriter.distributed_probe_store import _digest


def visit_gates(updates, inner_steps):
    visits=[v for row in updates for v in row['rows']]
    clocks={};clock_ok=True;disjoint=True;finite=True
    for visit in visits:
        key=(visit['domain'],visit['local_id'])
        expected=clocks.get(key,0)+inner_steps+1
        clock_ok &= visit['private_visit']==expected and visit['private_updates_this_visit']==inner_steps+1
        clocks[key]=expected
        inner=visit['private_inner_rows']
        clock_ok &= len(inner)==inner_steps
        joint_ids=set(visit['context_ids'])
        disjoint &= bool(joint_ids) and all(row['context_ids'] and not joint_ids.intersection(row['context_ids']) for row in inner)
        finite &= all(math.isfinite(float(row[k])) for row in [*inner,visit] for k in ('objective','mean_kl','private_preclip_gradient_norm'))
    return dict(private_tick_accounting=clock_ok,disjoint_support_and_joint=disjoint,finite_losses_and_gradients=finite,
                private_revisits=bool(clocks) and all(v>=2*(inner_steps+1) for v in clocks.values()),
                equal_domain_weights=all(abs(sum(v['state_weight'] for v in row['rows'] if v['domain']==d)-.5)<1e-12 for row in updates for d in ('gpt','cifar10')))


def numeric_visit(row):
    """Execution timing is not a numerical model output."""
    return {k:v for k,v in row.items() if not k.endswith('_seconds') and k!='private_snapshot_bytes'}


def compare(reference,candidate,resumed,output,*,inner_steps=2):
    paths={k:Path(v) for k,v in [('reference',reference),('candidate',candidate),('resumed',resumed)]}
    states={k:read_joint(p) for k,p in paths.items()}
    metrics={k:[json.loads(s) for s in (p/'metrics.jsonl').read_text().splitlines()] for k,p in paths.items()}
    updates={k:[r for r in rows if r['event']=='joint_update'] for k,rows in metrics.items()}
    result=dict(scope='Bounded tied-Q/private-inner execution and restart; no scientific convergence claim',world_size=len(states['reference'][1]),comparisons={},arms={})
    for a,b in [('reference','candidate'),('candidate','resumed')]:
        result['comparisons'][a+'_vs_'+b]=dict(
            shared_Adam_rank_RNG_exact=_digest(states[a][0])==_digest(states[b][0]),
            sampler_duals_compute_bytes_exact=_digest(states[a][1])==_digest(states[b][1]),
            private_parameters_Adam_clocks_exact=_digest(states[a][2])==_digest(states[b][2]),
            loss_gradient_strict_rank_rows_exact=_digest([[numeric_visit(v) for v in r['rows']] for r in updates[a]])==_digest([[numeric_visit(v) for v in r['rows']] for r in updates[b]]))
    for name,rows in updates.items():
        state=states[name];gates=visit_gates(rows,inner_steps)
        gates['tied_private_layout']=bool(state[2]) and all(v['private']['_extra_state'].get('frame_mode')=='tied_orthonormal' for v in state[2].values())
        final=[r for r in metrics[name] if r['event']=='full_fitting_sweep']
        result['arms'][name]=dict(gates=gates,updates=len(rows),wall_seconds=[r['wall_seconds'] for r in rows],
            strict_changes_during_joint=[sum(v['strict_topk_changes'] for v in r['rows']) for r in rows],final_fitting_summary=final[-1] if final else None)
    result['all_gates']=all(all(v.values()) for v in result['comparisons'].values()) and all(all(v['gates'].values()) for v in result['arms'].values())
    Path(output).write_text(json.dumps(result,indent=2)+'\n')
    if not result['all_gates']:raise RuntimeError('new coordinate program execution gate failed')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('reference','candidate','resumed','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--inner-steps',type=int,default=2);a=p.parse_args()
    compare(a.reference,a.candidate,a.resumed,a.output,inner_steps=a.inner_steps)
