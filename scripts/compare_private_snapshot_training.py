"""Exact numeric cross-backend and pending-store resume comparison."""
import argparse,copy,json,statistics
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.qualify_coordinate_fullbank import read_joint
from rewriter.distributed_probe_store import _digest


def numeric_rank_states(states):
    result=copy.deepcopy(states)
    physical={rank:state.pop('private_written_bytes') for rank,state in result.items()}
    return result,physical


def compare(reference,candidate,resumed,output):
    paths={'reference':Path(reference),'candidate':Path(candidate),'resumed':Path(resumed)}
    values={name:read_joint(path) for name,path in paths.items()}
    comparisons={}
    for left,right in [('reference','candidate'),('candidate','resumed')]:
        a,b=values[left],values[right]; ar,abytes=numeric_rank_states(a[1]);br,bbytes=numeric_rank_states(b[1])
        gates=dict(shared_parameters_Adam_rank_RNG_bitexact=_digest(a[0])==_digest(b[0]),
            private_parameters_Adam_clocks_bitexact=_digest(a[2])==_digest(b[2]),
            sampler_duals_compute_visits_bitexact=_digest(ar)==_digest(br))
        if left=='candidate':gates['same_backend_physical_bytes_equal']=abytes==bbytes
        comparisons[left+'_vs_'+right]=dict(gates=gates,physical_private_written_bytes={left:abytes,right:bbytes})
    runtimes={}
    for name,path in paths.items():
        rows=[json.loads(line) for line in (path/'metrics.jsonl').read_text().splitlines()]
        updates=[r for r in rows if r['event']=='joint_update'];visits=[v for r in updates for v in r['rows']]
        runtimes[name]=dict(wall_seconds=[r['wall_seconds'] for r in updates],
            first_eight_distinct_update_median_seconds=statistics.median(r['wall_seconds'] for r in updates[:8]),
            revisit_update_seconds=updates[-1]['wall_seconds'],
            phases_per_visit={d:{k:statistics.mean(v[k] for v in visits if v['domain']==d) for k in
                ('context_load_seconds','private_stage_seconds','finite_measure_seconds','private_adam_seconds','private_snapshot_seconds')}
                for d in ('gpt','cifar10')})
    result=dict(scope='Actual two-rank 32-distinct-state plus four-revisit fullbank updates; physical serialized byte counts are reported separately, never normalized into numeric/Adam/clock equality.',
                comparisons=comparisons,runtimes=runtimes,
                all_gates=all(all(row['gates'].values()) for row in comparisons.values()))
    Path(output).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    if not result['all_gates']:raise RuntimeError('snapshot full-step or resume parity failed')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('reference','candidate','resumed','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();compare(a.reference,a.candidate,a.resumed,a.output)
