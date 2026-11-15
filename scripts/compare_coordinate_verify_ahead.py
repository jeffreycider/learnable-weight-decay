"""Exact-state and complete-wall comparison of verification-only lookahead."""
from pathlib import Path
import argparse,json,statistics,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.qualify_coordinate_fullbank import read_joint
from rewriter.distributed_probe_store import _digest


def compare(paths,output):
    reference=None;rows=[]
    for path in paths:
        path=Path(path);state=read_joint(path)
        signatures=tuple(_digest(value) for value in state)
        if reference is None:reference=signatures
        metrics=[json.loads(line) for line in (path/'metrics.jsonl').read_text().splitlines()]
        updates=[row for row in metrics if row['event']=='joint_update'];visits=[v for row in updates for v in row['rows']]
        evidence=[json.loads(p.read_text()) for p in sorted(path.glob('verification-lookahead-rank*.json'))]
        gates=dict(shared_Adam_RNG_bitexact=signatures[0]==reference[0],sampler_dual_compute_physical_bytes_bitexact=signatures[1]==reference[1],
                   private_Adam_clocks_bitexact=signatures[2]==reference[2],nine_updates=len(updates)==9,
                   exact_workload=len(visits)==36 and len({(v['domain'],v['local_id']) for v in visits})==32,
                   weighting=all(abs(sum(v['state_weight'] for v in row['rows'] if v['domain']==d)-.5)<1e-12 for row in updates for d in ('gpt','cifar10')))
        rows.append(dict(path=str(path),enabled=evidence[0]['enabled'],gates=gates,
                         wall_seconds=[row['wall_seconds'] for row in updates],
                         first_eight_median_seconds=statistics.median(row['wall_seconds'] for row in updates[:8]),
                         revisit_seconds=updates[-1]['wall_seconds'],verification=evidence))
    summary=dict(scope='First-visit verification in fresh reader processes; OS/file caches may be warm. Identical pinned snapshot transport in both arms. Verification waits are included in full-update wall.',
                 arms=rows,all_gates=all(all(row['gates'].values()) for row in rows),
                 mode_medians={str(mode):statistics.median(v for row in rows if row['enabled']==mode for v in row['wall_seconds'][:8]) for mode in (False,True)})
    Path(output).write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
    if not summary['all_gates']:raise RuntimeError('verification overlap changed numeric state or workload')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--paths',nargs='+',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();compare(a.paths,a.output)
