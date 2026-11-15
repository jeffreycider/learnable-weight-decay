"""Bounded CPU-only multi-channel gauge feasibility; no model execution."""
import argparse,json,hashlib,time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from rewriter.gauge_energy_diagnostic import binary_coordinate_search


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--energies',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--max-sweeps',type=int,default=8);args=p.parse_args()
    torch.set_num_threads(1);started=time.perf_counter();raw=torch.load(args.energies,map_location='cpu',weights_only=False)
    rows=[]
    for state,record in raw['states'].items():
        for kind,power in [('attention',1),('mlp',2)]:
            bank=record['families'][kind]
            result=binary_coordinate_search(*[v.numpy() for v in bank['energies']],inverse_power=power,max_sweeps=args.max_sweeps)
            result.update(state_id=state,state_sha256=record['state_sha256'],family=kind,channels=bank['channels'])
            rows.append(result)
            print(json.dumps({k:v for k,v in result.items() if k not in ('channels','channel_exponents')}),flush=True)
    combined=[]
    for state in raw['states']:
        selected=[r for r in rows if r['state_id']==state]
        combined.append(dict(state_id=state,constructive_mean_group_l2_percentage_reduction=sum(r['best']['mean_group_l2_percentage_reduction'] for r in selected),
            analytic_upper_bound=sum(r['upper_bound']['mean_group_l2_percentage_reduction_upper_bound'] for r in selected),
            reason='Attention and MLP affected groups and tensors are disjoint; compose their real-arithmetic identities.'))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    summary=dict(rows=rows,combined=combined,source_sha256={str(f):sha(f) for f in [Path(__file__),Path('rewriter/gauge_energy_diagnostic.py')]},
        energy_artifact_sha256=sha(args.energies),elapsed_seconds=time.perf_counter()-started,max_sweeps=args.max_sweeps,
        extra_gpt_forwards=0,extra_gpt_backwards=0,output_or_audit_access=False,
        interpretation='Constructive discrete reward lower bounds and rigorous analytic upper bounds, not a global optimum. Function identity only in real arithmetic; finite logits/ranks unmeasured. Diagnostic outside the fixed anchored action family, not learned Chisel.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':main()
