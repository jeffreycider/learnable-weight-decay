"""Weight-only rank-limited binary-gauge control; no GPT execution."""
import argparse,hashlib,json,time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from rewriter.coordinate_anchor_gauge import capped_binary_greedy


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--energies',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--rank',type=int,default=8)
    p.add_argument('--require-matrix-improvement',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);raw=torch.load(a.energies,map_location='cpu',weights_only=False);started=time.perf_counter();rows=[]
    for sid,record in raw['states'].items():
        for family,power in [('attention',1),('mlp',2)]:
            b=record['families'][family]
            result=capped_binary_greedy(*[v.numpy() for v in b['energies']],[c['first_weight'] for c in b['channels']],inverse_power=power,rank=a.rank,
                                        require_matrix_improvement=a.require_matrix_improvement)
            result.update(state_id=sid,state_sha256=record['state_sha256'],family=family,
                          selected=[dict(c,exponent=e) for c,e in zip(b['channels'],result['channel_exponents']) if e])
            rows.append(result)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    summary=dict(rows=rows,combined={str(sid):sum(r['mean_group_l2_percentage_reduction'] for r in rows if r['state_id']==sid) for sid in raw['states']},
        energy_sha256=sha(a.energies),source_sha256={str(f):sha(f) for f in [Path(__file__),Path('rewriter/coordinate_anchor_gauge.py'),Path('rewriter/gauge_energy_diagnostic.py')]},
        elapsed_seconds=time.perf_counter()-started,gpt_forwards=0,gpt_backwards=0,output_access=False,
        require_matrix_improvement=a.require_matrix_improvement,
        interpretation='Constructive rank-limited gauge target; target-specific learned write spans required; not globally optimal; finite predictions unmeasured. No training objective or policy changed.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:f.write(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary['combined']))

if __name__=='__main__':main()
