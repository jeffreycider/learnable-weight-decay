"""Compare measured losses at completed updates; partial runs are labelled."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('runs/main-objective-local')
RUNS=[
 ('random-full-four-update-v1','Joint clipping; encoder/probe LR 1e-8'),
 ('random-roleclip-four-update-v1','Separate clipping; encoder/probe LR 1e-8'),
 ('random-private-lr1e4-twenty-v1','Separate clipping; encoder/probe LR 1e-4'),
]

def main():
    out=ROOT/'conditioning-comparison'
    out.mkdir(exist_ok=True)
    data=[]
    identity=None
    for name,label in RUNS:
        path=ROOT/name
        metadata=json.loads((path/'metadata.json').read_text())
        current=(metadata['objective'],[(c['sha256'],c['sequence_ids'],c['input_sha256']) for c in metadata['cases']])
        if identity is None:identity=current
        elif identity!=current:raise ValueError('Comparison changes objective, initialization or data')
        records=[json.loads(line) for line in (path/'metrics.jsonl').read_text().splitlines()]
        segments=[name]
        complete=(path/'complete.json').exists()
        if name=='random-roleclip-four-update-v1':
            resumed=ROOT/'random-roleclip-twenty-update-v1'
            if (resumed/'metrics.jsonl').exists():
                resumed_metadata=json.loads((resumed/'metadata.json').read_text())
                from scripts.diagnose_cpu_norm_preservation import resume_contract
                if resume_contract(metadata)!=resume_contract(resumed_metadata):
                    raise ValueError('Resumed control changes training identity')
                records.extend(json.loads(line) for line in (resumed/'metrics.jsonl').read_text().splitlines())
                segments.append(resumed.name)
                complete=(resumed/'complete.json').exists()
        points=[]
        for record in records:
            rows=record['rows']
            points.append(dict(updates=record['completed_updates_before_measure'],
                shrink_percent=sum(r['mean_group_l2_percentage_reduction'] for r in rows)/len(rows),
                rank_changes=sum(r['strict_topk_changes'] for r in rows),
                tokens=sum(r['token_count'] for r in rows),
                fixed_preservation_loss=sum(r['ranking_penalty']+r['tail_quadratic_penalty'] for r in rows)/len(rows),
                mean_kl=sum(r['mean_kl'] for r in rows)/len(rows),
                total_objective=record['mean_objective']))
        unique={}
        for point in points:
            step=point['updates']
            if step in unique and unique[step]!=point:
                raise ValueError('Resumed control endpoint differs from saved measurement')
            unique[step]=point
        points=[unique[step] for step in sorted(unique)]
        data.append(dict(run=name,segments=segments,label=label,complete=complete,points=points))
    fig,axes=plt.subplots(1,3,figsize=(13,4.3),constrained_layout=True)
    colors=['#777777','#2878a0','#ae3d32']
    for run,color in zip(data,colors):
        x=[p['updates'] for p in run['points']]
        label=run['label']+(' (running)' if not run['complete'] else '')
        for ax,key in zip(axes,['shrink_percent','rank_changes','fixed_preservation_loss']):
            ax.plot(x,[p[key] for p in run['points']],marker='o',markersize=3,color=color,label=label)
    titles=['Mean norm reduction','Preservation requires zero changes','Ranking and token-KL penalties']
    labels=['Mean group norm reduction (%)','Token positions with top-100 changes','Mean penalty (lower is better)']
    for ax,title,label in zip(axes,titles,labels):
        ax.set_title(title,fontsize=11)
        ax.set_ylabel(label)
        ax.set_xlabel('Completed Chisel updates')
        ax.grid(alpha=.2)
        ax.axhline(0,color='black',linewidth=.6)
    axes[2].set_yscale('symlog',linthresh=.0001)
    axes[0].legend(fontsize=7,loc='upper left')
    fig.suptitle('Fixed training data: GPT checkpoints 6 and 7, 4,096 token positions',fontsize=12)
    fig.savefig(out/'curves.png',dpi=160)
    (out/'data.json').write_text(json.dumps(dict(
        scope='Local training-fit diagnostic only. Exact teacher ties count as the same rank. Compare methods at shared update counts only; curves may have different current durations.',
        measurements='Measured before the following optimizer step. Norm metric averages equal-weight group percentage reductions over checkpoints.',
        runs=data),indent=2)+'\n')
    print(out/'curves.png')

if __name__=='__main__':main()
