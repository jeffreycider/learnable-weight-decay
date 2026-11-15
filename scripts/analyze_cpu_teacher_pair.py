"""Read-only paired teacher-fit evidence and curves; no preservation inference."""
import argparse
import json
from pathlib import Path
import torch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rewriter.grouped_norm import functional_groups


def history(run):
    meta=json.loads((run/'metadata.json').read_text())
    metrics=[json.loads(s) for s in (run/'metrics.jsonl').read_text().splitlines()]
    if 'resumed_from' in meta:
        previous=torch.load(meta['resumed_from']['path'],map_location='cpu',weights_only=False,mmap=True)
        older,initial=history(Path(previous['metadata']['args']['output']))
        return [r for r in older if r['step']<metrics[0]['step']]+metrics,initial
    return metrics,run/'checkpoint_0000.pt'


def branch_start(run):
    meta=json.loads((run/'metadata.json').read_text())
    fork=meta.get('objective_fork',meta.get('rate_fork'))
    if fork is not None:return fork['parent']['step']
    if 'resumed_from' in meta:
        parent=torch.load(meta['resumed_from']['path'],map_location='cpu',weights_only=False,mmap=True)
        return branch_start(Path(parent['metadata']['args']['output']))
    return 0


def main():
    p=argparse.ArgumentParser();p.add_argument('runs',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    rows={};initial=None
    for run in a.runs:
        meta=json.loads((run/'metadata.json').read_text())
        metrics,initial_path=history(run)
        ck=torch.load(initial_path,map_location='cpu',weights_only=False)
        if initial is None:initial=ck
        equal=all(torch.equal(v,initial[k][n]) if torch.is_tensor(v) else v==initial[k][n] for k in ('actor','private') for n,v in ck[k].items())
        end=metrics[-1]
        # Reuse the physical per-parameter projection evidence; only reweight it.
        caps=end['target_span_capture']['captured_squared_by_parameter']
        state=torch.load(meta['args']['state'],map_location='cpu',weights_only=False,mmap=True)
        groups=functional_groups(state['model'])
        relative_capture=sum(sum(caps.get(n,0.) for n in names)/meta['group_reference_squared'][g]
            for g,names in groups.items() if meta['group_reference_squared'][g]>0)/meta['relative_target_squared']
        rows[run.name]=dict(path=str(run),initial_actor_private_exactly_equal=equal,args=meta['args'],
            complete=json.loads((run/'complete.json').read_text()),
            relative_span_capture_fraction=relative_capture,relative_span_error_floor=1-relative_capture,
            endpoint={k:end[k] for k in ('normalized_edit_error','relative_normalized_edit_error','matrix_error_squared','bias_error_squared','selected_error_squared','zero_target_error_squared','group_error_squared','group_relative_error_squared','target_span_capture')})
    (a.output/'comparison.json').write_text(json.dumps(dict(runs=rows,learned_preservation_claim=False,scope='Single-state supervised teacher ability; nominal edits, no GPT calls'),indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    unique={}
    for run in a.runs:
        m,_=history(run);args=dict(rows[run.name]['args']);args['_plot_start']=branch_start(run)
        signature=tuple(args[k] for k in ('objective','head_lr','representation_lr','private_lr','seed','private_seed','_plot_start'))
        if signature not in unique or m[-1]['step']>unique[signature][0][-1]['step']:
            m=[r for r in m if r['step']>=args['_plot_start']]
            unique[signature]=(m,args)
    for index,(m,args) in enumerate(unique.values()):
        label=('Total-error objective' if args['objective']=='global' else 'Group-normalized objective')+f", head {args['head_lr']:.0e}, private {args['private_lr']:.0e}"
        if args['_plot_start']:label+=f", fork at {args['_plot_start']}"
        for ax,key in zip(axes,('normalized_edit_error','relative_normalized_edit_error')):
            ax.plot([r['step'] for r in m],[r[key] for r in m],label=label,linestyle=('-', '--', '-.', ':')[index%4])
    for ax,title in zip(axes,('Total squared edit error','Group-normalized squared edit error')):
        ax.axhline(1,color='gray',linewidth=.8,linestyle=':',label='No edit (identity)')
        ax.axhline(0,color='black',linewidth=.6)
        ax.set(xlabel='Completed updates',ylabel='Error / identity error (lower is better)',title=title,ylim=(0,None))
        ax.grid(alpha=.2);ax.ticklabel_format(useOffset=False)
    if max(r['normalized_edit_error'] for m,_ in unique.values() for r in m)>20:
        axes[0].set_yscale('symlog',linthresh=1)
        axes[0].set_title('Total squared edit error (linear 0–1; log above 1)')
    axes[0].legend(fontsize=6.5,loc='upper left')
    fig.suptitle('Single-state teacher fitting; zero is exact target fit',fontsize=11)
    fig.tight_layout();fig.savefig(a.output/'curves.png',dpi=160);fig.savefig(a.output/'curves.svg');plt.close(fig)



if __name__=='__main__':main()
