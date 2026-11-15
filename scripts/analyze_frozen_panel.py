"""Summarize all predeclared frozen trials; never select a winning checkpoint."""
import argparse
import json
from pathlib import Path


def summarize(directory):
    path=Path(directory)
    rows=[json.loads(line) for line in (path/'metrics.jsonl').read_text().splitlines() if line]
    config=json.loads((path/'config.json').read_text())
    training=[r for r in rows if r['event']=='train']
    learned=[r for r in rows if r['event']=='evaluation' and r['label']=='learned']
    finished=[r for r in rows if r['event']=='finished']
    final=[r for r in learned if r['step']==config['steps']]
    return dict(path=str(path),config=config,training=training,evaluations=learned,
                controls=[r for r in rows if r['event']=='evaluation' and 'gauge' in r['label']],
                complete=bool(finished),endpoint=final[-1] if final else None,
                latest_training=training[-1] if training else None,
                latest_audit=learned[-1] if learned else None,
                finished=finished[-1] if finished else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories',nargs='+',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    panel=[summarize(p) for p in args.directories]
    definitions={(a['config'].get('norm_objective','global_squared'),
                  a['config'].get('editable_scope','all')) for a in panel}
    if len(definitions)!=1:
        raise ValueError('plot one objective and editable scope at a time; denominators differ')
    objective,scope=next(iter(definitions))
    grouped=objective=='grouped_l2'
    metric_name='mean_group_l2_percentage_reduction' if grouped else 'squared_norm_reduction_percent'
    slim=[{k:v for k,v in arm.items() if k not in ('training','evaluations')} for arm in panel]
    (args.output/'summary.json').write_text(json.dumps(slim,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for i,arm in enumerate(panel):
        c=arm['config'];label=f"LR={c['lr']:g}, KL band={c['kl_budget']:g}"
        color=['#0072B2','#222222','#B36900'][i%3]
        train=arm['training'];audit=arm['evaluations']
        if train:
            axes[0].plot([r['step'] for r in train],[-100*r['norm_relative_change'] for r in train],label=label,color=color)
            axes[1].plot([r['step'] for r in train],[r['objective'] for r in train],label=label,color=color,alpha=.75)
        if audit:
            axes[2].plot([r['step'] for r in audit],[r['distributions']['historical_audit']['mean_kl'] for r in audit],label=label+' / historical',color=color,marker='o')
            axes[2].plot([r['step'] for r in audit],[r['distributions']['repeated']['mean_kl'] for r in audit],label=label+' / repeated',color=color,ls='--',marker='x')
    axes[0].set_ylabel('Mean functional-group L2-norm reduction (%)' if grouped else
                       'Editable squared-weight-norm reduction (%)')
    axes[1].set_ylabel('Current training-minibatch objective (symlog)')
    axes[1].set_yscale('symlog',linthresh=1)
    axes[1].axhline(0,color='gray',lw=.8,label='identity')
    axes[2].set_ylabel('Withheld mean KL (nats/token)');axes[2].set_yscale('log')
    for ax in axes:
        ax.set_xlabel('Controller optimization updates');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);axes[2].legend(fontsize=7)
    fig.suptitle(f'Frozen checkpoint: {objective}, {scope} scope; extra GPT compute')
    fig.savefig(args.output/'progress.png',dpi=160);fig.savefig(args.output/'progress.svg');plt.close(fig)
    for arm in slim:
        r=arm['endpoint'] or arm['latest_audit']
        print(json.dumps(dict(path=arm['path'],complete=arm['complete'],audit_step=r['step'] if r else None,
            **{metric_name:-100*r['norm_relative_change'] if r else None},
            historical_kl=r['distributions']['historical_audit']['mean_kl'] if r else None)))


if __name__=='__main__':main()
