"""Describe the predeclared SOAP pilots; never selects or launches a trial."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def trailing(values,width=16):
    values=np.asarray(values,dtype=float)
    return np.array([values[max(0,i-width+1):i+1].mean() for i in range(len(values))])


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('runs/soap-hyper-pilot-v1'));args=p.parse_args()
    arms={};summary={}
    for name in ['fixed','low','high']:
        path=args.root/name/'metrics.jsonl'
        if not path.exists():continue
        rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        train=[r for r in rows if r['event']=='train'];validation=[r for r in rows if r['event']=='validation']
        if not train:continue
        arms[name]=train
        summary[name]={'iterations_observed':len(train),'last_train_loss':train[-1]['train_loss'],
          'last16_mean_train_loss':float(np.mean([r['train_loss'] for r in train[-16:]])),
          'validation':validation,'final_hyperparameter_norm':train[-1]['hyperparameter_norm'],
          'steady_time_medians':{key:float(np.median([r.get(key,0.) for r in train[5:]])) for key in ['gpt_forward_backward_seconds','base_optimizer_seconds','credit_seconds','wall_seconds']},
          'final_residual_to_base_factor_mean':train[-1].get('residual_to_base_factor_mean'),
          'gradient_clip_fraction':float(np.mean([r.get('hypergradient_clip_factor',1.)<1 for r in train])),
          'all_gradients_finite':bool(all(np.isfinite(r.get('hypergradient_norm',0.)) for r in train))}
    if not arms:raise SystemExit('no metrics yet')
    colors={'fixed':'#444444','low':'#2171b5','high':'#d95f0e'}
    labels={'fixed':'Fixed Q = 0','low':'Learned H, LR 1e-5','high':'Learned H, LR 1e-4'}
    fig,ax=plt.subplots(2,1,figsize=(10,8),sharex=True)
    for name,rows in arms.items():
        steps=[r['step'] for r in rows];loss=[r['train_loss'] for r in rows]
        ax[0].plot(steps,loss,color=colors[name],alpha=.22)
        ax[0].plot(steps,trailing(loss),color=colors[name],label=labels[name])
        if name!='fixed' and 'fixed' in arms:
            base={r['step']:r['train_loss'] for r in arms['fixed']};paired=[r for r in rows if r['step'] in base]
            delta=[r['train_loss']-base[r['step']] for r in paired];x=[r['step'] for r in paired]
            ax[1].plot(x,delta,color=colors[name],alpha=.22);ax[1].plot(x,trailing(delta),color=colors[name],label=labels[name])
            summary[name]['last16_paired_mean_difference']=float(np.mean(delta[-16:]))
    ax[0].set_ylabel('Training loss (nats/token)');ax[0].legend();ax[0].set_title('Matched seed-0 SOAP residual pilots · faint raw, solid trailing 16-step mean')
    ax[1].axhline(0,color='black',lw=.8);ax[1].set_ylabel('Training loss − fixed Q=0');ax[1].set_xlabel('Ordinary GPT iteration');ax[1].legend()
    for a in ax:a.grid(alpha=.2)
    fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(args.root/('loss_comparison.'+ext),dpi=150)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4),sharey=True)
    for a,(name,rows) in zip(axes,arms.items()):
        steps=[r['step'] for r in rows]
        for key,label,color in [('gpt_forward_backward_seconds','GPT forward/backward','#3182bd'),('base_optimizer_seconds','Base optimizer','#31a354'),('credit_seconds','Delayed VJP + Adam','#e6550d')]:
            a.plot(steps,[r.get(key,0.) for r in rows],label=label,color=color)
        a.set_title(labels[name]);a.set_xlabel('GPT iteration');a.grid(alpha=.2);a.set_ylim(0,max(1.,np.percentile([r.get('wall_seconds',0.) for r in rows[5:]],95)*1.1))
    axes[0].set_ylabel('Seconds per iteration (cold compile excluded by axis)');axes[0].legend(fontsize=8);fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(args.root/('component_times.'+ext),dpi=150)
    (args.root/'summary.json').write_text(json.dumps({'interpretation':'One-seed early diagnostic; not a full-run or benchmark improvement claim. Smoothing is descriptive only.','arms':summary},indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
