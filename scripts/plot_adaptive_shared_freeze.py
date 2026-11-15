"""Closed/fixed TRAIN measurements; same-parent private learning-rate contrast."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
parent=json.loads(Path(a.parent).read_text())['arms']['warm1e-4']['readouts'][-1]
data=json.loads(Path(a.input).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
fig,axes=plt.subplots(2,2,figsize=(10,7));colors={'frozen':'#197b9c','joint':'#cb7226'}
for col,domain in enumerate(('gpt','cifar10')):
 axes[0,col].set_title('GPT · states 6 and 499' if domain=='gpt' else 'CIFAR · seeds 0 and 40')
 for arm,values in data['arms'].items():
  rows=[parent]+values['readouts'];xs=[x['update'] for x in rows]
  ys=[];zs=[]
  for r in rows:
   cases=[x for x in r['rows'] if x['domain']==domain];assert len(cases)==2
   ys.append(max(100*x['strict_topk_changes']/x['token_count'] for x in cases));zs.append(min(x['mean_group_l2_percentage_reduction'] for x in cases))
  label={'frozen':'Shared Chisel frozen','joint':'Joint fitting'}[arm]
  axes[0,col].plot(xs,ys,'-o',color=colors[arm],label=label,ms=4)
  axes[1,col].plot(xs,zs,'-o',color=colors[arm],label=label,ms=4)
 for row in range(2):
  ax=axes[row,col];ax.axhline(0,color='#555',ls='--',lw=.7);ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False);ax.set_xlabel('Data batches');ax.set_xticks([64,72,80,88,96])
 axes[0,col].set_yscale('symlog',linthresh=.01);axes[0,col].set_ylim(bottom=-.002);axes[0,col].set_ylabel('Worst strict-change fraction (%)')
 axes[1,col].set_yscale('symlog',linthresh=1e-6);axes[1,col].set_ylabel('Smallest group-mean reduction (%)')
axes[0,0].legend(frameon=False)
fig.suptitle('Frozen shared Chisel versus joint fitting')
fig.tight_layout(rect=(0,.11,1,.96))
fig.text(.5,.025,'Same batch-64 state, private LR 1e-4 and panels. Shared Adam stays at 64 steps when frozen; joint fitting reaches 96.\nFixed 128 GPT TRAIN sequences / 1,024 CIFAR TRAIN images. Zero strict changes and positive group means are required for every state.\nFunctional-group averages are not whole-weight L2 reduction. No query evaluation.',ha='center',fontsize=9)
for ext in ('png','svg'):fig.savefig(out/('fitting.'+ext),dpi=180,bbox_inches='tight')
(out/'data.json').write_text(json.dumps(dict(parent=parent,continuations=data),indent=2))
