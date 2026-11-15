"""Fixed TRAIN readouts only: worst preservation and least shrinkage by domain."""
import argparse,json,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
source=Path(a.input);data=json.loads(source.read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
fig,axes=plt.subplots(2,2,figsize=(10,7.2));colors={'head1e-8':'#187b9c','head1e-7':'#d47c29','head1e-6':'#8555a1'};values={}
for col,domain in enumerate(('gpt','cifar10')):
 axes[0,col].set_title('GPT · states 6 and 499' if domain=='gpt' else 'CIFAR · seeds 0 and 40')
 values[domain]={}
 for name,arm in data['arms'].items():
  rows=[]
  for readout in arm['readouts']:
   cases=[v for v in readout['rows'] if v['domain']==domain];assert len(cases)==2
   rows.append(dict(update=readout['update'],worst_strict_percent=max(100*v['strict_topk_changes']/v['token_count'] for v in cases),minimum_group_reduction=min(v['mean_group_l2_percentage_reduction'] for v in cases),cases=cases,readout_sha256=readout['sha256']))
  values[domain][name]=rows
  for r,key in enumerate(('worst_strict_percent','minimum_group_reduction')):
   axes[r,col].plot([v['update'] for v in rows],[v[key] for v in rows],'-o',ms=4,lw=2,color=colors[name],label=name.replace('head','Head LR '))
 for r in range(2):
  ax=axes[r,col];ax.axhline(0,color='#555',ls='--',lw=.8);ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False);ax.set_xticks([8,16,24,32]);ax.set_xlabel('Shared updates')
 axes[0,col].set_yscale('symlog',linthresh=.01);axes[0,col].set_ylabel('Worst strict-change fraction (%)');axes[0,col].set_ylim(bottom=-.002)
 axes[1,col].set_yscale('symlog',linthresh=1e-5);axes[1,col].set_ylabel('Smallest group-mean reduction (%)')
axes[0,0].legend(frameon=False,fontsize=9)
fig.suptitle('Adaptive edits on four fitting checkpoints',fontsize=16)
fig.tight_layout(rect=(0,.10,1,.96),h_pad=2)
fig.text(.5,.035,'Same fresh initialization and data; only head learning rate changes. Fixed TRAIN128 GPT / TRAIN1024 CIFAR readouts.\nTop: the worse of two checkpoints must reach zero. Bottom: the smaller functional-group mean reduction must stay positive.\nGroup means are not whole-weight L2 reduction. No held-out queries.',ha='center',fontsize=9)
for ext in ('png','svg'):fig.savefig(out/('fitting.'+ext),dpi=180,bbox_inches='tight')
(out/'data.json').write_text(json.dumps(dict(input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),values=values),indent=2)+'\n')
