"""Plot measured fitting counts/norm reward; never compare unlike loss totals."""
import argparse,json,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--lower-lr-input');args=p.parse_args()
 raw=Path(args.input).read_bytes();data=json.loads(raw);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 fig,axes=plt.subplots(2,2,figsize=(9,6.8),sharex=True)
 styles={'mean-control':('Mean ranking penalty','#167d9a'),'micro-max':('Worst violation per microbatch','#cb6329')}
 input_hashes={args.input:hashlib.sha256(raw).hexdigest()}
 if args.lower_lr_input:
  extra=Path(args.lower_lr_input).read_bytes();data['arms'].update(json.loads(extra)['arms']);input_hashes[args.lower_lr_input]=hashlib.sha256(extra).hexdigest()
  styles={'mean-control':('Mean penalty, original LR','#167d9a'),'micro-max':('Max penalty, original LR','#a3a3a3'),'micro-max-lr0.1':('Max penalty, LR ÷10','#cb6329'),'micro-max-lr0.01':('Max penalty, LR ÷100','#7658a6')}
 extracted={};last=[]
 for name,(label,color) in styles.items():
  rows=data['arms'][name]['readouts'];extracted[name]=rows;last.append(rows[-1]['update'])
  for column,uid in enumerate((6,499)):
   points=[(r['update']-1792,next(x for x in r['rows'] if x['domain']=='gpt' and x['local_id']==uid)) for r in rows]
   x=[a for a,b in points];axes[0,column].plot(x,[b['strict_topk_changes'] for a,b in points],'-o',label=label,color=color,lw=2)
   axes[1,column].plot(x,[b['mean_group_l2_percentage_reduction'] for a,b in points],'-o',color=color,lw=2)
 for column,uid in enumerate((6,499)):
  axes[0,column].set_title(f'GPT checkpoint {uid}',fontsize=13)
  axes[0,column].set_yscale('symlog',linthresh=1);axes[0,column].axhline(0,color='#333',lw=.7);axes[0,column].set_ylim(bottom=-.05)
  axes[1,column].set_xlabel('Updates after the common checkpoint')
  for row in (0,1):axes[row,column].grid(alpha=.2);axes[row,column].spines[['top','right']].set_visible(False)
 axes[0,0].set_ylabel('Strict top-100 changes\n(out of 65,536 fitting contexts)')
 axes[1,0].set_ylabel('Mean reduction across 15 groups (%)')
 fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=2 if args.lower_lr_input else 1,frameon=False,bbox_to_anchor=(.5,.99))
 fig.suptitle('Mean and worst-violation ranking penalties',y=1.045,fontsize=15)
 fig.tight_layout(rect=(0,.04,1,.91));fig.text(.5,.005,('Same parent1792 and TRAIN64 panels; lower-LR trials followed the original max failure.\nNo matched lower-LR mean controls. Group mean is not whole-model norm reduction.' if args.lower_lr_input else 'Same TRAIN64 panels, parent1792, optimizer state and LRs. No query evaluation.\nGroup mean is not whole-model norm reduction; different loss totals are not compared.'),ha='center',fontsize=9)
 for ext in ('png','svg'):fig.savefig(out/f'fitting.{ext}',dpi=170,bbox_inches='tight')
 (out/'data.json').write_text(json.dumps(dict(input_sha256=hashlib.sha256(raw).hexdigest(),input_hashes=input_hashes,through_updates=dict(zip(styles,last)),rows=extracted,scope='fixed fitting panels only'),indent=2)+'\n')
if __name__=='__main__':main()
