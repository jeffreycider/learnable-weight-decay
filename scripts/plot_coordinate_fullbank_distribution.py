"""Complete fixed-fitting distributions; no selection or model execution."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import SymmetricalLogLocator
from matplotlib.transforms import blended_transform_factory

CASES={'imported211':('Imported actor · update 211','#237a9a'),'fresh211':('Fresh actor · update 211','#d47729'),'imported422':('Imported actor · update 422','#7654a4')}
METRICS=[('strict_context_percent','Contexts with strict ranking changes (%)',.001),('mean_group_l2_percentage_reduction','Mean functional-group reduction (%)',.000001),('whole_editable_l2_percentage_reduction','Whole editable-weight L2 reduction (%)',.000001)]

def main():
 p=argparse.ArgumentParser();p.add_argument('--snapshot',action='append',required=True,help='imported211|fresh211|imported422=immutable .pt path');p.add_argument('--output',required=True);a=p.parse_args()
 snapshots={};pins={};reference=None
 for item in a.snapshot:
  key,path=item.split('=',1);assert key in CASES and key not in snapshots
  file=Path(path);raw=file.read_bytes();rows=torch.load(file,map_location='cpu',weights_only=False)['rows']
  identities={(r['domain'],r['local_id']):r for r in rows};assert len(identities)==len(rows)
  if reference is None:reference=identities
  else:
   assert identities.keys()==reference.keys(),'full state coverage differs'
   for uid,r in identities.items():assert r['context_ids']==reference[uid]['context_ids'],f'fixed panel differs: {uid}'
  snapshots[key]=rows;pins[key]=dict(path=str(file),sha256=hashlib.sha256(raw).hexdigest())
 order=[k for k in CASES if k in snapshots];out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 fig,axes=plt.subplots(3,2,figsize=(11,9.2));summary={};values={};rng=np.random.default_rng(0)
 for col,domain in enumerate(('gpt','cifar10')):
  n=sum(r['domain']==domain for r in snapshots[order[0]]);expected=800 if domain=='gpt' else 80;assert n==expected
  axes[0,col].set_title(f'{"GPT" if domain=="gpt" else "CIFAR"} · all {n} fitting states',fontsize=14,pad=12)
  summary[domain]={};values[domain]={}
  for arm,key in enumerate(order):
   rows=sorted((r for r in snapshots[key] if r['domain']==domain),key=lambda r:r['local_id'])
   values[domain][key]=[dict(local_id=r['local_id'],strict_context_percent=100*r['strict_topk_changes']/r['token_count'],mean_group_l2_percentage_reduction=r['mean_group_l2_percentage_reduction'],whole_editable_l2_percentage_reduction=r['whole_editable_l2_percentage_reduction']) for r in rows]
   summary[domain][key]=dict(states=n,strict_zero=sum(r['strict_topk_changes']==0 for r in rows),positive_group_mean=sum(r['mean_group_l2_percentage_reduction']>0 for r in rows),both=sum(r['strict_topk_changes']==0 and r['mean_group_l2_percentage_reduction']>0 for r in rows),whole_l2_reduced=sum(r['whole_editable_l2_percentage_reduction']>0 for r in rows))
   y=arm+rng.uniform(-.12,.12,n);color=CASES[key][1]
   for row,(metric,label,threshold) in enumerate(METRICS):
    v=np.array([r[metric] for r in values[domain][key]]);ax=axes[row,col]
    count=int(np.sum(v==0)) if row==0 else int(np.sum(v>0));word='zero' if row==0 else 'reduced'
    ax.text(.98,arm-.22,f'{count}/{n} {word}',ha='right',va='bottom',fontsize=8,color=color,transform=blended_transform_factory(ax.transAxes,ax.transData))
    ax.scatter(v,y,s=7 if n>100 else 13,color=color,alpha=.25,zorder=2,rasterized=True)
    q=np.quantile(v,[.05,.5,.95]);ax.plot([q[0],q[2]],[arm,arm],color=color,lw=3,zorder=3);ax.plot(q[1],arm,'|',color='#111',ms=15,mew=2,zorder=4)
    ax.set_xscale('symlog',linthresh=threshold);locator=SymmetricalLogLocator(base=10,linthresh=threshold);locator.set_params(numticks=5);ax.xaxis.set_major_locator(locator);ax.axvline(0,color='#333',ls='--',lw=.8,zorder=0);ax.set_xlabel(label,fontsize=11)
    ax.set_yticks(range(len(order)),[CASES[k][0].replace(' actor','') for k in order],fontsize=9);ax.set_ylim(len(order)-.6,-.6);ax.grid(axis='x',alpha=.17);ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
 fig.suptitle('Prediction preservation and weight reduction across every fitting state',fontsize=16,y=.995)
 fig.tight_layout(rect=(0,.085,1,.97),h_pad=2.5,w_pad=2)
 comparison=('Fresh 211 versus imported 211 compares initialization; imported 422 includes additional training.' if 'fresh211' in snapshots and 'imported422' in snapshots else 'Fresh 211 versus imported 211 compares initialization.' if 'fresh211' in snapshots else 'Imported initialization, first complete fitting-state readout; no later result shown.')
 fig.text(.5,.036,'Each point is one state; thick line: central 90%; black mark: median. Dashed line: zero.\nRanking changes should reach zero; reduction should stay positive. Group means and whole L2 are distinct.\n'+comparison+' TRAIN panels only.',ha='center',fontsize=10)
 for ext in ('png','svg'):fig.savefig(out/f'distributions.{ext}',dpi=180,bbox_inches='tight')
 (out/'data.json').write_text(json.dumps(dict(inputs=pins,summary=summary,values=values,scope='Complete fixed TRAIN panels; every fitting state included; no query evaluation or selected best checkpoint'),indent=2)+'\n')
 print(json.dumps(summary))
if __name__=='__main__':main()
