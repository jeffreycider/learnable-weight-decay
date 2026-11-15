"""Seen-only fixed-panel continuation curves; never ranks different Lagrangians."""
import argparse,json,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(source,output,through):
 source=Path(source);output=Path(output);output.mkdir(parents=True,exist_ok=True);data=json.loads(source.read_text())
 names=['absolute-original','absolute-separate','relative-0.0001','relative-0.001','relative-0.01','relative-0.1']
 labels=['Absolute loss, original optimizer','Absolute loss, stronger body updates','Relative margin × 0.0001','Relative margin × 0.001','Relative margin × 0.01','Relative margin × 0.1']
 if 'gamma0.1-lr1' in data:
  names=['gamma0.1-lr1','gamma1-lr1','gamma10-lr1','gamma100-lr1','gamma10-lr0.1','gamma100-lr0.1']
  labels=['Rank weight 0.1','Rank weight 1','Rank weight 10','Rank weight 100','Rank weight 10, learning rates ÷10','Rank weight 100, learning rates ÷10']
 elif set(data)=={'gamma100-lr1','gamma100-lr0.1'}:
  names=['gamma100-lr1','gamma100-lr0.1']
  labels=['Continue current learning rates','All primal learning rates ÷10']
 boundary=min(a['update'] for n in names for a in data[n]['audits'])
 colors=['#777777','#222222','#5099c5','#4f9a69','#d29436','#9a5a9c'];rows={}
 plt.rcParams.update({'font.size':12,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(2,2,figsize=(10,8),layout='constrained')
 for i,name in enumerate(names):
  gamma=float(name.removeprefix('relative-')) if name.startswith('relative-') else float(name.removeprefix('gamma').split('-lr')[0]) if name.startswith('gamma') else None
  audits=[a for a in data[name]['audits'] if a['update']<=through];points=[]
  for a in audits:
   rr=[r for r in a['rows'] if r['domain']=='gpt']
   points.append(dict(update=a['update'],strict=sum(r['strict_topk_changes'] for r in rr),contexts=sum(r['token_count'] for r in rr),mean_kl=sum(r['mean_kl'] for r in rr)/len(rr),mean_group_reduction=sum(r['mean_group_l2_percentage_reduction'] for r in rr)/len(rr),min_checkpoint_group_reduction=min(r['mean_group_l2_percentage_reduction'] for r in rr),weighted_ranking_penalty=sum(r['ranking_penalty'] for r in rr)/len(rr),unweighted_relative_ranking=(sum(r['ranking_penalty'] for r in rr)/len(rr)/gamma) if gamma is not None else None,state_rows=rr))
  rows[name]=points;x=[p['update'] for p in points];style='--' if (name.startswith('absolute') or name.endswith('-lr0.1')) else '-'
  for ax,key in [(axes[0,0],'strict'),(axes[0,1],'mean_group_reduction'),(axes[1,0],'mean_kl')]:ax.plot(x,[p[key] for p in points],style,color=colors[i],marker='o',ms=4,label=labels[i])
  if gamma is not None:axes[1,1].plot(x,[p['unweighted_relative_ranking'] for p in points],style,color=colors[i],marker='o',ms=4,label=labels[i])
 axes[0,0].set_ylim(bottom=0);axes[0,0].axhline(0,color='#555',ls=':',lw=1);axes[0,0].set_ylabel('Strict top-100 changes / 32,768 tokens');axes[0,0].set_title('Prediction ranking on fitting data')
 axes[0,1].set_ylabel('Mean group L2 reduction (%)');axes[0,1].set_title('Weight reduction on the same states')
 axes[1,0].set_ylabel('Mean KL (nats/token)');axes[1,0].set_yscale('log');axes[1,0].axhline(1e-10,color='#555',ls=':',lw=1,label='KL target')
 axes[1,1].set_ylabel('Mean relative-margin loss (unweighted)');axes[1,1].set_yscale('log');axes[1,1].set_title('Comparable ranking violation loss')
 for ax in axes.flat:ax.set_xlabel('Global optimizer update');ax.grid(alpha=.2);ax.axvline(boundary,color='#888',ls=':',lw=1)
 handles,legend=axes[0,0].get_legend_handles_labels();fig.legend(handles,legend,loc='outside lower center',ncol=2,fontsize=10)
 fig.suptitle('Ranking preservation and weight reduction on fitting data',fontsize=16)
 fig.savefig(output/'fitting.svg');fig.savefig(output/'fitting.png',dpi=150);plt.close(fig)
 (output/'data.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),through_update=through,start_update=boundary,scope='Each stage begins from one pinned actor/private optimizer checkpoint. Two GPT fitting states 6/499, 16 sequences each; no query. Relative ranking curves divide each logged penalty by its pinned gamma, giving the same unweighted objective across relative-loss arms.',series=rows),indent=2))


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--through-update',type=int,required=True);a=p.parse_args();plot(a.input,a.output,a.through_update)
