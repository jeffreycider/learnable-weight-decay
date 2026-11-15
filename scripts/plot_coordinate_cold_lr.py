"""Fitting-only private LR controls; earlier LR1e-5 reference is explicit."""
import argparse,json,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot(old_path,new_path,output):
 old_path,new_path=Path(old_path),Path(new_path);old=json.loads(old_path.read_text());new=json.loads(new_path.read_text());output=Path(output);output.mkdir(parents=True,exist_ok=True)
 cases=['gpt-251-random','cifar10-21-random','gpt-501-warm499'];titles=['GPT 251 · new probes','CIFAR seed 20 · new probes','GPT 501 · warm start from 499']
 colors=['#777777','#167a9d','#b76924'];rates=['1e-5','1e-4','1e-3'];series={}
 plt.rcParams.update({'font.size':11,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
 for j,(case,title) in enumerate(zip(cases,titles)):
  runs=[old['cases'][case]['readouts']]+[[r for r in new[a]['readouts'] if r['event']=='fixed_support_readout' and r['case']==case] for a in ('cold-private-lr0.0001','cold-private-lr0.001')]
  series[case]={}
  for i,rows in enumerate(runs):
   points=[dict(update=r['private_update'],strict=r['row']['strict_topk_changes'],norm=r['row']['mean_group_l2_percentage_reduction'],mean_kl=r['row']['mean_kl']) for r in rows];series[case][rates[i]]=points
   for ax,key in ((axes[0,j],'strict'),(axes[1,j],'norm')):ax.plot([p['update'] for p in points],[p[key] for p in points],marker='o',color=colors[i],label='Private LR '+rates[i],ls='--' if i==0 else '-')
  axes[0,j].set_title(title,fontsize=12);axes[0,j].set_yscale('symlog',linthresh=1);axes[0,j].set_ylim(bottom=0);axes[0,j].set_ylabel('Strict ranking changes');axes[0,j].axhline(0,color='#555',lw=.8)
  axes[1,j].set_ylabel('Mean group L2 reduction (%)')
  for ax in axes[:,j]:ax.grid(alpha=.2);ax.set_xlabel('Private fitting updates')
 handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3)
 fig.suptitle('Faster cold fitting does not imply a better warm-start learning rate',fontsize=17)
 fig.savefig(output/'fitting.svg');fig.savefig(output/'fitting.png',dpi=150);plt.close(fig)
 (output/'data.json').write_text(json.dumps(dict(source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (old_path,new_path)},scope='Frozen shared actor1024. Fresh Adam/dual each case; identical initialization and fitting panels. GPT16 sequences, CIFAR1024 images. Earlier1e-5 controls retained. No query or shared updates.',series=series),indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--old',required=True);p.add_argument('--new',required=True);p.add_argument('--output',required=True);a=p.parse_args();plot(a.old,a.new,a.output)
