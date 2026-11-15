"""Fixed fitting-panel curves; common metrics rather than varying Lagrangians."""
import argparse,json,pathlib,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--input',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--through',type=int,default=64);a=p.parse_args();raw=a.input.read_bytes();data=json.loads(raw);a.output.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':11,'axes.titlesize':14,'axes.labelsize':12,'xtick.labelsize':10,'ytick.labelsize':10,'svg.fonttype':'none'})
fig,axes=plt.subplots(4,1,figsize=(7,13),sharex=True);series=[]
colors=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#111111','#e377c2']
order=['unclipped_scaled-eta0.0001-rank0','unclipped_scaled-eta0.0001-rank1','unclipped_scaled-eta0.01-rank0','unclipped_scaled-eta0.01-rank1','clipped_relative-eta0.01-rank0','clipped_relative-eta0.01-rank1','clipped_relative-eta0.01-rank1-lr1e-7','clipped_relative-eta0.01-rank1-lr1e-8']
labels=['Unclipped η=0.0001 · rank off','Unclipped η=0.0001 · rank on','Unclipped η=0.01 · rank off','Unclipped η=0.01 · rank on','Clipped η=0.01 · rank off','Clipped η=0.01 · rank on','Clipped + rank · LR 10⁻⁷','Clipped + rank · LR 10⁻⁸']
display_labels={order[0]:'Shrinkage-focused, no ranking penalty',order[1]:'Shrinkage-focused + ranking penalty',order[5]:'Lowest worst-checkpoint KL'}
highlight_colors={order[0]:'#2474a6',order[1]:'#dd7b16',order[5]:'#3b8155'}
for i,(name,label) in enumerate(zip(order,labels)):
 arm=next((x for x in data['arms'] if x['arm']==name),None)
 if arm is None:continue
 by_update={r['update']:r for r in arm['rows'] if r.get('event')=='audit_summary' and r.get('role')=='fixed_training' and r['update']<=a.through};rows=[by_update[u] for u in sorted(by_update)];x=[r['update'] for r in rows]
 vals=[[r['mean_kl'] for r in rows],[r['maximum_kl'] for r in rows],[100*r['top10_order_changes']/r['token_count'] for r in rows],[r['mean_group_l2_percentage_reduction'] for r in rows]]
 for ax,y in zip(axes,vals):ax.plot(x,y,color=highlight_colors.get(name,'#999999'),label=display_labels.get(name,'Other five settings' if i==2 else '_nolegend_'),marker='s' if i>=6 else 'o',linestyle='--' if i>=6 else '-',lw=2 if name in display_labels else 1,ms=4,alpha=1 if name in display_labels else .35,zorder=3 if name in display_labels else 1)
 series.append({'arm':name,'label':label,'updates':x,'mean_kl':vals[0],'worst_checkpoint_mean_kl':vals[1],'top10_order_change_percent':vals[2],'mean15group_shrink_percent':vals[3],'metrics_sha256':arm['metrics_sha256'],'matrix_group_mean_percent':[sum(v for k,v in r['group_l2_percentage_reduction'].items() if k.endswith('weight'))/8 for r in rows],'bias_group_mean_percent':[sum(v for k,v in r['group_l2_percentage_reduction'].items() if k.endswith('bias'))/7 for r in rows]})
for ax in axes:ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
for ax in axes[:2]:ax.set_yscale('log');ax.axhline(1e-10,color='#777',ls=':',lw=1.5)
axes[0].set_ylabel('Mean output KL');axes[0].set_title('Prediction preservation and weight reduction\non the same fitting panels',loc='left')
axes[1].set_ylabel('Worst checkpoint-mean KL');axes[2].set_ylabel('Top-10 order changed (%)');axes[3].set_ylabel('Mean group norm reduction (%)');axes[3].axhline(0,color='#777',lw=.7);axes[3].set_xlabel('Chisel optimizer updates');axes[3].set_xticks([x for x in [0,8,32,64,128] if x<=a.through]);axes[3].set_xlim(0,a.through)
handles,ls=axes[0].get_legend_handles_labels();fig.legend(handles,ls,loc='lower center',bbox_to_anchor=(.5,.015),ncol=1,fontsize=11,frameon=False);fig.subplots_adjust(left=.18,right=.96,top=.95,bottom=.18,hspace=.18)
fig.savefig(a.output/'calibration.svg');fig.savefig(a.output/'calibration.png',dpi=160);plt.close(fig)
(a.output/'data.json').write_text(json.dumps({'source_sha256':hashlib.sha256(raw).hexdigest(),'through_update':a.through,'fixed_panel_scope':'Four fitting GPT checkpoints6/251/499/999;16cachedtrain sequences each,65536teacher-forcedtokenpositions. No heldout/generalization/online-validation claim.','original_factorial_arms':6,'additional_lr_diagnostics':2,'target':'checkpoint-panel mean KL1e-10, not per-token bound or ranking guarantee','series':series},indent=2)+'\n')

fig,axs=plt.subplots(2,1,figsize=(7,8),sharex=True)
for i,row in enumerate(series):
 for ax,key in zip(axs,['matrix_group_mean_percent','bias_group_mean_percent']):
  ax.plot(row['updates'],row[key],color=highlight_colors.get(row['arm'],'#999999'),label=display_labels.get(row['arm'],'Other five settings' if i==2 else '_nolegend_'),marker='s' if i>=6 else 'o',linestyle='--' if i>=6 else '-',lw=2 if row['arm'] in display_labels else 1,ms=4,alpha=1 if row['arm'] in display_labels else .35,zorder=3 if row['arm'] in display_labels else 1)
for ax in axs:
 ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False);ax.axhline(0,color='#777',lw=.7)
axs[0].set_title('Most norm reduction comes from biases',loc='left');axs[0].set_ylabel('Matrix-group mean reduction (%)');axs[1].set_ylabel('Bias-group mean reduction (%)');axs[1].set_xlabel('Chisel optimizer updates');axs[1].set_xticks([x for x in [0,8,32,64,128] if x<=a.through]);axs[1].set_xlim(0,a.through)
fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.015),ncol=1,fontsize=11,frameon=False);fig.subplots_adjust(left=.2,right=.96,top=.93,bottom=.29,hspace=.2);fig.savefig(a.output/'matrix-bias.svg');fig.savefig(a.output/'matrix-bias.png',dpi=160);plt.close(fig)
