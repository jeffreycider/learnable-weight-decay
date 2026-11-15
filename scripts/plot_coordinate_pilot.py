"""Closed eight-arm coordinate prototype readouts; no experiment selection."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(source,output,curves=None):
    source=Path(source);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    data=json.loads(source.read_text());arms={a['name']:a for a in data['arms']}
    settings=[('0.0001','1e-08'),('0.0001','1e-12'),('0.001','1e-08'),('0.001','1e-12')]
    labels=['10⁻⁴ / 10⁻⁸','10⁻⁴ / 10⁻¹²','10⁻³ / 10⁻⁸','10⁻³ / 10⁻¹²'];x=np.arange(4)
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    def arm(i,kind):return arms[f'private-lr{settings[i][0]}-eps{settings[i][1]}-{kind}']
    def query(a,domain):return next(v['query'][-1] for v in a['transfer']['results'] if v['domain']==domain)
    fig,axes=plt.subplots(2,1,figsize=(7,7.5),layout='constrained')
    for ax,domain,title in zip(axes,['gpt','cifar10'],['GPT: top-100 strict rank changes','CIFAR: full ten-class strict rank changes']):
        for kind,offset,color,label in [('frozen',-.18,'#a5a5a5','Frozen initial residual blocks'),('trained',.18,'#2769b1','Trained residual blocks')]:
            values=[]
            for i in range(4):
                q=query(arm(i,kind),domain);values.append(100*q['strict_topk_changes']/q['token_count'])
            bars=ax.bar(x+offset,values,width=.34,color=color,label=label)
            if domain=='cifar10':
                for b in bars:ax.text(b.get_x()+b.get_width()/2,.015,'0',ha='center',va='bottom',fontsize=10)
        ax.set_title(title);ax.set_ylabel('Query contexts changed (%)');ax.set_xticks(x,labels);ax.set_xlabel('Private learning rate / Adam epsilon')
        if domain=='cifar10':ax.set_ylim(0,.3)
        ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=10)
    fig.suptitle('Local probes fit, but GPT ranking changes remain',fontsize=16)
    fig.savefig(output/'query.svg');fig.savefig(output/'query.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5.5),layout='constrained')
    for family,offset,color,label in [('encoder',-.27,'#4586bb','GPT private encoders'),('decoder',-.09,'#4a956a','GPT private decoders'),('trajectory',.09,'#cb8c3c','GPT private trajectory maps'),('blocks',.27,'#8c5caa','Shared residual blocks')]:
        values=[]
        for i in range(4):
            a=arm(i,'trained')
            value=a['shared_motion']['blocks']['relative_l2'] if family=='blocks' else np.mean([v[family]['relative_l2'] for k,v in a['private_motion'].items() if k.startswith("('gpt'")])
            values.append(100*value)
        ax.bar(x+offset,values,width=.17,color=color,label=label)
    ax.set_yscale('log');ax.set_ylabel('Relative parameter L2 change (%)');ax.set_xticks(x,labels);ax.set_xlabel('Private learning rate / Adam epsilon');ax.set_title('Private maps move much more than the shared backbone')
    ax.legend(fontsize=10,loc='upper left');ax.grid(axis='y',alpha=.2)
    fig.savefig(output/'movement.svg');fig.savefig(output/'movement.png',dpi=150);plt.close(fig)
    if curves:
        from matplotlib.lines import Line2D
        colors=['#2769b1','#37a080','#d28a32','#9d5fa3']
        fig,axes=plt.subplots(2,2,figsize=(10,8),layout='constrained')
        curve_data={}
        for i in range(4):
            for kind in ('frozen','trained'):
                a=arm(i,kind);path=Path(curves)/a['name']/'results/metrics.jsonl'
                reads=[json.loads(l) for l in path.read_text().splitlines() if json.loads(l)['event']=='fixed_fitting_readout']
                curve_data[a['name']]=dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),readouts=reads)
                for column,domain in enumerate(('gpt','cifar10')):
                    xx=[];changed=[];shrink=[]
                    for read in reads:
                        rr=[r for r in read['rows'] if r['domain']==domain];xx.append(read['update'])
                        changed.append(100*sum(r['strict_topk_changes'] for r in rr)/sum(r['token_count'] for r in rr))
                        shrink.append(sum(r['mean_group_l2_percentage_reduction'] for r in rr)/len(rr))
                    style='--' if kind=='frozen' else '-'
                    axes[0,column].plot(xx,changed,style,color=colors[i],lw=1.8,marker='o',ms=3)
                    axes[1,column].plot(xx,shrink,style,color=colors[i],lw=1.8,marker='o',ms=3)
        for column,title in enumerate(('GPT fitting states 6 and 499','CIFAR fitting models 0 and 40')):
            axes[0,column].set_title(title);axes[0,column].set_ylabel('Strict rank changes (% of contexts)')
            axes[1,column].set_ylabel('Mean group L2 reduction (%)')
            for ax in axes[:,column]:ax.set_xlabel('Shared optimizer updates');ax.grid(alpha=.2);ax.set_xlim(0,32)
        handles=[Line2D([],[],color=colors[i],label='Private LR / epsilon '+labels[i]) for i in range(4)]
        handles+=[Line2D([],[],color='black',ls='-',label='Trained residual blocks'),Line2D([],[],color='black',ls='--',label='Frozen initial residual blocks')]
        fig.legend(handles=handles,loc='outside lower center',ncol=2,fontsize=10)
        fig.suptitle('Private probes learn shrinkage; GPT fitting ranks still change',fontsize=16)
        fig.savefig(output/'fitting.svg');fig.savefig(output/'fitting.png',dpi=150);plt.close(fig)
        (output/'fitting-data.json').write_text(json.dumps(curve_data,indent=2))
    (output/'data.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),scope=data['scope'],settings=settings,arms=data['arms']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--curves');a=p.parse_args();plot(a.input,a.output,a.curves)
