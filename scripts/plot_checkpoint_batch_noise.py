"""Plot the measured finite-panel gradient statistics, not convergence forecasts."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();data=json.loads(args.source.read_text());args.output.mkdir(parents=True,exist_ok=True)
    count=data['checkpoints'];mean=data['panel_mean_gradient_norm'];variance=data['checkpoint_gradient_variance_trace']
    if mean<=0:raise ValueError('relative gradient error is undefined for zero panel mean')
    batches=np.arange(1,count+1)
    errors=np.sqrt((1/batches-1/count)*variance)/mean*100
    plt.rcParams.update({'font.size':14,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,1,figsize=(9,10),layout='constrained')
    axes[0].scatter(data['state_ids'],np.sqrt(data['checkpoint_squared_gradient_norms']),s=19,color='#216382')
    axes[0].set(xlabel='GPT checkpoint iteration',ylabel='Checkpoint gradient norm\n(before averaging and clipping)',yscale='log',
        title='Each checkpoint contributes a different gradient')
    axes[1].plot(batches[batches>=8],errors[batches>=8],color='#216382',linewidth=2)
    for batch in (40,80,200):
        if batch<=count:
            value=float(errors[batch-1]);axes[1].scatter([batch],[value],s=45,color='#9b601a')
            axes[1].annotate(f'{batch}: {value:.1f}%',(batch,value),xytext=(0,10),textcoords='offset points',ha='center')
    axes[1].set(xlabel='Checkpoints per Chisel update',ylabel=f'Expected RMS difference\nfrom the {count}-checkpoint gradient (%)',
        xlim=(8,count+12),title='Smaller batches trade fewer tokens for a noisier gradient')
    for ax in axes:ax.grid(alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('One frozen Chisel policy · 64 sequences per checkpoint',fontsize=17)
    fig.savefig(args.output/'gradient-batches.svg');fig.savefig(args.output/'gradient-batches.png',dpi=150);plt.close(fig)
    record=dict(source=str(args.source),source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        measurements=data,subset_estimates=[dict(checkpoints=int(b),rms_difference_percent=float(e)) for b,e in zip(batches,errors)],
        caveat='Uniform subsets of the measured200 panel, not the exact stratified production sampler; raw gradients, not Adam updates or convergence.')
    (args.output/'data.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
