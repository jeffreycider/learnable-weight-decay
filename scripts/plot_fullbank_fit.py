"""Plot the current full-bank fit from an atomic, provenance-bound monitor copy."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot(source, output, through_update=None):
    raw = source.read_bytes()
    snapshot = json.loads(raw)
    rows = snapshot['rows']
    if through_update is not None:
        if through_update < 0:
            raise ValueError('Nonnegative plot cutoff required')
        rows = [r for r in rows if r.get('update', 0) <= through_update]
    updates = [r for r in rows if r.get('event') == 'update']
    audits = [r for r in rows if r.get('event') == 'audit_summary']
    start = next(r for r in rows if r.get('event') == 'start')
    # Bind the plotted target to the recorded run settings, never a chart default.
    options = start['metadata']['options']
    if 'kl_tolerance' not in options:
        raise ValueError('Run start must expose the actual KL tolerance')
    target = options['kl_tolerance']
    coefficient = options['shrinkage_coefficient']
    tokenwise = options.get('objective') == 'tokenwise-v1'
    for r in updates:
        expected = -coefficient*r['mean_group_l2_percentage_reduction'] + r['kl_penalty']
        if abs(expected-r['objective']) > 1e-6*max(1., abs(expected)):
            raise ValueError('Loss components do not reproduce logged objective')
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 13, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'})

    def save(fig, name):
        fig.tight_layout()
        for ext in ('svg', 'png'):
            fig.savefig(output/f'{name}.{ext}', dpi=160, bbox_inches='tight')
        plt.close(fig)

    keys = (('mean_kl', 'maximum_token_kl', 'top10_order_changes_fraction', 'mean_group_l2_percentage_reduction')
            if tokenwise else ('mean_kl', 'maximum_kl', 'mean_group_l2_percentage_reduction'))
    fig, axes = plt.subplots(len(keys), 1, figsize=(6, 3.3*len(keys)), sharex=True)
    for ax, key in zip(axes, keys):
        scale = 100 if key == 'top10_order_changes_fraction' else 1
        ax.plot([r['update'] for r in updates], [scale*r[key] for r in updates],
                color='#a4a4a4', linewidth=1, label='Changing training batches')
        for role, label, color, marker in (
                ('fixed_training', 'Fixed seen tokens (32 checkpoints)', '#135c86', 'o'),
                ('heldout_sequences', 'Withheld tokens (same 32)', '#986226', 's'),
                ('heldout_states', 'Withheld checkpoints (200)', '#45764a', '^')):
            points = [r for r in audits if r['role'] == role]
            ax.plot([r['update'] for r in points], [scale*r[key] for r in points],
                    marker=marker, color=color, markersize=5, label=label)
        ax.grid(alpha=.15)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    axes[0].axhline(target, color='#333', linestyle='--', label=f'Average-KL target: {target:g}')
    axes[0].set(yscale='log', ylabel='Mean checkpoint KL\n(nats/token)',
                title='Preservation and shrinkage during fitting')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=11, loc='upper center', bbox_to_anchor=(.5, -.01))
    if tokenwise:
        axes[1].axhline(options['token_tail_tolerance'], color='#333', linestyle='--')
        axes[1].set(yscale='log', ylabel='Maximum individual-token KL\n(dashed line: tail penalty threshold)')
        axes[2].set(ylabel='Top-10 order changed (%)')
    else:
        axes[1].axhline(target, color='#333', linestyle='--')
        axes[1].set(yscale='log', ylabel='Worst checkpoint-average KL\n(nats/token)')
    axes[-1].axhline(0, color='#444', linewidth=.8)
    axes[-1].set(xlabel='Chisel optimizer updates', ylabel='Mean norm reduction (%)')
    save(fig, 'fit')

    fig, axes = plt.subplots(3 if tokenwise else 2, 1, figsize=(6, 10 if tokenwise else 7), sharex=True)
    x = [r['update'] for r in updates]
    for key, label, color, style in (
            ('augmented_penalty', 'Quadratic preservation penalty', '#986226', '-'),
            ('dual_linear_term', 'Learned preservation penalty', '#135c86', '-'),
            ('objective', 'Total objective', '#222222', '--')):
        axes[0].plot(x, [r[key] for r in updates], label=label, color=color, linestyle=style)
    axes[0].set(yscale='symlog', ylabel='Loss contribution', title='The dual penalty changes during fitting')
    if tokenwise:
        axes[0].plot(x, [r['ranking_penalty'] for r in updates], label='Top-10 margin penalty', color='#70489b')
        axes[0].set_yscale('symlog', linthresh=1e-4)
    axes[0].legend(fontsize=10)
    axes[1].plot(x, [-coefficient*r['mean_group_l2_percentage_reduction'] for r in updates],
                 color='#45764a', label='Norm term: negative rewards shrinkage')
    axes[1].set(xlabel='Chisel optimizer updates', ylabel='Norm loss contribution')
    axes[1].legend(fontsize=10)
    if tokenwise:
        for role,label,color,marker in (
                ('fixed_training','Fixed seen tokens','#135c86','o'),
                ('heldout_sequences','Withheld tokens','#986226','s'),
                ('heldout_states','Withheld checkpoints','#45764a','^')):
            points=[r for r in audits if r['role']==role]
            values=[r['norm_objective']+r['tail_quadratic_penalty']+r['ranking_penalty'] for r in points]
            axes[2].plot([r['update'] for r in points],values,color=color,marker=marker,label=label)
        axes[2].set(xlabel='Chisel optimizer updates',ylabel='Fixed-coefficient loss\n(lower is better)',
                    title='Fixed-panel loss\nNorm + tail + ranking; excludes the dual term')
        axes[2].legend(fontsize=10)
    for ax in axes:
        ax.grid(alpha=.15)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    save(fig, 'components')
    if tokenwise:
        fig,axes=plt.subplots(2,1,figsize=(6,7),sharex=True)
        for ax,key,label in zip(axes,('matrix_group_mean_percent','bias_group_mean_percent'),
                                ('Matrix-group mean reduction (%)','Bias-group mean reduction (%)')):
            for role,name,color,marker in (('fixed_training','32 fitting checkpoints','#135c86','o'),
                                         ('heldout_states','200 development checkpoints','#45764a','^')):
                points=[r for r in audits if r['role']==role]
                ax.plot([r['update'] for r in points],[r[key] for r in points],color=color,marker=marker,label=name)
            ax.set_ylabel(label);ax.axhline(0,color='#444',linewidth=.8);ax.grid(alpha=.15)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=6))
        axes[0].set_title('Matrix and bias shrinkage need separate scales');axes[0].legend(fontsize=10)
        axes[1].set_xlabel('Chisel optimizer updates');save(fig,'families')
    record = dict(source=str(source), source_sha256=hashlib.sha256(raw).hexdigest(),
        metrics_sha256=snapshot['metrics_sha256'], captured_at=snapshot['timestamp'],
        target=target, shrinkage_coefficient=coefficient, updates=updates, audits=audits,
        through_update=through_update, source_optimizer_updates=snapshot['optimizer_updates'],
        notes=['KL target constrains a panel average, not every individual token.',
               ('Individual-token maxima and top-10 changes are shown separately from mean KL.' if tokenwise else
                'Worst checkpoint-average KL is the largest panel mean; individual-token maxima are not shown.'),
               'Training batches change checkpoints/tokens; fixed audits support comparison over time.',
               'Shrinkage is equal-weight mean relative L2 reduction across 15 parameter groups.',
               'The dual multiplier changes the objective; total objective need not decrease as KL falls.',
               'These are offline Chisel measurements, not GPT validation loss.'])
    (output/'data.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(dict(optimizer_updates=max([r['update'] for r in updates], default=0), audits=len(audits),
                         source_sha256=record['source_sha256'], output=str(output))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--through-update', type=int, default=None)
    args = parser.parse_args()
    plot(args.source, args.output, args.through_update)
