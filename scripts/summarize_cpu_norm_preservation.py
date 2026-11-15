"""Plot actual measured weights, not the following optimizer update's losses."""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(path):
    source = path / 'metrics.jsonl'
    data = source.read_bytes()
    # A running writer can append while we plot. Freeze complete JSONL records
    # once and hash that exact prefix rather than rereading a newer log below.
    data = data[:data.rfind(b'\n')+1]
    if not data:
        raise ValueError('No complete metric records yet')
    records = [json.loads(line) for line in data.splitlines()]
    rows = []
    for record in records:
        for row in record['rows']:
            rows.append(dict(
                updates=record['completed_updates_before_measure'],
                state=row['local_id'], tokens=row['token_count'],
                shrinkage_percent=row['mean_group_l2_percentage_reduction'],
                mean_kl=row['mean_kl'], max_kl=row['token_kl_quantiles']['max'],
                top100_changes=row['strict_topk_changes'],
                greedy_changes=row['tie_aware_argmax_changes'],
                norm_loss=row['norm_objective'],
                ranking_loss=row['ranking_penalty'],
                tail_loss=row['tail_quadratic_penalty'],
                dual_loss=row['dual_linear_term'],
                objective=row['objective'],
                measurement_scope=row.get('panel_scope', 'fixed_training_panel'),
                sequence_ids=row.get('context_ids', []),
                panel_indices=row.get('panel_indices'),
            ))
    metadata = json.loads((path / 'metadata.json').read_text())
    final_endpoint = not records[-1]['learning'] and not records[-1].get('gradient_profile',False)
    rotating = metadata.get('pool_schedule') is not None
    return dict(
        scope=('Rotating training subsets on two training checkpoints; final readout covers the entire training pool. '
               if rotating else 'Fixed training sequences on two training checkpoints. ') + 'No heldout or online claim.',
        rotating_training_subsets=rotating,
        pool_schedule=metadata.get('pool_schedule'),
        completed_updates=records[-1]['updates_after'],
        last_measured_update=records[-1]['completed_updates_before_measure'],
        final_endpoint_present=final_endpoint,
        initialization_sources=[dict(paths=c['paths'], sha256=c['sha256']) for c in metadata['cases']],
        trainable_policy=metadata['args']['policy'],
        timing_convention='Horizontal axis counts completed updates at measurement time. Training rows precede their optimizer update; final_endpoint_present identifies a final forward-only measurement.',
        norm_convention='Equal average percentage L2 reduction across editable functional groups; positive is shrinkage.',
        rank_convention='Token positions with any top100 order or membership change; exact teacher ties interchangeable.',
        objective_warning='Total objectives can change when dual multipliers change; fixed-coefficient preservation penalties are plotted separately.' +
            (' Successive training measurements use different sequences; their differences do not isolate learning. The full-pool final readout is shown separately.' if rotating else ''),
        source=str(source), source_prefix_bytes=len(data),
        source_hash_scope='Complete JSONL prefix at capture time; the live log may subsequently grow.',
        source_sha256=hashlib.sha256(data).hexdigest(), rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    result = summarize(args.run)
    (args.run / 'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for index, state in enumerate(sorted({r['state'] for r in result['rows']})):
        rows = [r for r in result['rows'] if r['state'] == state]
        subsets = [('measured', rows)]
        if result['rotating_training_subsets']:
            subsets = [('minibatch', [r for r in rows if r['measurement_scope']=='training_subset']),
                       ('full training pool', [r for r in rows if r['measurement_scope']=='full_training_pool'])]
        for scope, selected in subsets:
            if not selected: continue
            x = [r['updates'] for r in selected]
            rotating = result['rotating_training_subsets']
            style = dict(color=f'C{index}', marker='D' if scope=='full training pool' else ('o' if index==0 else 's'),
                         linestyle='none' if rotating else ('-' if index==0 else '--'),
                         label=f'GPT checkpoint {state}' + (f': {scope}' if rotating else ''))
            axes[0,0].plot(x, [r['shrinkage_percent'] for r in selected], **style)
            axes[0,1].plot(x, [r['max_kl'] for r in selected], **style)
            axes[1,0].plot(x, [100*r['top100_changes']/r['tokens'] for r in selected], **style)
            axes[1,1].plot(x, [r['ranking_loss']+r['tail_loss'] for r in selected], **style)
    titles = ['Norm shrinkage by checkpoint', 'Worst-token preservation', 'Top-100 ranking preservation', 'Fixed-coefficient preservation penalties']
    labels = ['Mean group norm reduction (%)\npositive = shrinkage', 'Maximum token KL (nats)', 'Training token positions changed (%)', 'Ranking + tail loss\nfixed coefficients; lower is better']
    for ax, title, label in zip(axes.flat, titles, labels):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Completed Chisel updates')
        ax.set_ylabel(label)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0,0].axhline(0, color='gray', linewidth=.8)
    axes[0,1].set_ylim(bottom=0)
    axes[0,1].ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
    if axes[0,1].get_ylim()[1] >= 1e-10:
        axes[0,1].axhline(1e-10, color='gray', linestyle=':', label='token KL tolerance')
        axes[0,1].legend(fontsize=8)
    # A fixed 0..100% axis hides the few changed tokens we must eliminate.
    maximum_change = max(100*r['top100_changes']/r['tokens'] for r in result['rows'])
    one_token = max(100/r['tokens'] for r in result['rows'])
    axes[1,0].set_ylim(0, max(one_token, maximum_change*1.15))
    axes[1,1].set_ylim(bottom=0)
    axes[1,1].ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
    first_update = min(r['updates'] for r in result['rows'])
    tokens = sum(r['tokens'] for r in result['rows'] if r['updates']==first_update)
    coverage = (('Rotating training minibatches; diamonds evaluate the entire training pool'
                 if result['final_endpoint_present'] else
                 'Rotating training minibatches; final full-pool evaluation pending')
                if result['rotating_training_subsets'] else f'Two training checkpoints; {tokens:,} fixed training token positions total')
    fig.suptitle(f'Chisel training: shrinkage and preservation\n{coverage}', fontsize=12)
    fig.savefig(args.run / 'loss_components.png', dpi=160)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
