"""Plot atomic online mirrors without selecting checkpoints or changing training.

All plots retain the actual recorded iteration indices. Sparse diagnostics are
points, not interpolated measurements. The plot data and source hashes accompany
the SVG/PNG files so the report can cite the exact inputs it renders.
"""
import argparse
import hashlib
import json
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


COLORS = {'learned': '#007a87', 'shadow': '#a45200'}
LABELS = {'learned': 'Chisel + next-batch learning', 'shadow': 'Matched control: no edits'}


def read_mirror(path):
    payload = path.read_bytes()
    document = json.loads(payload)
    # On a resume the logger may repeat an event. Equal duplicates add no data;
    # conflicting duplicates must be investigated rather than silently averaged.
    unique = {}
    for row in document['rows']:
        if row.get('event') not in ('train', 'validation', 'weight_group_norms'):
            continue
        key = row['event'], row['step']
        if key in unique and unique[key] != row:
            raise ValueError(f'Conflicting repeated event: {path}: {key}')
        unique[key] = row
    return sorted(unique.values(), key=lambda r: r['step']), hashlib.sha256(payload).hexdigest()


def values(rows, key, event='train'):
    return [(r['step'], r[key]) for r in rows if r['event'] == event and key in r]


def plot(inputs, output, *, mobile=False):
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 14 if mobile else 12, 'axes.titlesize': 15,
                         'svg.fonttype': 'none', 'axes.spines.top': False,
                         'axes.spines.right': False})
    data, hashes = {}, {}
    for arm, path in inputs.items():
        data[arm], hashes[arm] = read_mirror(path)
    record = {'generated_utc': datetime.now(timezone.utc).isoformat(),
              'source_sha256': hashes, 'mobile': mobile, 'figures': {}}

    def figure(name, title, ylabel, note):
        fig, ax = plt.subplots(figsize=(6, 8) if mobile else (10, 5.5))
        fig.subplots_adjust(left=.19 if mobile else .12, right=.97,
                            bottom=.42 if mobile else .24, top=.82 if mobile else .83)
        display_title = textwrap.fill(re.sub(r'^\d+\.\s*', '', title), 43) if mobile else title
        fig.suptitle(display_title, x=.07 if mobile else .12, ha='left', fontsize=17 if mobile else 16)
        wrapped = '\n'.join(textwrap.fill(part, width=57 if mobile else 110) for part in note.splitlines())
        fig.text(.07 if mobile else .12, .035 if mobile else .04, wrapped,
                 va='bottom', fontsize=11 if mobile else 10)
        if mobile:
            ylabel = textwrap.fill(ylabel, 32)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.set(xlabel='Completed GPT iterations', ylabel=ylabel)
        ax.grid(alpha=.18)
        record['figures'][name] = {'title': title, 'note': note, 'series': {}}
        return fig, ax

    def line(ax, name, label, pairs, color, *, points=False, linestyle='-'):
        if pairs:
            x, y = zip(*pairs)
            ax.plot(x, y, color=color, label=label, marker='o' if points else None,
                    markersize=4, linewidth=1.6, linestyle='none' if points else linestyle)
            record['figures'][name]['series'][label] = pairs

    def save(fig, ax, name):
        if ax.get_legend_handles_labels()[0]:
            if mobile:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, [textwrap.fill(label, 35) for label in labels], fontsize=11.5,
                          frameon=False, loc='upper left', bbox_to_anchor=(-.10, -.26))
            else:
                ax.legend(fontsize=10, frameon=False)
        else:
            ax.text(.5, .5, 'No measurements yet', ha='center', transform=ax.transAxes)
        for suffix in ('svg', 'png'):
            fig.savefig(output/f'{name}.{suffix}', dpi=150)
        plt.close(fig)

    fig, ax = figure('training-loss', '1. Training loss on each ordinary GPT batch',
                     'Mean cross-entropy (nats/token)',
                     '524,288 tokens per iteration; FP32 GPT; raw values, no smoothing.\n'
                     'Consecutive batches differ. A falling curve alone does not establish a Chisel benefit.')
    for arm, rows in data.items():
        line(ax, 'training-loss', LABELS[arm], values(rows, 'train_loss'), COLORS[arm])
    save(fig, ax, 'training-loss')

    fig, ax = figure('validation-loss', '2. Validation loss at scheduled checkpoints',
                     'Mean cross-entropy (nats/token)',
                     'Same fixed validation data and live-weight readout; no validation feedback to Chisel.\n'
                     'The intended endpoint is iteration 3250; incomplete curves are not endpoint results.')
    validation = {}
    for arm, rows in data.items():
        pairs = values(rows, 'val_loss', 'validation')
        validation[arm] = dict(pairs)
        line(ax, 'validation-loss', LABELS[arm], pairs, COLORS[arm], points=True)
    save(fig, ax, 'validation-loss')

    fig, ax = figure('validation-difference', '3. Does Chisel improve validation loss?',
                     'Chisel − matched control (nats/token)',
                     'Negative favors Chisel. Only matching completed checkpoints are compared.\n'
                     'One common training prefix; no across-seed reliability claim.')
    common = sorted(validation.get('learned', {}).keys() & validation.get('shadow', {}).keys())
    differences = [(t, validation['learned'][t]-validation['shadow'][t]) for t in common]
    line(ax, 'validation-difference', 'Paired difference', differences, COLORS['learned'], points=True)
    ax.axhline(0, color='#666666', linewidth=1, linestyle='--')
    save(fig, ax, 'validation-difference')

    fig, ax = figure('action-shrinkage', '4. How much does each measured Chisel edit shrink?',
                     'L2 norm reduction per edit (%)',
                     'Positive means shrinkage; negative means growth. Points are recorded every 64 steps.\n'
                     'Each percentage uses weights immediately before that edit, not the original checkpoint.')
    rows = data.get('learned', [])
    pairs = values(rows, 'chisel_action_mean_group_l2_percentage_reduction')
    line(ax, 'action-shrinkage', 'Equal mean of all 15 weight/bias groups', pairs, COLORS['learned'], points=True)
    for suffix, label, color in (('.weight', 'Mean of 8 matrix groups', '#333333'),
                                 ('.bias', 'Mean of 7 bias groups', '#a45200')):
        pairs = []
        for row in rows:
            groups = row.get('chisel_action_group_l2_percentage_reduction', {})
            selected = [v for k, v in groups.items() if k.endswith(suffix)]
            if selected:
                pairs.append((row['step'], sum(selected)/len(selected)))
        line(ax, 'action-shrinkage', label, pairs, color, points=True)
    ax.axhline(0, color='#666666', linewidth=1, linestyle='--')
    save(fig, ax, 'action-shrinkage')

    fig, ax = figure('matrix-action-shrinkage', 'Matrix shrinkage on its own scale',
                     'Mean matrix-group L2 reduction per edit (%)',
                     'Equal mean over 8 matrix groups, including embedding and output head; biases excluded.\n'
                     'Positive means shrinkage. Sparse measured points; this scale differs from the all-group plot.')
    pairs = record['figures']['action-shrinkage']['series'].get('Mean of 8 matrix groups', [])
    line(ax, 'matrix-action-shrinkage', 'Mean of 8 matrix groups', pairs, COLORS['learned'], points=True)
    ax.axhline(0, color='#666666', linewidth=1, linestyle='--')
    save(fig, ax, 'matrix-action-shrinkage')

    fig, ax = figure('iteration-time', '5. Online GPT iteration time, including Chisel',
                     'Slowest-rank wall time (seconds)',
                     'Includes data, GPT forward/backward, optimizer, sensors, Chisel and in-step diagnostics.\n'
                     'Excludes validation and checkpoint I/O. First step includes cold startup; 8 H100s.')
    for arm, rows in data.items():
        line(ax, 'iteration-time', LABELS[arm], values(rows, 'iteration_wall_seconds_max_rank'), COLORS[arm])
    save(fig, ax, 'iteration-time')

    fig, ax = figure('edit-base-ratio', 'Chisel edit sizes relative to base optimizer updates',
                     'Edit norm / base update norm',
                     'The most recent 250 recorded iterations, using whole-model L2 norms.\n'
                     'The horizontal line marks equal size; the final identity edit has ratio zero.')
    motion = values(data.get('learned', []), 'chisel_edit_to_base_norm_ratio_all_parameters')
    if motion:
        cutoff = motion[-1][0] - 249
        motion = [(step, ratio) for step, ratio in motion if step >= cutoff]
    line(ax, 'edit-base-ratio', 'Chisel edit / base update', motion, COLORS['learned'])
    if motion:
        ax.axhline(1, color='#666666', linewidth=1, linestyle='--', label='Equal size')
    ax.set_ylim(bottom=0)
    save(fig, ax, 'edit-base-ratio')

    fig, ax = figure('task-gradient', '6. Learning signal from the next ordinary GPT batch',
                     'Chisel gradient L2 norm, before clipping',
                     'One delayed Chisel update per executed edit; no extra GPT backward.\n'
                     'Gradient magnitude measures learning activity, not whether the edit helps validation.')
    rows = data.get('learned', [])
    line(ax, 'task-gradient', 'Measured gradient norm', values(rows, 'task_gradient_norm'), COLORS['learned'])
    limits = values(rows, 'task_gradient_clip_threshold')
    line(ax, 'task-gradient', 'Clipping threshold', limits, '#666666', linestyle='--')
    save(fig, ax, 'task-gradient')

    fig, ax = figure('net-shrinkage', '7. Weight norms compared with the matched control',
                     'Weight / bias group',
                     'Positive means a smaller Chisel-run norm at the same completed iteration.\n'
                     'Includes all subsequent GPT optimization; this is not a sum of individual edit percentages.')
    norms = {arm: {r['step']: r['group_l2_norm'] for r in rows
                   if r['event'] == 'weight_group_norms'} for arm, rows in data.items()}
    common = sorted(norms.get('learned', {}).keys() & norms.get('shadow', {}).keys())
    net_xlabel = 'Norm reduction versus control (%)'
    ax.set_xlabel(textwrap.fill(net_xlabel, 30) if mobile else net_xlabel)
    if common:
        step = common[-1]
        reference, treated = norms['shadow'][step], norms['learned'][step]
        if reference.keys() != treated.keys() or len(reference) != 15:
            raise ValueError('Expected the same 15 weight/bias groups')
        if any(v <= 0 for v in reference.values()):
            raise ValueError('A zero reference norm needs explicit treatment')
        pairs = [(g, 100*(1-treated[g]/reference[g])) for g in reference]
        labels, differences = zip(*pairs)
        fig.set_size_inches(6, 10.5) if mobile else fig.set_size_inches(10, 8)
        fig.subplots_adjust(left=.40 if mobile else .30, bottom=.32 if mobile else .18)
        if mobile:
            ax.tick_params(axis='y', labelsize=11)
        ax.barh(labels, differences, color=COLORS['learned'], label=f'Iteration {step}')
        ax.invert_yaxis()
        ax.axvline(0, color='#666666', linewidth=1)
        record['figures']['net-shrinkage']['series'][f'Iteration {step}'] = pairs
    save(fig, ax, 'net-shrinkage')
    fig, ax = figure('matrix-net-shrinkage', 'Matrix norms compared with the matched control',
                     'Matrix group',
                     'The eight matrix groups use their own scale so bias changes cannot hide them.\n'
                     'Positive means a smaller norm at the same completed iteration, after all GPT updates.')
    ax.set_xlabel(textwrap.fill(net_xlabel, 30) if mobile else net_xlabel)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    if common:
        pairs = [(g, 100*(1-treated[g]/reference[g])) for g in reference if g.endswith('.weight')]
        if len(pairs) != 8:
            raise ValueError('Expected eight matrix groups')
        labels, differences = zip(*pairs)
        fig.set_size_inches(6, 9) if mobile else fig.set_size_inches(10, 6)
        fig.subplots_adjust(left=.33 if mobile else .24, bottom=.36 if mobile else .24)
        ax.barh([g.removesuffix('.weight') for g in labels], differences,
                color=COLORS['learned'], label=f'Iteration {step}')
        ax.invert_yaxis()
        ax.axvline(0, color='#666666', linewidth=1)
        record['figures']['matrix-net-shrinkage']['series'][f'Iteration {step}'] = pairs
    save(fig, ax, 'matrix-net-shrinkage')
    (output/'plot-data.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--learned', type=Path, required=True)
    parser.add_argument('--shadow', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mobile', action='store_true', help='phone-readable layout with no internal figure numbering')
    args = parser.parse_args()
    inputs = {'learned': args.learned}
    if args.shadow is not None:
        inputs['shadow'] = args.shadow
    result = plot(inputs, args.output, mobile=args.mobile)
    print(json.dumps({'figures': list(result['figures']), 'source_sha256': result['source_sha256']}))
