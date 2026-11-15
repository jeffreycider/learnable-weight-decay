"""Plot closed fitting trials and the separate numerical construction control."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/research/adaptive-probes'
OUT.mkdir(parents=True, exist_ok=True)
sources = {}


def read(path):
    data = (ROOT / path).read_bytes()
    sources[path] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def save(fig, name):
    fig.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=170, bbox_inches='tight')
    plt.close(fig)


plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.facecolor': '#faf9f5',
                     'axes.facecolor': '#faf9f5'})
ladder = read('runs/chisel-owner/adaptive-fitting-ladder/latest.json')
for arm in ladder['arms'].values():
    assert arm['status']['status'] == 'completed'
    assert arm['last_update'] == 32 and arm['readouts'][-1]['update'] == 32
control = read('runs/adaptive-gauge-representation/finite/summary.json')

fig, axes = plt.subplots(2, 1, figsize=(4.8, 5.4), sharex=True)
styles = [('direct_binary', 'Direct rescaling', '#aaa', -.25),
          ('adaptive_fp32', 'FP32 factors', '#285a70', 0),
          ('adaptive_fp64_cast', 'FP64 factors → FP32 weights', '#cf9c57', .25)]
for key, label, color, offset in styles:
    for ax, metric in zip(axes, ('strict_topk_changes', 'mean_group_l2_percentage_reduction')):
        values = [r['arms'][key][metric] if metric == 'strict_topk_changes'
                  else r['arms'][key]['norm'][metric] for r in control['results']]
        bars = ax.bar(np.arange(2) + offset, values, .23, label=label, color=color)
        if metric == 'strict_topk_changes':
            for bar, value in zip(bars, values):
                ax.annotate(str(value), (bar.get_x()+bar.get_width()/2, value),
                            xytext=(0, 3), textcoords='offset points', ha='center')
axes[0].set_title('Figure 1. Factor rounding breaks an\notherwise exact rescaling', loc='left')
axes[0].set_ylabel('Changed top-100 positions\nout of 16,384 (target: zero)')
axes[0].set_ylim(0, 95)
axes[1].set_ylabel('Mean group norm reduction (%)')
axes[1].set_xticks([0, 1], ['GPT checkpoint 6', 'GPT checkpoint 499'])
axes[1].legend(fontsize=8, frameon=False, loc='upper left')
save(fig, 'precision')

fig, axes = plt.subplots(4, 1, figsize=(4.8, 9), sharex=True)
colors = ['#285a70', '#b27b35', '#777777']
linestyles = ['-', '--', ':']
for name, color, ls in zip(('head1e-8', 'head1e-7', 'head1e-6'), colors, linestyles):
    arm = ladder['arms'][name]
    for row, state in enumerate((6, 499)):
        points = [(readout['update'], next(r for r in readout['rows']
                   if r['domain'] == 'gpt' and r['local_id'] == state))
                  for readout in arm['readouts']]
        steps = [p[0] for p in points]
        for offset, metric in enumerate(('strict_topk_changes', 'mean_group_l2_percentage_reduction')):
            values = [p[1][metric] for p in points]
            if offset == 0:
                values = [100*v/p[1]['token_count'] for v, p in zip(values, points)]
            axes[row*2+offset].plot(steps, values, color=color, linestyle=ls, marker='o',
                                   markersize=3, label='Head LR '+name[4:])
for row, state in enumerate((6, 499)):
    axes[2*row].set_title(f'GPT checkpoint {state}', fontsize=10, loc='left')
    axes[2*row].set_yscale('log')
    axes[2*row].set_ylim(.1, 100)
    axes[2*row].set_yticks([.1, 1, 10, 100], ['0.1', '1', '10', '100'])
    axes[2*row].minorticks_off()
    axes[2*row].set_ylabel('Changed top-100\npositions (%)')
    axes[2*row+1].set_ylabel('Mean group norm\nreduction (%)')
    axes[2*row+1].axhline(0, color='#999', linewidth=.8)
    axes[2*row+1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
axes[0].legend(fontsize=8, frameon=False)
axes[-1].set_xlabel('Shared Chisel updates')
axes[-1].set_xticks([8, 16, 24, 32])
fig.suptitle('Figure 2. Larger tested learning rates\ndo not solve fitting preservation', fontsize=11, y=1.01)
save(fig, 'fitting')

fig, axes = plt.subplots(3, 1, figsize=(4.8, 6.5), sharex=True)
for state, color, ls in ((6, '#285a70', '-'), (499, '#777777', '--')):
    points = [(p['update'], next(r for r in p['rows'] if r['domain'] == 'gpt' and r['local_id'] == state))
              for p in ladder['arms']['head1e-8']['readouts']]
    for ax, metric in zip(axes, ('mean_group_l2_percentage_reduction', 'ranking_penalty', 'mean_kl')):
        ax.plot([p[0] for p in points], [p[1][metric] for p in points],
                marker='o', color=color, linestyle=ls, label=f'GPT checkpoint {state}')
axes[0].set_title('Figure 3. Ranking penalty remains\nafter 32 updates at the smallest LR', loc='left')
axes[0].set_ylabel('Norm reward\n(% group reduction)')
axes[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
axes[0].legend(fontsize=8, frameon=False)
axes[1].set_ylabel('Weighted ranking penalty')
axes[2].set_ylabel('Mean KL (nats/token)')
axes[2].set_yscale('log')
axes[2].axhline(1e-10, color='#999', linestyle=':', label='Mean KL constraint')
axes[2].legend(fontsize=8, frameon=False)
axes[-1].set_xlabel('Shared Chisel updates'); axes[-1].set_xticks([8, 16, 24, 32])
save(fig, 'components')

(OUT / 'data.json').write_text(json.dumps({'sources': sources, 'ladder': ladder,
    'precision_control': control, 'scope': 'Closed TRAIN-only readouts; no held-out or online claims.'}, indent=2)+'\n')
print(json.dumps({'output': str(OUT), 'figures': ['precision', 'fitting', 'components'], 'sources': sources}))
