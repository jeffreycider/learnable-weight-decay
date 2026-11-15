"""Plot measured online replay training without smoothing or selecting checkpoints."""
import argparse
import json
import math
import textwrap
from datetime import datetime, timezone
from pathlib import Path

from plot_online_chisel import read_mirror, values
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot(source, output):
    rows, sha = read_mirror(source)
    rows = [r for r in rows if r['event'] == 'train']
    replay = [r for r in rows if r.get('chisel_replay_candidate_metrics')]
    output.mkdir(parents=True, exist_ok=True)
    record = dict(generated_utc=datetime.now(timezone.utc).isoformat(),
                  source=str(source), source_sha256=sha, figures={})
    plt.rcParams.update({'font.size': 13, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})

    def draw(name, title, ylabel, note, series, *, target=None, log=False):
        fig, ax = plt.subplots(figsize=(6, 8))
        fig.subplots_adjust(left=.20, right=.97, bottom=.43, top=.80)
        fig.suptitle(textwrap.fill(title, 42), x=.07, ha='left', fontsize=17)
        fig.text(.07, .025, textwrap.fill(note, 57), va='bottom', fontsize=11)
        ax.set(xlabel='Completed GPT iterations', ylabel=textwrap.fill(ylabel, 30))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.grid(alpha=.18)
        colors=['#007a87', '#a45200', '#7957a2', '#57711a']
        for (label, points), color in zip(series.items(), colors):
            if any(not math.isfinite(float(y)) for _, y in points):
                raise ValueError('Nonfinite measured series: '+label)
            if points:
                x, y = zip(*points)
                ax.plot(x, y, label=label, color=color, linewidth=1.5)
        if log:
            ax.set_yscale('symlog', linthresh=1e-9)
        if target is not None:
            ax.axhline(target, color='#666666', linestyle='--', linewidth=1,
                       label='Pilot mean KL target')
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, [textwrap.fill(x, 33) for x in labels], frameon=False,
                      fontsize=11, loc='upper left', bbox_to_anchor=(-.13, -.28))
        else:
            ax.text(.5, .5, 'No measurements yet', transform=ax.transAxes, ha='center')
        for suffix in ('png', 'svg'):
            fig.savefig(output/f'{name}.{suffix}', dpi=150)
        plt.close(fig)
        record['figures'][name] = dict(title=title, note=note, series=series,
                                       target=target, yscale='symlog' if log else 'linear')

    tolerances = {c['kl_tolerance'] for r in replay for c in r['chisel_replay_candidate_metrics']}
    if len(tolerances)>1:
        raise ValueError('Cannot plot changing KL targets as one fixed target')
    target = next(iter(tolerances), None)
    draw('replay-kl', 'Does online replay keep Chisel edits small in prediction space?',
         'Mean prediction KL (nats/token)',
         'Each GPT step uses a new panel of past training sequences. Both curves use that same panel; '
         'final KL is measured after auxiliary training. These are fitting measurements, not held-out results.',
         {'Before auxiliary training': [(r['step'], r['chisel_replay_candidate_metrics'][0]['candidate_replay_kl']) for r in replay],
          'Final proposed edit': values(replay, 'chisel_replay_final_kl')}, target=target, log=True)

    shrink = {'Mean of all 15 groups': values(rows, 'chisel_action_mean_group_l2_percentage_reduction')}
    for label, suffix in [('Mean of 8 matrix groups', '.weight'), ('Mean of 7 bias groups', '.bias')]:
        points=[]
        for r in rows:
            group=r.get('chisel_action_group_l2_percentage_reduction', {})
            measured=[v for k,v in group.items() if k.endswith(suffix)]
            if measured:
                expected=8 if suffix=='.weight' else 7
                if len(measured)!=expected:
                    raise ValueError('Unexpected group inventory for '+label)
                points.append((r['step'], sum(measured)/len(measured)))
        shrink[label]=points
    draw('replay-shrinkage', 'How much does each committed Chisel edit shrink?',
         'Mean group L2 reduction (%)',
         'Positive means shrinkage; negative means growth. Each edit is compared with its own post-optimizer '
         'weights. These values are not cumulative reductions relative to the control.', shrink)
    draw('replay-matrix-shrinkage', 'Matrix shrinkage on its own scale',
         'Mean matrix L2 reduction (%)',
         'Equal mean over the eight matrix groups, including embedding and output head. Biases are excluded. '
         'Positive means shrinkage. This scale differs from the all-group plot.',
         {'Mean of 8 matrix groups': shrink['Mean of 8 matrix groups']})
    draw('replay-dual', 'How strongly is preservation penalized?', 'Preservation multiplier',
         'The multiplier shown is its value after auxiliary training at each GPT step. It increases when '
         'the measured mean KL exceeds the target. A changing multiplier makes total training loss unsuitable '
         'as a stationary measure of progress.',
         {'Multiplier after auxiliary updates': values(replay, 'chisel_replay_dual_multiplier')})
    draw('replay-runtime', 'Where does online training time go?', 'Measured wall time (seconds)',
         'Whole iteration uses the slowest rank; component timers are recorded on rank zero and are included '
         'in the whole iteration. They cannot be subtracted to infer an exact runtime without diagnostics. '
         'Validation and checkpoint I/O are excluded.',
         {'Whole GPT iteration': values(rows, 'iteration_wall_seconds_max_rank'),
          'Replay training and final KL': values(replay, 'chisel_replay_wall_seconds'),
          'Replica hashing': values(rows, 'chisel_replica_check_wall_seconds')})
    (output/'plot-data.json').write_text(json.dumps(record, indent=2)+'\n')
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    result=plot(args.input, args.output)
    print(json.dumps(dict(figures=list(result['figures']), source_sha256=result['source_sha256'])))
