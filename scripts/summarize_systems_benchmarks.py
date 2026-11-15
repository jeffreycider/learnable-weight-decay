"""Unified table for the Chisel offline systems benchmarks, labelled by scope.

Reads any mix of
  * scripts/profile_chisel_update.py summaries (single-rank or torchrun): per-arm
    seconds/state, max-rank wall, cold warmup, per-phase component totals,
    gradient / parameter / Adam / dual digests;
  * scripts/benchmark_chisel_systems.py outputs (trials.json / summary.json):
    per-variant full-update seconds with bit-exactness flags;
  * production metrics_compact.jsonl update rows for the reference wall.
Every printed line carries its scope so component time, local (one-rank /
few-state) contributions and full 200-checkpoint updates are never mixed, and
cold (warmup / first-touch) numbers are shown separately from warm medians.
"""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path

NESTED = {'student_gpt_forward_seconds', 'teacher_gpt_forward_seconds'}


def load_profile_summary(path):
    s = json.loads(Path(path).read_text())
    measured = [p for p in s['passes'] if not p['warmup']]
    states = s['states'] if isinstance(s['states'], list) else []
    full = s.get('world_size', 1) > 1 and len(s.get('batch_states') or []) == s.get('checkpoint_batch')
    scope = ('FULL %d-checkpoint update, %d ranks, max-rank wall' % (len(s['batch_states']), s['world_size']) if full
             else 'LOCAL %d-state contribution, 1 rank, no all-reduce' % len(states))
    rows = []
    for p in measured:
        comps = {k: v / max(p['states'], 1) for k, v in p['component_totals'].items()
                 if k.endswith('_seconds') and not k.endswith('_host_seconds') and not k.startswith('prefetch_worker_') and k not in NESTED}
        rows.append(dict(seconds=p.get('wall_seconds_max_over_ranks', p['wall_seconds']), per_state=p['seconds_per_state'],
                         same_update=p.get('update_bit_identical_to_first_pass_all_ranks', p['gradient_bit_identical_to_first_pass']),
                         components_per_state=comps, untimed_per_state=p['untimed_remainder_seconds'] / max(p['states'], 1)))
    return dict(kind='profile', label=s['label'], arm=s['arm'], scope=scope, cold_seconds=s['warmup_wall_seconds'],
                warm_median_seconds=statistics.median(r['seconds'] for r in rows), warm_per_state=statistics.median(r['per_state'] for r in rows),
                all_same_update=all(r['same_update'] for r in rows), digests=dict(gradient=s['gradient_sha256'], parameters=s.get('parameters_after_step_sha256'),
                adam=s.get('optimizer_state_sha256'), dual=s.get('dual_state_sha256')), rows=rows, profile=s.get('profile'))


def load_astra_trials(path):
    data = json.loads(Path(path).read_text())
    trials = data['rows'] if isinstance(data, dict) and 'rows' in data else data
    out = {}
    for t in trials:
        out.setdefault(t['variant'], []).append(t)
    results = []
    for variant, rows in out.items():
        warm = [r for r in rows if not r.get('warmup')]
        results.append(dict(kind='astra', label=variant, scope='FULL 200-checkpoint update (astra harness, max-rank wall)' if isinstance(data, dict) and data.get('checkpoint_batch') == 200 else 'astra harness trial',
                            cold_seconds=next((r['seconds'] for r in rows if r.get('warmup')), None),
                            warm_median_seconds=statistics.median(r['seconds'] for r in warm) if warm else None,
                            all_same_update=all(r.get('bit_exact_to_baseline', False) for r in rows), rows=rows))
    return results


def production_reference(path, last=20):
    ups = []
    for line in Path(path).open():
        try:
            x = json.loads(line)
        except json.JSONDecodeError:
            continue
        if x.get('event') == 'update':
            ups.append(x)
    tail = ups[-last:]
    return dict(kind='production', label=Path(path).parent.name, scope='PRODUCTION full 200-checkpoint update, 8 ranks (wall_seconds of last %d updates)' % len(tail),
                warm_median_seconds=statistics.median(u['wall_seconds'] for u in tail), cold_seconds=ups[0]['wall_seconds'] if ups else None,
                updates_per_hour=3600 / statistics.median(u['wall_seconds'] for u in tail))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', type=Path, nargs='*', default=[], help='profile_chisel_update summary.json files')
    p.add_argument('--astra', type=Path, nargs='*', default=[], help='benchmark_chisel_systems trials.json / summary.json files')
    p.add_argument('--production', type=Path, nargs='*', default=[], help='metrics_compact.jsonl of a live fit for the reference wall')
    p.add_argument('--json', type=Path, help='write the combined table as JSON')
    args = p.parse_args(argv)
    entries = [production_reference(x) for x in args.production]
    entries += [load_profile_summary(x) for x in args.profile]
    for x in args.astra:
        entries += load_astra_trials(x)
    baseline = next((e for e in entries if e['kind'] == 'profile' and e['arm'].get('timing') == 'sync' and not any(e['arm'].get(k) for k in ('deferred_kl', 'prefetch', 'prefetch_panel_only', 'constants_cache'))), None)
    print('%-34s %-62s %10s %10s %8s %s' % ('label', 'scope', 'cold s', 'warm s', 'ratio', 'same post-update bits'))
    for e in entries:
        ratio = ''
        if baseline and e is not baseline and e['kind'] == 'profile' and e['scope'] == baseline['scope'] and e['warm_median_seconds']:
            ratio = '%.3fx' % (baseline['warm_median_seconds'] / e['warm_median_seconds'])
            same = all(e['digests'][k] == baseline['digests'][k] for k in e['digests'] if e['digests'][k] and baseline['digests'][k])
            e['same_as_baseline_arm'] = same
        print('%-34s %-62s %10s %10s %8s %s' % (e['label'][:34], e['scope'][:62], '%.1f' % e['cold_seconds'] if e.get('cold_seconds') else '-',
              '%.2f' % e['warm_median_seconds'] if e.get('warm_median_seconds') else '-', ratio,
              e.get('same_as_baseline_arm', e.get('all_same_update', ''))))
    for e in entries:
        if e['kind'] == 'profile' and e['rows']:
            comps = e['rows'][0]['components_per_state']
            top = sorted(comps.items(), key=lambda kv: -kv[1])[:8]
            print('  components/state (%s): ' % e['label'] + ', '.join('%s %.3f' % (k.replace('_seconds', ''), v) for k, v in top) + ', untimed %.3f' % e['rows'][0]['untimed_per_state'])
            if e.get('profile'):
                print('  profiler: busy fraction %s over %s states, launches %s' % (e['profile'].get('device_busy_fraction'), e['profile'].get('states'), e['profile'].get('kernel_launches')))
    if args.json:
        args.json.write_text(json.dumps(entries, indent=2, default=str) + '\n')
    return entries


if __name__ == '__main__':
    main()
