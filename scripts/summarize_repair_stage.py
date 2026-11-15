"""Read-only table of a preservation-repair stage's audits from latest_progress.json.

Reports, per audited update and population, the checkpoint-panel mean KL,
maximum, count at or below the trainer tolerance, 15-group shrink retained
relative to the initialization audit, CE change, and for comparison the
retention that PURE radial attenuation of the initial edit would give IF
checkpoint-mean KL scaled with the square of edit magnitude
(sqrt(mean_KL_now / mean_KL_init)); that column is an assumption-based
reference, not a measurement of the policy. No selection, no claims.
"""
import argparse,json
from pathlib import Path

ROLE_TEXT={'fixed_training':'fitting checkpoints, 128 seen seq each',
           'heldout_sequences':'same fitting checkpoints, 64 withheld seq',
           'heldout_states':'checkpoint-held-out, 64 withheld seq; development validation',
           'fresh_next_training':'fresh next-batch sequences (audit only)'}
ROLE_DEFAULT_N={'fixed_training':32,'heldout_sequences':32,'heldout_states':200}


def collect_audits(progress,watch_path,history_path=None):
    """Union of audit summaries: mirror (keeps only the latest per role), watcher
    log lines (history without state IDs) and an optional persisted history file
    that this script appends to, so earlier rows with state IDs are not lost."""
    rows={}
    def add(a,source):
        key=(a['update'],a['role'])
        have=rows.get(key)
        if have is None or (a.get('state_ids') and not have.get('state_ids')):
            rows[key]=dict(a,_source=source)
    if history_path and history_path.exists():
        for line in history_path.read_text().splitlines():
            try:add(json.loads(line),'history')
            except json.JSONDecodeError:continue
    if watch_path.exists():
        for line in watch_path.read_text().splitlines():
            try:entry=json.loads(line)
            except json.JSONDecodeError:continue
            for a in entry.get('audits') or []:add(a,'watch')
    for a in progress.get('audits',[]):add(a,'mirror')
    if history_path:
        with history_path.open('a') as out:
            for a in progress.get('audits',[]):
                if a.get('state_ids'):
                    key=(a['update'],a['role'])
                    if rows[key].get('_source')=='mirror':out.write(json.dumps({k:v for k,v in a.items()})+'\n')
    return sorted(rows.values(),key=lambda a:(a['update'],a['role']))


def population(a):
    ids=a.get('state_ids')
    n=len(ids) if ids else None
    label=f"{a['role']}_{n if n is not None else '?'} ({ROLE_TEXT.get(a['role'],a['role'])})"
    if n is None:label+=' [state IDs not mirrored; n assumed %s]'%ROLE_DEFAULT_N.get(a['role'],'?')
    if a['role']=='fixed_training' and n and n>32:label=f"fixed_training_{n} (ALL fitting checkpoints, 128 seen seq each)"
    return label,(n if n is not None else ROLE_DEFAULT_N.get(a['role']))


def matched(a,b):
    """Retention/radial references only between audits of the same checkpoint set."""
    ia,ib=a.get('state_ids'),b.get('state_ids')
    if ia and ib:return set(ia)==set(ib)
    # Without IDs on one side: the periodic 32/32/200 panels are the trainer's
    # deterministic predeclared sets, so equal role and equal (assumed) size matches;
    # an 800-state final audit never matches a 32-state initial audit.
    na=len(ia) if ia else ROLE_DEFAULT_N.get(a['role']);nb=len(ib) if ib else ROLE_DEFAULT_N.get(b['role'])
    return a['role']==b['role'] and na==nb and na is not None


def table(progress,tolerance=1e-6,watch_path=None,history_path=None):
    audits=collect_audits(progress,watch_path,history_path) if watch_path else sorted(progress.get('audits',[]),key=lambda a:(a['update'],a['role']))
    initial={a['role']:a for a in audits if a['update']==0}
    lines=['| update | population | n states | mean KL | max KL | states <= %g | shrink %% | retained vs init (same state set only) | radial-attenuation retention if KL~edit^2 | CE change |'%tolerance,'|---|---|---|---|---|---|---|---|---|---|']
    for a in audits:
        label,n=population(a)
        feasible=a.get('feasible_fraction')
        count=f"{round(feasible*n)}/{n}" if feasible is not None and n else 'n/a'
        base=initial.get(a['role'])
        ok=base is not None and matched(a,base) and base.get('mean_group_l2_percentage_reduction')
        retained=f"{a['mean_group_l2_percentage_reduction']/base['mean_group_l2_percentage_reduction']:.3f}" if ok else 'n/a (population differs)'
        radial=f"{(a['mean_kl']/base['mean_kl'])**.5:.3f}" if ok and base.get('mean_kl') and a['mean_kl']>=0 else 'n/a'
        lines.append(f"| {a['update']} | {label} | {n} | {a['mean_kl']:.3e} | {a.get('maximum_kl',float('nan')):.3e} | {count} | {a['mean_group_l2_percentage_reduction']:.4f} | {retained} | {radial} | {a.get('ce_change',float('nan')):+.2e} |")
    return '\n'.join(lines)+'\n'


def update_rows(watch_path):
    """Distinct per-update rows from the watcher log (pre-update batch measurements)."""
    rows={}
    if not watch_path.exists():return []
    for line in watch_path.read_text().splitlines():
        try:entry=json.loads(line)
        except json.JSONDecodeError:continue
        u=entry.get('update') or {}
        if u.get('update') is not None:rows[u['update']]=u
    return [rows[k] for k in sorted(rows)]


def update_table(rows,tolerance=1e-6):
    lines=['| update | batch mean KL (200 ckpt x 64 seen seq, before the step) | batch max KL | ckpt <= %g | 15-group shrink %% | wall s |'%tolerance,'|---|---|---|---|---|---|']
    for u in rows:
        feas=u.get('feasible_fraction');wall=u.get('wall_seconds')
        lines.append(f"| {u['update']} | {u['mean_kl']:.3e} | {u.get('maximum_kl') or float('nan'):.3e} | {round(feas*200) if feas is not None else 'n/a'}/200 | {u['mean_group_l2_percentage_reduction']:.4f} | {wall:.1f} |" if wall is not None else f"| {u['update']} | {u['mean_kl']:.3e} | n/a | n/a | {u['mean_group_l2_percentage_reduction']:.4f} | n/a |")
    return '\n'.join(lines)+'\n'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('progress',type=Path)
    parser.add_argument('--tolerance',type=float,default=1e-6)
    parser.add_argument('--history',type=Path,help='append-only local JSONL of audit rows with state IDs (the mirror keeps only the latest per role)')
    args=parser.parse_args(argv)
    progress=json.loads(args.progress.read_text())
    print(table(progress,args.tolerance,args.progress.with_name('watch.jsonl'),args.history))
    rows=update_rows(args.progress.with_name('watch.jsonl'))
    if rows:print(update_table(rows,args.tolerance))
    last=progress.get('last',{})
    print('last event:',last.get('event'),'update',last.get('update'),'elapsed',round(last.get('elapsed_seconds',0)),'s')
    if last.get('event')=='update':
        print('last update: mean KL %.3e max %.3e feasible %.3f shrink %.4f'%(last['mean_kl'],last.get('maximum_kl',float('nan')),last.get('feasible_fraction',float('nan')),last['mean_group_l2_percentage_reduction']))


if __name__=='__main__':main()
