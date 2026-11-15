"""Read-only progress table for the paired online continuation (identity vs task-lag-one).

Accepts the owner's live mirror format ``{status, rows, last_event}`` (rows are
trainer events; the arm name comes from ``status.arm`` or the file's parent
directory) and the older smoke format ``{arms: {name: [events]}}`` or
``status.summary.arms``. Identical replayed events are deduplicated; conflicting
repeats of the same step are reported, never hidden.

Per arm: step range, whole-iteration wall statistics over a declared post-warmup
window (the trainer's iteration wall spans ordinary training, optimizer, Chisel and
per-step diagnostics and EXCLUDES validation and checkpoint I/O), peak memory,
validations, committed edit norm first->last, action-head relative step, task
gradient norm/clip count, replica-checked rows versus the expected cadence
(completed steps 1001+64k; a check failure raises in the trainer, so there is no
failure flag to read), finiteness, and an ESTIMATED finish time from elapsed-second
differences over the window (includes validation/checkpoint time inside it).
Treatment-minus-identity validation at common steps is descriptive; one prefix
pair, no CI, no endpoint selection.
"""
import argparse,json,math,statistics,datetime
from pathlib import Path


def load_arms(paths):
    arms={}
    for p in paths:
        p=Path(p);d=json.loads(p.read_text())
        if 'rows' in d:
            name=(d.get('status') or {}).get('arm') or p.parent.name
            arms.setdefault(name,[]).extend(d['rows'])
        else:
            src=d.get('arms') or d.get('status',{}).get('summary',{}).get('arms')
            if src is None:raise ValueError(f'{p}: neither rows nor arms found')
            for name,value in src.items():
                events=value if isinstance(value,list) else (value.get('iterations',[])+value.get('validation',[]))
                arms.setdefault(name,[]).extend(events)
    return arms


TIMING_KEYS={'elapsed_seconds','iteration_wall_seconds_max_rank','chisel_replica_check_wall_seconds','peak_memory_bytes','compile_counters'}


def dedupe(events):
    """Drop byte-identical replays only. Any two rows for the same step that differ in a
    non-timing field (train loss, actor/edit telemetry, validation loss) are reported as
    conflicts; replayed rows that merely differ in wall times are not conflicts."""
    seen=set();out=[];conflicts=[]
    for e in events:
        key=json.dumps(e,sort_keys=True)
        if key in seen:continue
        seen.add(key);out.append(e)
    by_key={}
    for e in out:
        if e.get('event') not in ('train','validation'):continue
        k=(e['event'],e['step']);core={a:b for a,b in e.items() if a not in TIMING_KEYS}
        if k in by_key and by_key[k]!=core:conflicts.append(k)
        by_key.setdefault(k,core)
    return out,sorted(set(conflicts))


def last_attempt(train):
    """Rows after the last decrease of elapsed_seconds (a restart resets the clock)."""
    start=0
    for i in range(1,len(train)):
        if train[i].get('elapsed_seconds',0)<train[i-1].get('elapsed_seconds',0):start=i
    return train[start:],start>0


def arm_summary(events,window,endpoint,cadence_start,cadence):
    events,conflicts=dedupe(events)
    train=sorted([e for e in events if e.get('event')=='train'],key=lambda e:e['step'])
    val={e['step']:e['val_loss'] for e in events if e.get('event')=='validation'}
    attempt,resumed=last_attempt(train)
    tail=attempt[-window:] if len(attempt)>window else attempt[1:] if len(attempt)>1 else attempt
    walls=[e['iteration_wall_seconds_max_rank'] for e in tail if 'iteration_wall_seconds_max_rank' in e]
    edits=[(e['step'],e['committed_edit_norm']) for e in train if e.get('committed_edit_norm')]
    heads=[(e['step'],(e.get('chisel_actor_relative_step_norm_by_family') or {}).get('action_heads')) for e in train if e.get('chisel_actor_relative_step_norm_by_family')]
    tg=[e['task_gradient_norm'] for e in train if 'task_gradient_norm' in e]
    clipped=sum(1 for e in train if e.get('task_gradient_clipped'))
    checked=sorted(e['step'] for e in train if e.get('chisel_pre_edit_replica_count')==8)
    last=train[-1]['step'] if train else None
    expected=[s for s in range(cadence_start,(last or 0)+1,cadence)] if last else []
    missing=[s for s in expected if s not in checked]
    nonfinite=[e['step'] for e in train if not math.isfinite(e.get('train_loss',0.)) or not math.isfinite(e.get('committed_edit_norm',0.) or 0.)]
    projection=None
    if len(tail)>=2 and endpoint and last and last<endpoint:
        dt=(tail[-1]['elapsed_seconds']-tail[0]['elapsed_seconds']);ds=tail[-1]['step']-tail[0]['step']
        if ds>0 and dt>0:
            per=dt/ds;remaining=(endpoint-last)*per
            projection=dict(window_steps=ds,window_seconds=dt,seconds_per_step_incl_ancillary=per,remaining_steps=endpoint-last,
                            estimated_remaining_hours=remaining/3600,note='estimate from same-attempt elapsed differences; includes validation/checkpoint time that fell inside the window')
    return dict(steps=len(train),first_step=train[0]['step'] if train else None,last_step=last,conflicting_repeats=conflicts,resumed_attempt=resumed,
                window_steps=len(tail),wall_median=statistics.median(walls) if walls else None,wall_mean=statistics.mean(walls) if walls else None,
                wall_max=max(walls) if walls else None,validation=val,edit_norm_first=edits[0] if edits else None,edit_norm_last=edits[-1] if edits else None,
                heads_relative_step_last=heads[-1] if heads else None,task_gradient_norm_median=statistics.median(tg) if tg else None,task_gradient_clipped_rows=clipped,
                replica_checked_rows=len(checked),replica_expected_rows=len(expected),replica_missing_steps=missing[:10],nonfinite_steps=nonfinite[:10],
                peak_memory_gb=max((e.get('peak_memory_bytes',0) for e in train),default=0)/1e9,procedure=train[-1].get('training_procedure') if train else None,
                meta_updates=train[-1].get('meta_updates') if train else None,projection=projection)


def table(arms,identity='shadow',treatment='learned',window=200,endpoint=3250,cadence_start=1001,cadence=64):
    s={name:arm_summary(ev,window,endpoint,cadence_start,cadence) for name,ev in arms.items()}
    lines=['| arm | procedure | steps (first-last) | window | iter wall median/mean/max (s; excl. val+ckpt) | peak GB | edit norm first -> last | heads rel step (last) | task grad median / clipped rows | replica-checked / expected | nonfinite | validations |','|---|---|---|---|---|---|---|---|---|---|---|---|']
    for name,a in s.items():
        v=', '.join(f"{k}: {x:.6f}" for k,x in sorted(a['validation'].items()))
        w='n/a' if a['wall_median'] is None else f"{a['wall_median']:.2f} / {a['wall_mean']:.2f} / {a['wall_max']:.2f}"
        e='none' if not a['edit_norm_first'] else f"{a['edit_norm_first'][1]:.3f} (s{a['edit_norm_first'][0]}) -> {a['edit_norm_last'][1]:.3f} (s{a['edit_norm_last'][0]})"
        h='n/a' if not a['heads_relative_step_last'] or a['heads_relative_step_last'][1] is None else f"{a['heads_relative_step_last'][1]*100:.2f} % (s{a['heads_relative_step_last'][0]})"
        g='n/a' if a['task_gradient_norm_median'] is None else f"{a['task_gradient_norm_median']:.3f} / {a['task_gradient_clipped_rows']}"
        lines.append(f"| {name} | {a['procedure']} | {a['steps']} ({a['first_step']}-{a['last_step']}) | last {a['window_steps']} | {w} | {a['peak_memory_gb']:.1f} | {e} | {h} | {g} | {a['replica_checked_rows']} / {a['replica_expected_rows']} | {len(a['nonfinite_steps'])} | {v} |")
        if a['conflicting_repeats']:lines.append(f"  CONFLICTING repeated (event, step) rows in {name}: {a['conflicting_repeats'][:10]}")
        if a['resumed_attempt']:lines.append(f"  {name}: elapsed clock reset detected; window and ETA use the last attempt only")
        if a['replica_missing_steps']:lines.append(f"  replica check missing at expected steps in {name}: {a['replica_missing_steps']}")
        if a['projection']:
            pj=a['projection'];lines.append(f"  {name} finish ESTIMATE: {pj['seconds_per_step_incl_ancillary']:.2f} s/step over the last {pj['window_steps']} steps incl. ancillary -> {pj['remaining_steps']} steps ~ {pj['estimated_remaining_hours']:.2f} h remaining ({pj['note']})")
    out='\n'.join(lines)+'\n'
    if identity in s and treatment in s:
        common=sorted(set(s[identity]['validation'])&set(s[treatment]['validation']))
        if common:
            out+='\nTreatment minus identity validation loss at common steps (nats/token over the trainer validation tokens; negative favors task-lag-one; descriptive; one prefix pair, no CI; endpoint 3250 predeclared):\n'
            for step in common:out+=f"  step {step}: {s[treatment]['validation'][step]-s[identity]['validation'][step]:+.3e}\n"
    return out


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mirrors',nargs='+',type=Path);p.add_argument('--identity',default='shadow');p.add_argument('--treatment',default='learned')
    p.add_argument('--window',type=int,default=200);p.add_argument('--endpoint',type=int,default=3250)
    a=p.parse_args(argv);print(datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC'));print(table(load_arms(a.mirrors),a.identity,a.treatment,a.window,a.endpoint))


if __name__=='__main__':main()
