"""Measure completed optimizer-update cadence from one uninterrupted log segment.

Elapsed gaps include audits, saves, logging and barriers; they are not an IO
profile. Never bridge a resumed clock, skipped update, or concatenated run.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def analyze_window(rows, start_update, end_update, *, online=False):
    if end_update <= start_update:
        raise ValueError('positive update window required')
    event, counter, timer = ('train','step','iteration_wall_seconds_max_rank') if online else ('update','update','wall_seconds')
    selected = [(i,r) for i,r in enumerate(rows) if r.get('event') == event
                and start_update <= r[counter] <= end_update]
    updates = [r for _,r in selected]
    if [r[counter] for r in updates] != list(range(start_update, end_update+1)):
        raise ValueError('unique consecutive update endpoints required')
    if any(r.get('event') == 'start' for r in rows[selected[0][0]+1:selected[-1][0]+1]):
        raise ValueError('window crosses a process restart')
    if online:
        required = dict(global_tokens_per_iteration=524288, world_size=8, microbatch_sequences=16,
                        ordinary_training_forwards_per_rank_this_iteration=4,
                        ordinary_training_backwards_per_rank_this_iteration=4)
        if any(any(r.get(k)!=v for k,v in required.items()) for r in updates):
            raise ValueError('online window requires the declared eight-rank524288-token workload')
        if len({r.get('training_procedure') for r in updates}) != 1 or not updates[0].get('training_procedure'):
            raise ValueError('online training procedure changes within window')
    intervals = []
    for before, after in zip(updates, updates[1:]):
        elapsed = after['elapsed_seconds']-before['elapsed_seconds']
        timed = after[timer]
        if not all(math.isfinite(v) for v in (elapsed, timed)) or timed <= 0 or elapsed < timed-1e-3:
            raise ValueError('clock reset, invalid timing or interval shorter than update')
        intervals.append(dict(after_update=before[counter], completed_update=after[counter],
            elapsed_seconds=elapsed, timed_update_seconds=timed,
            outside_update_timer_seconds=elapsed-timed))
    duration = sum(r['elapsed_seconds'] for r in intervals)
    training = sum(r['timed_update_seconds'] for r in intervals)
    return dict(training_procedure=updates[0]['training_procedure'] if online else 'offline_chisel',
        start_after_update=start_update, end_after_update=end_update,
        completed_updates=end_update-start_update, elapsed_seconds=duration,
        updates_per_elapsed_hour=3600*(end_update-start_update)/duration,
        timed_update_seconds=training, outside_update_timer_seconds=duration-training,
        outside_timer_seconds_per_update=(duration-training)/(end_update-start_update),
        scope='end-to-end between completed update logs, including intervening audits/saves/logging/barriers; excludes initial setup and work after final endpoint',
        limitation='rank0 log clock; outside-timer time is not isolated IO; no candidate production-throughput inference',
        intervals=intervals)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('log',type=Path);p.add_argument('--start-update',type=int,required=True)
    p.add_argument('--end-update',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--online',action='store_true',help='validate and analyze eight-rank524288-token online train records')
    a=p.parse_args();raw=a.log.read_bytes()
    result=analyze_window([json.loads(l) for l in raw.splitlines() if l.strip()],a.start_update,a.end_update,online=a.online)
    result.update(source=str(a.log),source_sha256=hashlib.sha256(raw).hexdigest())
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('completed_updates','elapsed_seconds','updates_per_elapsed_hour','outside_timer_seconds_per_update')},indent=2))
