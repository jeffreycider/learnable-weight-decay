"""Descriptive paired readout for the Chisel online 3250 pair and the 1250 replay pilot.

Reads the local mirrors only; prints validation trajectories with denominators, the
treatment-minus-identity differences at common steps, the end-of-schedule comparison, and the
pilot-vs-identity comparison. No intervals, no pass/fail wording, one run per arm.
"""
import json
import sys
from pathlib import Path

ROOT = Path('runs')
PAIR = ROOT / 'chisel-online-paired-3250-policy8-v1'
PILOT = ROOT / 'chisel-online-replay-pilot250-v1' / 'latest.json'
TOKENS = 10_485_760


def load(path):
    d = json.loads(Path(path).read_text())
    rows = d['rows'] if isinstance(d, dict) else d
    status = d.get('status') if isinstance(d, dict) else None
    vals = {r['step']: r['val_loss'] for r in rows if r.get('event') == 'validation'}
    train = {r['step']: r for r in rows if r.get('event') == 'train'}
    return status, vals, train


def main():
    out = {}
    s_id, v_id, t_id = load(PAIR / 'shadow' / 'latest.json')
    s_le, v_le, t_le = load(PAIR / 'learned' / 'latest.json')
    out['status'] = dict(identity=s_id, task_lag_one=s_le)
    common = sorted(set(v_id) & set(v_le))
    out['validation_tokens_per_readout'] = TOKENS
    out['paired_validation'] = [dict(step=s, identity=v_id[s], task_lag_one=v_le[s], task_minus_identity=v_le[s] - v_id[s]) for s in common]
    worse = sum(1 for s in common if v_le[s] > v_id[s])
    out['task_lag_one_worse_count'] = f'{worse}/{len(common)} common validations (step 1000 identical by construction)'
    end = [s for s in common if s >= 3100]
    out['end_of_schedule'] = [dict(step=s, identity=v_id[s], task_lag_one=v_le[s]) for s in end]
    if 3200 in v_id and 3250 in v_id:
        out['identity_change_3200_to_3250'] = v_id[3250] - v_id[3200]
    if 3200 in v_le and 3250 in v_le:
        out['task_lag_one_change_3200_to_3250'] = v_le[3250] - v_le[3200]
    if PILOT.exists():
        s_p, v_p, t_p = load(PILOT)
        out['pilot_status'] = s_p
        out['pilot_vs_identity'] = [dict(step=s, identity=v_id.get(s), pilot=v_p[s], pilot_minus_identity=None if s not in v_id else v_p[s] - v_id[s]) for s in sorted(v_p)]
        out['pilot_note'] = 'pilot step 1250 is a terminal identity step; the task-only run commits an edit at 1250, so only pilot-vs-identity is a clean comparison there; at 1125 both edited arms are comparable'
        if 1125 in v_p and 1125 in v_le:
            out['pilot_minus_task_lag_one_1125'] = v_p[1125] - v_le[1125]
    print(json.dumps(out, indent=1))
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
