import pytest
from scripts.analyze_chisel_cadence import analyze_window


def rows():
    # Initial prefix offset is deliberately large. A 30s audit lies between
    # update1 and update2; update3 adds 2s of saves/logging beyond its 10s step.
    return [dict(event='update',update=1,elapsed_seconds=1000.,wall_seconds=10.),
            dict(event='audit_begin',update=1,elapsed_seconds=1000.1),
            dict(event='update',update=2,elapsed_seconds=1040.,wall_seconds=10.),
            dict(event='update',update=3,elapsed_seconds=1052.,wall_seconds=10.)]


def test_elapsed_cadence_includes_between_update_work_and_excludes_prefix():
    r=analyze_window(rows(),1,3)
    assert r['completed_updates']==2 and r['elapsed_seconds']==52.
    assert r['timed_update_seconds']==20. and r['outside_update_timer_seconds']==32.
    assert r['updates_per_elapsed_hour']==pytest.approx(7200/52)


@pytest.mark.parametrize('defect',['duplicate','missing','reset','nan','restart_event'])
def test_refuses_incomplete_or_resumed_clock_windows(defect):
    r=rows()
    if defect=='duplicate':r.append(r[-1].copy())
    elif defect=='missing':r.pop(2)
    elif defect=='reset':r[-1]['elapsed_seconds']=12.
    elif defect=='restart_event':r.insert(2,dict(event='start',update=1,elapsed_seconds=1010.))
    else:r[-1]['wall_seconds']=float('nan')
    with pytest.raises(ValueError):analyze_window(r,1,3)


def test_online_cadence_requires_matching_real_token_and_pass_counts():
    online=[]
    for r in rows():
        if r['event']!='update':continue
        online.append(dict(event='train',step=r['update'],elapsed_seconds=r['elapsed_seconds'],
            iteration_wall_seconds_max_rank=r['wall_seconds'],training_procedure='fp32_chisel_task_lag_one',
            global_tokens_per_iteration=524288,world_size=8,microbatch_sequences=16,
            ordinary_training_forwards_per_rank_this_iteration=4,
            ordinary_training_backwards_per_rank_this_iteration=4))
    result=analyze_window(online,1,3,online=True)
    assert result['elapsed_seconds']==52. and result['timed_update_seconds']==20.
    online[-1]['ordinary_training_backwards_per_rank_this_iteration']=3
    with pytest.raises(ValueError,match='workload'):analyze_window(online,1,3,online=True)
    online[-1]['ordinary_training_backwards_per_rank_this_iteration']=4
    online[-1]['training_procedure']='ordinary_fp32_gpt'
    with pytest.raises(ValueError,match='procedure'):analyze_window(online,1,3,online=True)
