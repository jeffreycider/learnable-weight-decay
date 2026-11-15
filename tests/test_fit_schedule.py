import pytest
from rewriter.fit_schedule import FitSchedule


def test_plateau_trains_after_each_learning_rate_reduction():
    schedule=FitSchedule()
    first=schedule.observe(0.,0,1e-5)
    assert first['new_best'] and not first['stop']
    lr=first['lr'];rows=[]
    for epoch in range(5,45,5):
        row=schedule.observe(0.,epoch,lr);lr=row['lr'];rows.append(row)
    assert [5*(i+1) for i,row in enumerate(rows) if row['lr_reduced']]==[15,30]
    assert not any(row['stop'] for row in rows[:-1])
    assert rows[-1]['stop'] and rows[-1]['checks_since_reduction']==2
    assert rows[-1]['lr']==pytest.approx(2.5e-6)


def test_resume_preserves_counters_and_best_checkpoint_in_small_improvements():
    uninterrupted=FitSchedule();uninterrupted.observe(0.,0,1e-5)
    tiny=uninterrupted.observe(-.0004,5,1e-5)
    assert tiny['new_best'] and not tiny['significant_improvement']
    resumed=FitSchedule.from_state_dict(uninterrupted.state_dict())
    lr=1e-5
    for epoch,value in [(10,-.002),(15,-.0021),(20,-.0022),(25,-.0022),(30,-.0022)]:
        a=uninterrupted.observe(value,epoch,lr);b=resumed.observe(value,epoch,lr)
        assert a==b;lr=a['lr']
    assert uninterrupted.best_epoch==20


def test_nonfinite_validation_is_not_a_plateau():
    with pytest.raises(ValueError):FitSchedule().observe(float('nan'),5,1e-5)


def test_identity_stays_best_while_infeasible_trained_policy_improves():
    schedule=FitSchedule()
    initial=schedule.observe(0.,0,1e-5,track_progress=False)
    assert initial['new_best'] and initial['baseline_only']
    assert not initial['significant_improvement']
    for epoch,value in [(5,8.),(10,5.),(15,3.),(20,2.),(25,1.5),(30,1.2)]:
        result=schedule.observe(value,epoch,1e-5)
        assert result['significant_improvement']
        assert not result['new_best'] and not result['lr_reduced'] and not result['stop']
    assert schedule.best==0. and schedule.best_epoch==0
    feasible=schedule.observe(-.01,35,1e-5)
    assert feasible['new_best'] and feasible['significant_improvement']


def test_separate_baseline_progress_resumes_and_still_stops_on_actual_plateau():
    schedule=FitSchedule();schedule.observe(0.,0,1e-5,track_progress=False)
    resumed=FitSchedule.from_state_dict(schedule.state_dict());lr=1e-5
    for epoch in range(5,50,5):
        a=schedule.observe(2.,epoch,lr);b=resumed.observe(2.,epoch,lr)
        assert a==b;lr=a['lr']
        assert a['stop']==(epoch==45)
    assert schedule.best==0. and schedule.progress_best==2.


def test_historical_schedule_checkpoint_keeps_original_behavior():
    schedule=FitSchedule();schedule.observe(0.,0,1e-5)
    old=schedule.state_dict();old.pop('progress_started')
    restored=FitSchedule.from_state_dict(old)
    assert restored.observe(2.,5,1e-5)==schedule.observe(2.,5,1e-5)


def test_cannot_exclude_trained_policy_from_progress_tracking():
    with pytest.raises(ValueError,match='initial baseline'):
        FitSchedule().observe(2.,5,1e-5,track_progress=False)
