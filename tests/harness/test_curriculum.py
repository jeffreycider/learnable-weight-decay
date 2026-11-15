import pytest
from chisel.curriculum import interpolate, schedule
from chisel.harness import RunState
from chisel.preservation import Dual


def test_geometric_schedule_and_resume_are_step_determined():
    knots = [dict(step=0, preservation_weight=.01, kl_tolerance=1e-6, private_lr=.003),
             dict(step=100, preservation_weight=1., kl_tolerance=1e-12, private_lr=.0003)]
    middle = interpolate(knots, 50)
    assert middle['preservation_weight'] == pytest.approx(.1)
    assert middle['kl_tolerance'] == pytest.approx(1e-9)
    state = RunState(step=50, extras={'dual': Dual(1., 2., 3., 4.)})
    schedule(knots)({}, state)
    assert state.extras['offline_curriculum'] == middle
    assert state.extras['dual'].tolerance == pytest.approx(1e-9)
    assert interpolate(knots, 200)['private_lr'] == pytest.approx(.0003)


def test_schedule_rejects_missing_or_invalid_knots():
    for knots in [[], [dict(step=1, x=2)], [dict(step=0, x=0)],
                  [dict(step=0, x=2), dict(step=0, x=3)],
                  [dict(step=0, x=2), dict(step=2, y=3)]]:
        with pytest.raises(ValueError):
            interpolate(knots, 0)
    with pytest.raises(ValueError):
        schedule([dict(step=0, arbitrary=1.)])
