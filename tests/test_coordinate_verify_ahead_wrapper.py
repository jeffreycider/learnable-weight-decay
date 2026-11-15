from types import SimpleNamespace
import pytest
from scripts.qualify_coordinate_verify_ahead import upcoming_gpt_ids,verification_lookahead,trainer,FixedContexts


def test_resume_plan_uses_actual_remaining_sampler_clock():
    calls=[]
    def visit(schedule,batch,domain):
        calls.append((batch,domain));return SimpleNamespace(local_id=schedule[batch]),{'unchanged':True}
    assert upcoming_gpt_ids([10,20,30,10],visit,2,4)==[30,10]
    assert calls==[(2,'gpt'),(3,'gpt')]


def test_wrapper_restores_default_entry_points_after_failure():
    initialize=FixedContexts.__init__;visit=trainer.OwnerBalancedSchedule.visit
    with pytest.raises(RuntimeError,match='injected'):
        with verification_lookahead(False,9,{}):
            assert FixedContexts.__init__ is not initialize
            raise RuntimeError('injected')
    assert FixedContexts.__init__ is initialize and trainer.OwnerBalancedSchedule.visit is visit
