from types import SimpleNamespace
import pytest
import rewriter.coordinate_verify_ahead as module
from scripts.train_coordinate_chisel import validate_recipe
from tests.test_coordinate_training_driver import recipe


class Schedule:
    def visit(self, batch, domain):
        assert domain == 'gpt'
        return SimpleNamespace(local_id=[3, 7, 3, 9][batch]), {}


def test_default_off_never_requires_reader_or_changes_schedule():
    with module.CoordinateVerifyAhead('off', None, None, 4) as ahead:
        assert ahead.wait(2, 3) == 0.
        assert ahead.worker is None


def test_resume_uses_remaining_actual_schedule_and_closes_on_failure(monkeypatch):
    events=[]
    class Worker:
        def __init__(self, reader, cache, **kw):
            assert (reader,cache,kw)==('bank','teacher',dict(roles=('train',),depth=1))
        def schedule(self, ids):events.append(('schedule',ids))
        def wait(self, uid):events.append(('wait',uid))
        def close(self):events.append(('close',))
    monkeypatch.setattr(module,'VerifyAhead',Worker)
    with pytest.raises(RuntimeError,match='injected'):
        with module.CoordinateVerifyAhead('depth1_v1',SimpleNamespace(reader='bank',cache='teacher'),Schedule(),4) as ahead:
            ahead.wait(2,3);ahead.wait(3,9)
            raise RuntimeError('injected')
    assert events==[('schedule',[3,9]),('wait',3),('wait',9),('close',)]


def test_recipe_allowlist_and_schedule_fail_closed():
    config=recipe();validate_recipe(config)
    config['verification_lookahead']='depth1_v1';validate_recipe(config)
    config['verification_lookahead']=True
    with pytest.raises(ValueError,match='verification lookahead'):validate_recipe(config)
    with module.CoordinateVerifyAhead('depth1_v1',None,Schedule(),4) as ahead:
        with pytest.raises(ValueError,match='schedule'):ahead.wait(2,7)
