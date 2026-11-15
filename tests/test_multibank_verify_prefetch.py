import pytest
from rewriter.multibank import MultiStateBankReader, MultiTeacherCacheReader
from rewriter.verify_prefetch import VerifyAhead, verify_state, verify_panel
import rewriter.verify_prefetch as module
from tests.test_multibank import banks


def test_verify_ahead_uses_seed_namespace_without_changing_local_clock(banks):
    reader = MultiStateBankReader(banks, verify_once=True, mmap_model=True)
    ahead = VerifyAhead(reader, depth=1)
    try:
        ahead.schedule([1, 6])
        ahead.wait(1)
        assert 1 in reader.readers[0]._verified
        ahead.wait(6)
        assert 1 in reader.readers[1]._verified
        assert ahead.hashed == {'states':2, 'panels':0}
        first, second = reader.read(1), reader.read(6)
        assert first.completed_steps == second.completed_steps == 1
        assert (first.state_id, second.state_id) == (1, 6)
        assert first.assert_verified() != second.assert_verified()
        assert not verify_state(reader, 6)
    finally:
        ahead.close()


def test_composite_panel_verification_routes_to_original_child(banks, monkeypatch):
    states = MultiStateBankReader(banks, verify_once=True)
    cache = object.__new__(MultiTeacherCacheReader)
    cache.states = states; cache.readers = {0:object(), 1:object()}
    calls = []
    def child(reader, state_id, role):
        calls.append((reader, state_id, role));return True
    monkeypatch.setattr(module, 'verify_panel', child)
    assert verify_panel(cache, 6, 'train')
    assert calls == [(cache.readers[1], 1, 'train')]
    with pytest.raises(IndexError): verify_panel(cache, 11, 'train')
