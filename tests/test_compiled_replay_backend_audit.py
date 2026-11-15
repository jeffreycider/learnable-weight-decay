"""Independent installed-backend/pending-resume audit; CPU AOT, not GPU proof."""
import copy
from functools import partial
from unittest.mock import patch

import pytest

from rewriter.compiled_thin_student import CompiledThinStudent
from tests import test_online_preservation as fixture
from tests.test_online_norm_lifecycle import assert_tree
from tests.test_transformer_online import _ordinary_step


@pytest.mark.parametrize('actual_actor', [False, True], ids=['factor_actor', 'transformer_actor'])
def test_configured_compiled_backend_pending_restart_matches_eager(tmp_path, actual_actor):
    original_settings = fixture.settings

    def create(compiled):
        def settings(directory, world=1):
            value = original_settings(directory, world)
            value['teacher_cache_device'] = 'device'
            if compiled:
                value['student_backend'] = 'compiled_student_v1_experimental'
            return value
        with patch.object(fixture, 'settings', settings):
            return fixture.make(tmp_path, actual_actor=actual_actor)

    def snapshot(model, optimizer, adapter):
        return copy.deepcopy(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                  adapter=adapter.state_dict()))

    def numerical_state(saved):
        # Backend identity is intentionally different; do not strip any tensor,
        # clock, dual, pending action or optimizer state from the comparison.
        result = copy.deepcopy(saved)
        def strip_backend(value):
            if isinstance(value, dict):
                value.pop('student_backend', None)
                for item in value.values():
                    strip_backend(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    strip_backend(item)
        strip_backend(result)
        return result

    def run(compiled):
        model, optimizer, adapter = create(compiled)
        restored = None
        try:
            for step in (2, 3):
                _ordinary_step(model, optimizer, adapter, step)
            saved = snapshot(model, optimizer, adapter)
            untouched = copy.deepcopy(saved)
            assert adapter.lifecycle.action is not None
            helper = getattr(adapter.replay, '_experimental_compiled_student', None)
            assert (helper is not None) == compiled
            if compiled:
                assert saved['adapter']['replay']['settings']['student_backend'] == 'compiled_student_v1_experimental'
            m2, o2, a2 = create(compiled)
            restored = a2
            m2.load_state_dict(saved['model'])
            o2.load_state_dict(copy.deepcopy(saved['optimizer']))
            a2.load_state_dict(copy.deepcopy(saved['adapter']))
            # Compiled executables must be rebuilt lazily, never serialized.
            assert not hasattr(a2.replay, '_experimental_compiled_student')
            assert a2.lifecycle.action is not None
            for m, o, a in ((model, optimizer, adapter), (m2, o2, a2)):
                _ordinary_step(m, o, a, 4)
                states = list(a.lifecycle.optimizer.state.values())
                assert states
                assert all(int(state['step']) == 5 for state in states)
                if actual_actor:
                    assert len(states) > 10  # Exercise the Transformer body VJP.
                assert a.lifecycle.meta_updates == 2
                assert a.replay.updates == a.replay.dual.updates == 3
                assert a.lifecycle.action is not None
                assert not a.collector.active and a.collector.ctx is None
            if compiled:
                assert a2.replay._experimental_compiled_student is not helper
            live, resumed = snapshot(model, optimizer, adapter), snapshot(m2, o2, a2)
            assert_tree(live, resumed)
            assert_tree(saved, untouched)
            return numerical_state(live)
        finally:
            adapter.close()
            if restored is not None:
                restored.close()

    # Exercise the real settings dispatch and helper lifecycle; only replace
    # the compiler backend, since this audit runs on CPU without Inductor CUDA.
    with patch('rewriter.compiled_replay.CompiledThinStudent',
               partial(CompiledThinStudent, compiler_backend='aot_eager')):
        eager, compiled = run(False), run(True)
    assert_tree(eager, compiled)
