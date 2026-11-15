"""Default identity, strict resume binding and local compilation boundary."""
import copy
from unittest.mock import patch
import pytest
import torch
from rewriter.online_preservation import validate_replay_maintenance
from rewriter.compiled_replay import compiled_replay_student
from tests.test_online_preservation import settings,make


def test_legacy_settings_are_not_normalized_or_migrated(tmp_path):
    original=settings(tmp_path)
    assert validate_replay_maintenance(original)==original
    assert 'student_backend' not in validate_replay_maintenance(original)
    explicit={**original,'student_backend':'compiled_student_v1_experimental','teacher_cache_device':'device'}
    assert validate_replay_maintenance(explicit)==explicit
    with pytest.raises(ValueError,match='student backend'):
        validate_replay_maintenance({**original,'student_backend':'compiled'})
    with pytest.raises(ValueError,match='resident device'):
        validate_replay_maintenance({**original,'student_backend':'compiled_student_v1_experimental'})


def test_backend_is_part_of_existing_strict_resume_identity(tmp_path):
    model,opt,adapter=make(tmp_path)
    saved=copy.deepcopy(adapter.replay.state_dict())
    assert 'student_backend' not in saved['settings']
    adapter.replay.settings['student_backend']='compiled_student_v1_experimental'
    assert adapter.replay.state_dict()['settings']['student_backend']=='compiled_student_v1_experimental'
    with pytest.raises(ValueError,match='resume configuration changed'):
        adapter.replay.load_state_dict(saved,adapter.lifecycle)
    adapter.close()


def test_helper_is_lazy_local_only_and_teacher_transient(tmp_path):
    model,opt,adapter=make(tmp_path);replay=adapter.replay
    with pytest.raises(ValueError,match='explicit experimental'):
        compiled_replay_student(replay,None,None,set(),backward=True)
    replay.settings.update(student_backend='compiled_student_v1_experimental',teacher_cache_device='device')
    wire=object();teachers=object();editable={'proj.weight'}
    with patch('rewriter.compiled_replay.CompiledThinStudent') as factory:
        factory.return_value.editable=frozenset(editable)
        factory.return_value.step.return_value=('loss','cotangents')
        assert replay._student(wire,teachers,editable,backward=True)==('loss','cotangents')
        replay._student(wire,teachers,editable,backward=False)
        factory.assert_called_once_with(replay.engine,editable,global_tokens=8,embedding_names=replay.embedding_names,compile_kl=False)
        assert factory.return_value.step.call_count==2
    assert not hasattr(replay,'teachers')
    adapter.close()


def test_actual_ordinary_pending_lifecycle_with_cpu_compiler(tmp_path):
    from rewriter.compiled_thin_student import CompiledThinStudent
    from tests.test_online_preservation import _ordinary_step
    def run(compiled):
        model,opt,adapter=make(tmp_path)
        if compiled:adapter.replay.settings.update(student_backend='compiled_student_v1_experimental',teacher_cache_device='device')
        rows=[]
        with patch('rewriter.compiled_replay.CompiledThinStudent',side_effect=lambda *a,**kw:CompiledThinStudent(*a,**kw,compiler_backend='aot_eager')):
            for step in (2,3,4):
                _ordinary_step(model,opt,adapter,step,0,1)
                rows.append(({n:p.detach().clone() for n,p in model.named_parameters()},adapter.metrics['chisel_replay_final_kl']))
        adapter.close();return rows
    for (want,wkl),(got,gkl) in zip(run(False),run(True)):
        assert abs(wkl-gkl)<1e-9
        for name in want:torch.testing.assert_close(got[name],want[name],rtol=2e-5,atol=2e-6)
