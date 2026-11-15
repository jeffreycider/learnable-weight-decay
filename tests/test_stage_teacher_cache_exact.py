import hashlib
import json
from pathlib import Path
import pytest

from scripts.stage_teacher_cache_exact import make_plan,stage_exact,panel_selection,digest


def cache(tmp_path):
    source=tmp_path/'source';source.mkdir();panels=[]
    for state in range(1,6):
        for role in ('train','audit'):
            name=f'state_{state:04d}_{role}.pt';data=(name*9).encode();(source/name).write_bytes(data)
            row=dict(state_id=state,role=role,file=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),
                     state_sha256=str(state),input_sha256='input',feature_sha256='features',sequence_ids=[state])
            (source/Path(name).with_suffix('.json')).write_text(json.dumps(row));panels.append(row)
    (source/'manifest.json').write_text(json.dumps(dict(complete=True,panels=panels)))
    return source


def test_exact_selection_atomic_copy_remote_fallback_and_resume(tmp_path):
    source=cache(tmp_path);dest=tmp_path/'local'
    metadata=dict(state_bank_identity=dict(selected_states=5),heldout_state_ids=[5],heldout_sequence_state_ids=[2])
    selection=panel_selection(metadata,dict(split=dict(train=[1,2,3,4],audit=[5])))
    assert len(selection)==6
    plan=make_plan(source,selection)
    result=stage_exact(source,dest,plan,max_stage_bytes=plan['selected_bytes'],reserve_bytes=0)
    assert result['selected_count']==6 and result['remote_fallback_count']==4
    assert (dest/'manifest.json').read_bytes()==(source/'manifest.json').read_bytes()
    for row in json.loads((source/'manifest.json').read_text())['panels']:
        target=dest/row['file']
        assert target.read_bytes()==(source/row['file']).read_bytes()
        assert target.is_symlink()==((row['state_id'],row['role']) not in selection)
        assert target.with_suffix('.json').read_bytes()==(source/row['file']).with_suffix('.json').read_bytes()
    again=stage_exact(source,dest,plan,max_stage_bytes=plan['selected_bytes'],reserve_bytes=0)
    assert again['copied_bytes']==0


def test_corrupt_source_never_publishes_local_panel_or_complete_marker(tmp_path):
    source=cache(tmp_path);dest=tmp_path/'local';plan=make_plan(source,[(1,'train')])
    (source/'state_0001_train.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='verification'):
        stage_exact(source,dest,plan,max_stage_bytes=plan['selected_bytes'],reserve_bytes=0)
    assert not (dest/'stage_complete.json').exists()
    assert (dest/'state_0001_train.pt').is_symlink()
    assert not list(dest.glob('*.stage-partial'))


def test_byte_limit_and_manifest_binding_are_enforced(tmp_path):
    source=cache(tmp_path);plan=make_plan(source,[(1,'train')]);dest=tmp_path/'local'
    with pytest.raises(ValueError,match='limit'):
        stage_exact(source,dest,plan,max_stage_bytes=0,reserve_bytes=0)
    changed=dict(plan,manifest_sha256='bad')
    with pytest.raises(ValueError,match='changed'):
        stage_exact(source,dest,changed,max_stage_bytes=plan['selected_bytes'],reserve_bytes=0)
