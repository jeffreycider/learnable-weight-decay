import hashlib
from pathlib import Path

import pytest
import torch

from scripts.migrate_feature_resume_checkpoint import migrate, SCRIPT, OLD_SCRIPT


def fixture(tmp_path):
    source = tmp_path/'source'; (source/'scripts').mkdir(parents=True)
    actual = Path(__file__).resolve().parents[1]/SCRIPT
    (source/SCRIPT).write_bytes(actual.read_bytes())
    (source/'other.py').write_text('immutable source')
    checkpoint = tmp_path/'original.pt'
    state = dict(controller={'p':torch.tensor([1.,2.])},optimizer={'moments':torch.tensor([.1,.2])},
                 primal_dual={'multipliers':{1:.5}},updates=8,next_epoch=2,next_batch=0,
                 metadata=dict(same_run_direction_warmup={'updates':1000},
                    implementation_sha256={SCRIPT:OLD_SCRIPT,'other.py':hashlib.sha256((source/'other.py').read_bytes()).hexdigest()}))
    torch.save(state,checkpoint)
    return source,checkpoint,state


def test_exact_migration_only_changes_metadata_and_preserves_original(tmp_path):
    source,checkpoint,state=fixture(tmp_path);before=checkpoint.read_bytes();output=tmp_path/'resume.pt'
    record=migrate(checkpoint,output,source);result=torch.load(output,weights_only=False)
    assert checkpoint.read_bytes()==before and not record['numerical_state_changed']
    assert result['updates']==state['updates'] and result['primal_dual']==state['primal_dual']
    for key in ('controller','optimizer'):
        for name,value in state[key].items():torch.testing.assert_close(result[key][name],value,rtol=0,atol=0)
    assert result['metadata']['same_run_direction_warmup']==state['metadata']['same_run_direction_warmup']
    with pytest.raises(ValueError,match='new output'):migrate(checkpoint,output,source)


def test_unapproved_source_change_is_rejected(tmp_path):
    source,checkpoint,_=fixture(tmp_path);(source/'other.py').write_text('different numerical algorithm')
    with pytest.raises(ValueError,match='unapproved source'):migrate(checkpoint,tmp_path/'resume.pt',source)
