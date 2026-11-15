import pytest
from scripts.prepare_cpu_teacher_state import transplant_pattern,validate_train_panel


def test_pattern_transplant_copies_choices_not_old_claims():
    old={'rows':[dict(rank_limit=8,inverse_power=2,selected=[dict(channel=3,exponent=1)],state_id=6,state_sha256='old',finite_logits_qualified=True,reference_group_squared_norms={'x':7},mean_group_l2_percentage_reduction=1)]}
    out=transplant_pattern(old,7,'new')['rows'][0]
    assert out['state_id']==7 and out['state_sha256']=='new' and not out['finite_logits_qualified']
    assert 'reference_group_squared_norms' not in out and 'mean_group_l2_percentage_reduction' not in out
    out['selected'][0]['channel']=4
    assert old['rows'][0]['selected'][0]['channel']==3


def test_only_matching_train_panel_accepted():
    meta=dict(sha256='panel',state_sha256='state',sequence_ids=[1])
    panel={'metadata':dict(role='train',state_sha256='state',sequence_ids=[1])}
    validate_train_panel(panel,meta,'state','panel')
    panel['metadata']['role']='validation'
    with pytest.raises(ValueError,match='TRAIN'):validate_train_panel(panel,meta,'state','panel')
    with pytest.raises(ValueError,match='hash'):validate_train_panel(panel,meta,'wrong','panel')
