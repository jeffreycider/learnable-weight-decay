import copy
import dataclasses

import pytest
import torch

from rewriter.ranking_dual import RankingDual
from rewriter.tokenwise_reference import TokenwiseDual, logit_terms_and_cotangent
from scripts.diagnose_cpu_norm_preservation import objective_config, make_diagnostic_dual, resume_contract
from scripts.evaluate_cpu_joint_unseen_tokens import restore_evaluation_dual


def fixture():
    return RankingDual([6,7],config=objective_config(),dual_lr=.01,rank_scale=1e-5,rank_dual_lr=1e-4)


def advance(dual,values=(2e-5,0.)):
    dual.step([(6,2e-10),(7,3e-10)],expected_ids=[6,7],ranking_measurements=list(zip((6,7),values)))


def test_checkpoint_specific_ascent_zero_constraint_and_loss_accounting():
    dual=fixture();base=dual.config
    assert dual.config_for_state(6)==base
    advance(dual)
    assert dual.updates==1 and dual.rank_multipliers=={6:2e-4,7:0.}
    assert dual.config_for_state(6).ranking_weight==pytest.approx(120.)
    assert dual.config_for_state(7)==base and dual.config_for_state(999)==base
    row=dual.describe_measurement(6,.0024)
    assert row['ranking_constraint_loss']==pytest.approx(2e-5)
    assert row['ranking_base_penalty']+row['ranking_dual_linear_term']==pytest.approx(.0024)
    advance(dual,(0.,0.))
    assert dual.rank_multipliers=={6:2e-4,7:0.}
    assert dual.config==base  # No mutation of KL config or default weight.


@pytest.mark.parametrize('rows', [[(6,-1.),(7,0.)],[(6,float('nan')),(7,0.)],[(6,0.),(6,0.)],[(6,0.)],[(6,0.),(999,0.)]])
def test_bad_or_heldout_measurements_do_not_mutate_any_dual_state(rows):
    dual=fixture();before=copy.deepcopy(dual.state_dict())
    with pytest.raises(ValueError):
        dual.step([(6,2e-10),(7,3e-10)],expected_ids=[6,7],ranking_measurements=rows)
    assert dual.state_dict()==before


def test_invalid_kl_measurement_does_not_partially_advance_ranking_state():
    dual=fixture();before=copy.deepcopy(dual.state_dict())
    with pytest.raises(FloatingPointError):
        dual.step([(6,float('nan')),(7,0.)],expected_ids=[6,7],ranking_measurements=[(6,1.),(7,2.)])
    assert dual.state_dict()==before


def test_exact_resume_and_readonly_evaluation_restore():
    dual=fixture();advance(dual);saved=copy.deepcopy(dual.state_dict())
    resumed=fixture();resumed.load_state_dict(saved)
    advance(dual,(3e-5,1e-6));advance(resumed,(3e-5,1e-6))
    assert dual.state_dict()==resumed.state_dict()
    metadata=dict(dual_lr=.01,ranking_constraint=dict(scale=1e-5,dual_lr=1e-4,tolerance=0.,measurement='mean_nonnegative_relative_gap_margin_loss'))
    restored=restore_evaluation_dual([6,7],objective_config(),metadata,saved)
    before=copy.deepcopy(restored.state_dict())
    restored.describe_measurement(6,.0024);restored.config_for_state(7)
    assert restored.state_dict()==before
    with pytest.raises(ValueError,match='requires RankingDual'):
        TokenwiseDual([6,7],config=objective_config(),dual_lr=.01).load_state_dict(saved)
    for key,value in [('scale',1e-4),('dual_lr',.1),('tolerance',1e-6)]:
        bad=copy.deepcopy(saved);bad['ranking_constraint'][key]=value
        with pytest.raises(ValueError,match='configuration'):fixture().load_state_dict(bad)


def test_added_ranking_cotangent_uses_same_logit_evaluation_and_no_teacher_gradient():
    torch.set_num_threads(1)
    dual=fixture();advance(dual)
    teacher=torch.tensor([[4.,3.,2.,0.]],dtype=torch.float64).log_softmax(-1)
    student=torch.tensor([[4.,1.9,2.1,0.]])
    # All KL terms are disabled by generous tolerances, leaving rank gradients.
    cfg=dataclasses.replace(dual.config,mean_tolerance=10.,tail_tolerance=10.)
    stronger=dataclasses.replace(cfg,ranking_weight=dual.config_for_state(6).ranking_weight)
    g,parts,_=logit_terms_and_cotangent(student,teacher,config=cfg,multiplier=0.,panel_tokens=1)
    g2,parts2,_=logit_terms_and_cotangent(student,teacher,config=stronger,multiplier=0.,panel_tokens=1)
    assert parts[2]>0
    torch.testing.assert_close(g2,g*1.2)
    assert parts2[2]==pytest.approx(float(parts[2])*1.2)
    assert teacher.grad is None


def test_disabled_is_exact_legacy_and_settings_are_immutable_on_resume():
    assert type(make_diagnostic_dual([6],objective_config())) is TokenwiseDual
    for kwargs in [dict(rank_scale=1e-5),dict(rank_dual_lr=1e-4),dict(rank_scale=0.,rank_dual_lr=1e-4)]:
        with pytest.raises(ValueError):make_diagnostic_dual([6],objective_config(),**kwargs)
    meta=dict(args={},objective={},dual_lr=.01,cases=[],state_weights=[.5,.5],private_gradient_normalization='undo')
    explicit=copy.deepcopy(meta);explicit['ranking_constraint']=None
    assert resume_contract(meta)==resume_contract(explicit)
    explicit['ranking_constraint']=dict(scale=1e-5,dual_lr=1e-4)
    assert resume_contract(meta)!=resume_contract(explicit)
