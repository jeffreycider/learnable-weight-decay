import copy
import pytest
from rewriter.primal_dual import PrimalDual
from scripts.evaluate_chisel_fresh import checkpoint_objective


def test_historical_readout_recovers_original_edit_penalty():
    saved={'metadata':{'options':{'kl_budget':.1}}}
    stats=dict(norm_term=-2.,realized_edit_norm=3.,hard_radius=10.)
    result=checkpoint_objective(saved,5,.2,stats)
    assert result['trust_term']==pytest.approx(.0009)
    assert result['objective']==pytest.approx(-.9991)
    assert result['edit_size_penalty_enabled']
    saved['metadata']['objective_policy']={'kind':'legacy-deadband','edit_size_penalty':False}
    result=checkpoint_objective(saved,5,.2,stats)
    assert result['trust_term']==0 and result['objective']==-1


def test_primal_dual_fresh_readout_reports_feasibility_without_dual_mutation():
    dual=PrimalDual([1,2],tolerance=.01,rho=.2,dual_lr=.03)
    dual.step([(1,.1),(2,.2)],expected_ids=[1,2])
    saved={'metadata':{'options':dict(objective='primal-dual',kl_tolerance=.01,augmented_rho=.2,dual_lr=.03)},'primal_dual':dual.state_dict()}
    before=copy.deepcopy(saved)
    result=checkpoint_objective(saved,5,.02,dict(norm_term=-2.))
    assert result['dual_multiplier']==0 and result['constraint_violation']==1
    assert not result['feasible'] and result['kl_penalty']==pytest.approx(.1)
    assert result['objective']==pytest.approx(-1.9) and result['trust_term']==0
    assert not result['objective_used_for_selection'] and saved==before
    with pytest.raises(ValueError,match='held-out'):checkpoint_objective(saved,1,.02,dict(norm_term=-2.))
    del saved['primal_dual']
    with pytest.raises(ValueError,match='lacks'):checkpoint_objective(saved,5,.02,dict(norm_term=-2.))


def test_unknown_objective_fails_closed():
    with pytest.raises(ValueError,match='unknown'):
        checkpoint_objective({'metadata':{'options':{'objective':'unqualified'}}},5,0,dict(norm_term=0))


def test_preservation_repair_readout_does_not_reintroduce_shrinkage_reward():
    dual=PrimalDual([1],tolerance=.01,rho=.2,dual_lr=.03)
    saved={'metadata':{'options':dict(objective='primal-dual',kl_tolerance=.01,augmented_rho=.2,dual_lr=.03,shrinkage_coefficient=0.)},'primal_dual':dual.state_dict()}
    for stats in [dict(norm_term=-2.),dict(norm_term=0.,unweighted_norm_term=-2.)]:
        result=checkpoint_objective(saved,5,.02,stats)
        assert result['norm_term']==0 and result['unweighted_norm_term']==-2.
        assert result['objective']==pytest.approx(.1)
