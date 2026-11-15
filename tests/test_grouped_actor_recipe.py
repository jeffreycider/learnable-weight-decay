import copy
import pytest
import torch
from tests.test_coordinate_training_driver import recipe
from scripts.train_coordinate_chisel import validate_recipe,build_actor,build_shared_adam,configure_shared_trainability
from rewriter.adaptive_coordinate_chisel import AdaptiveCoordinateChisel
from rewriter.grouped_coordinate_chisel import GroupedCoordinateChisel


def grouped_recipe():
    r=recipe();r.update(action_family='adaptive_ucv_v1',student_backend='dense')
    r['architecture'].update(output_head_sharing='functional_group',factor_precision='fp64',residual_precision='fp32')
    return r


def test_opt_in_factory_heads_optimizer_policy_and_strict_checkpoint():
    r=grouped_recipe();validate_recipe(r)
    actor=build_actor(r);assert isinstance(actor,GroupedCoordinateChisel)
    optimizer=build_shared_adam(actor,r['shared_optimizer'])
    headids={id(p) for h in actor.output_heads for p in h.parameters()}
    assert {id(p) for p in optimizer.param_groups[0]['params']}==headids
    assert not headids & {id(p) for p in optimizer.param_groups[1]['params']}
    configure_shared_trainability(actor,'heads_only')
    assert all(p.requires_grad==(id(p) in headids) for p in actor.parameters())
    restored=build_actor(r);restored.load_state_dict(copy.deepcopy(actor.state_dict()))
    old=copy.deepcopy(r);old['architecture'].pop('output_head_sharing')
    legacy=build_actor(old);assert type(legacy) is AdaptiveCoordinateChisel
    with pytest.raises(ValueError,match='architecture'):legacy.load_state_dict(actor.state_dict())


def test_default_factory_keeps_original_draws_and_configuration():
    r=grouped_recipe();r['architecture'].pop('output_head_sharing')
    torch.manual_seed(739);actual=build_actor(r)
    torch.manual_seed(739);expected=AdaptiveCoordinateChisel(**r['architecture'])
    assert actual.configuration()==expected.configuration()
    for name,p in actual.named_parameters():assert torch.equal(p,dict(expected.named_parameters())[name])


def test_unqualified_grouped_graphs_and_wrong_family_fail():
    r=grouped_recipe();r['actor_backend']='adaptive_actor_graph'
    with pytest.raises(ValueError,match='graph qualification'):validate_recipe(r)
    r=grouped_recipe();r['action_family']='fixed_anchored_v1'
    with pytest.raises(ValueError,match='requires adaptive'):validate_recipe(r)
    r=grouped_recipe();r['architecture']['output_head_sharing']='per_weight'
    with pytest.raises(ValueError,match='head sharing'):validate_recipe(r)
