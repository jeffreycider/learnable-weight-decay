import torch
from scripts.measure_coordinate_gradient_balance import component_stats,families


def test_component_subtraction_cosine_and_combined_clip_are_not_independent_clips():
    norm={'a':torch.tensor([3.,0.])};total={'a':torch.tensor([3.,4.])}
    s=component_stats(norm,total,['a'],.2)
    assert s['norm_reward_gradient_norm']==3 and s['preservation_gradient_norm']==4
    assert s['cosine']==0 and s['total_gradient_norm']==5
    assert abs(s['norm_reward_after_combined_clip']-.6)<1e-15
    assert s['preservation_after_combined_clip']==.8


def test_family_mapping_keeps_heads_and_representation_clipping_distinct():
    n={'matrix_output.weight':torch.ones(2),'blocks.0.weight':torch.ones(2),'matrix_input.weight':torch.ones(2)}
    t={k:2*v for k,v in n.items()};s=families(n,t,'shared',{'action_heads':.25,'representation':.5})
    assert s['heads']['combined_clip_factor']==.25 and s['body']['combined_clip_factor']==.5
    assert s['all']['parameter_elements']==6 and s['body']['cosine']>.999999


def test_continuation_metadata_keeps_original_protocol_and_data_identity():
    from scripts.measure_coordinate_gradient_balance import original_metadata
    parent={'protocol':{'architecture':'fixed'},'base_recipe_sha256':'bound'}
    assert original_metadata({'recipe':{},'parent_metadata':parent}) is parent


def test_three_component_decomposition_retains_antagonistic_directions():
    from scripts.analyze_coordinate_gradient_components import statistics
    n={'a':torch.tensor([2.,0.])};z={'a':torch.tensor([1.,3.])};t={'a':torch.tensor([1.,7.])}
    s=statistics(n,z,t,['a'],.1)
    assert s['norms']['ranking']==4 and s['norms']['norm']==2
    assert s['ranking_over_norm']==2 and s['reconstruction_max_abs']==0
    assert s['cosines']['norm_vs_kl_tail']<0
