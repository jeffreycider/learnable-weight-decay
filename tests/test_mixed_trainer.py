import copy
import pytest
import torch
from scripts.train_mixed_chisel import validate_recipe,validate_resume,domain_summary
from rewriter.tokenwise_reference import TokenwiseConfig,TokenwiseDual
from dataclasses import asdict


def recipe():
    cfg=asdict(TokenwiseConfig(1e-10,1e-10,1e-8,.01,ranking_k=100,ranking_tie_policy='exact_teacher_ties'))
    return dict(schema='chisel_mixed_pilot_v1',domain_weights={'gpt':.5,'cifar10':.5},architecture={'trajectory_mask_features':True},
        objectives={'gpt':cfg,'cifar10':cfg.copy()},image_microbatch=128,images_per_model=128,gpt_microbatch=16,
        sequences_per_checkpoint=64,head_lr=1e-7,representation_lr_multiplier=10,updates=64,readout_updates=[0,8,32,64])


def test_recipe_rejects_legacy_image_semantics_and_shape_changes():
    r=recipe();validate_recipe(r)
    for mutate in (lambda x:x['objectives']['cifar10'].update(ranking_tie_policy='legacy'),
                   lambda x:x.update(image_microbatch=256),lambda x:x['architecture'].update(trajectory_mask_features=False)):
        bad=copy.deepcopy(r);mutate(bad)
        with pytest.raises(ValueError):validate_recipe(bad)


def test_review_extension_preserves_source_objective_and_learning_rate():
    old=dict(recipe=recipe(),source_sha256={'file':'a'},world_size=2)
    new=copy.deepcopy(old);new['recipe']['updates']=128;new['recipe']['readout_updates']+=[128]
    validate_resume(old,new)
    for change in (lambda x:x['recipe'].update(head_lr=1e-8),lambda x:x['source_sha256'].update(file='b'),
                   lambda x:x['recipe'].update(readout_updates=[128])):
        bad=copy.deepcopy(new);change(bad)
        with pytest.raises(ValueError):validate_resume(old,bad)


def test_domain_weighted_shared_adam_and_visit_only_duals_match_dense_oracle():
    # Different context counts must not change equal domain contribution.
    x=torch.tensor([.2,-.3],requires_grad=True);y=x.detach().clone().requires_grad_(True)
    opt=torch.optim.Adam([x],lr=1e-4);oracle=torch.optim.Adam([y],lr=1e-4)
    domains={'gpt':[(1,torch.tensor([2.,1.])),(2,torch.tensor([-1.,3.]))],
             'cifar10':[(1001,torch.tensor([4.,-2.]))]}
    config=TokenwiseConfig(1e-10,1e-10,1e-8,.01,dual_update_mode='clipped_relative',ranking_k=100,ranking_tie_policy='exact_teacher_ties')
    duals={d:TokenwiseDual([i for i,_ in rows]+[9999],config=config,dual_lr=.01) for d,rows in domains.items()}
    for d,rows in domains.items():
        for i,target in rows:(.5/len(rows)*(x-target).square().mean()).backward()
    loss=sum(.5*torch.stack([(y-t).square().mean() for _,t in rows]).mean() for rows in domains.values());loss.backward()
    torch.testing.assert_close(x.grad,y.grad);opt.step();oracle.step();torch.testing.assert_close(x,y,atol=0,rtol=0)
    for d,rows in domains.items():
        duals[d].step([(i,2e-10) for i,_ in rows],expected_ids=[i for i,_ in rows])
        assert duals[d].multipliers[9999]==0 and duals[d].updates==1
        other=TokenwiseDual([i for i,_ in rows]+[9999],config=config,dual_lr=.01);other.load_state_dict(duals[d].state_dict())
        assert other.state_dict()==duals[d].state_dict()


def test_graph_denominator_counts_domain_weight_not_actual_gpt_visits():
    from scripts.train_mixed_chisel import graph_denominator
    assert graph_denominator(8,.5)==16
    assert 8/graph_denominator(8,.5)==.5
    with pytest.raises(ValueError):graph_denominator(8,.3)


def test_graph_then_cnn_backward_adds_without_losing_gpt_accumulation():
    p=torch.nn.Parameter(torch.tensor([1.,2.]));graph_buffer=torch.tensor([.2,.3])
    p.grad=graph_buffer
    (.5*p.square().mean()).backward()
    torch.testing.assert_close(p.grad,torch.tensor([.7,1.3]))
    assert p.grad.data_ptr()==graph_buffer.data_ptr()


def test_image_measurements_merge_domain_without_duplicate_keywords():
    from scripts.train_mixed_chisel import measurement_row
    assert measurement_row({'domain':'cifar10','mean_kl':0.},{'norm_term':1.},domain='cifar10',state_id=1001)['mean_kl']==0.
    with pytest.raises(ValueError):measurement_row({'domain':'gpt'},{},domain='cifar10')


def test_graph_identity_accepts_second_seed_but_rejects_domain_offset():
    from types import SimpleNamespace
    from scripts.train_mixed_chisel import require_graph_identity
    require_graph_identity(SimpleNamespace(state_id=1006),1006)
    with pytest.raises(ValueError):require_graph_identity(SimpleNamespace(state_id=6),106)


def test_image_teacher_role_depends_on_panel_not_backward():
    from scripts.train_mixed_chisel import image_teacher_role
    assert image_teacher_role('train')=='past_training'
    assert image_teacher_role('audit')=='withheld_training'


def test_resume_keeps_original_header_and_records_extension(tmp_path):
    import json
    from scripts.train_mixed_chisel import write_config_header
    old={'updates':64};new={'updates':200}
    write_config_header(tmp_path,old);before=(tmp_path/'config.json').read_bytes()
    write_config_header(tmp_path,new,previous=old,updates=64)
    assert (tmp_path/'config.json').read_bytes()==before
    assert json.loads((tmp_path/'config_resume_000064.json').read_text())==new
