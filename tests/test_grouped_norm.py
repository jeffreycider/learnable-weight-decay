import pytest
import torch

from baseline.model import GPT
from rewriter.edit_scope import editable_names,readonly_hashes,verify_readonly
from rewriter.grouped_norm import GROUP_NAMES,GroupedNormObjective,functional_groups
from rewriter.online import OnlineRewriter


def test_pr323_fifteen_roles_aggregate_layers_and_restore_embedding_head_biases():
    model=GPT(vocab_size=16,num_layers=2,model_dim=128)
    parameters=dict(model.named_parameters())
    parameters['standalone_scalar']=torch.tensor(2.)
    parameters['learned_scale']=torch.tensor([3.])
    parameters['scalar_matrix']=torch.ones(1,1)
    parameters['norm.bias']=torch.ones(3)
    parameters['blocks.0.norm1.weight']=torch.ones(3,3)
    groups=functional_groups(parameters)
    assert tuple(groups)==GROUP_NAMES and len(groups)==15
    assert groups['embedding.weight']==('embed.weight',)
    assert groups['head.weight']==('proj.weight',) and groups['head.bias']==('proj.bias',)
    assert groups['attention.q.weight']==('blocks.0.attn.q.weight','blocks.1.attn.q.weight')
    assert groups['mlp.in.bias']==('blocks.0.mlp.fc.bias','blocks.1.mlp.fc.bias')
    selected=editable_names(parameters,'grouped')
    assert selected=={n for names in groups.values() for n in names}
    assert all(not n.endswith('gains') for n in selected)
    assert 'standalone_scalar' not in selected and 'learned_scale' not in selected
    assert 'scalar_matrix' not in selected and 'norm.bias' not in selected
    assert 'blocks.0.norm1.weight' not in selected
    assert {'embed.weight','proj.weight','proj.bias'}.issubset(selected)
    hashes=readonly_hashes(parameters,selected)
    assert 'standalone_scalar' in hashes
    assert verify_readonly(parameters,selected,hashes)['unchanged']


def test_groups_have_equal_percentage_weight_despite_parameter_size_or_scale():
    base={'large.weight':torch.full((100,100),100.,dtype=torch.float64),
          'small.bias':torch.tensor([3.,4.],dtype=torch.float64),
          'norm.gains':torch.full((100,),1e6,dtype=torch.float64)}
    objective=GroupedNormObjective(base,{'large':('large.weight',),'small':('small.bias',)})
    updated=dict(base,**{'small.bias':base['small.bias']*.9})
    result=objective.evaluate(updated)
    torch.testing.assert_close(result.mean_relative_change,torch.tensor(-.05,dtype=torch.float64))
    assert result.metrics()['mean_group_l2_percentage_reduction']==pytest.approx(5.)
    # Each group contributes 10%, hence the requested loss is exactly -10 percent.
    both={n:v*.9 if n!='norm.gains' else v for n,v in base.items()}
    result=objective.evaluate(both)
    assert 100*float(result.mean_relative_change)==pytest.approx(-10.)
    assert float(result.squared_norm_change/sum(objective.reference_squared.values()))==pytest.approx(-.19)


def test_group_reference_is_joint_l2_across_layers_not_average_tensor_percentages():
    base={'blocks.0.attn.q.bias':torch.tensor([3.,0.],dtype=torch.float64),
          'blocks.1.attn.q.bias':torch.tensor([4.,0.],dtype=torch.float64)}
    objective=GroupedNormObjective(base,{'q_bias':tuple(base)})
    assert objective.reference_norms['q_bias']==5
    updated=dict(base,**{'blocks.0.attn.q.bias':torch.tensor([0.,0.],dtype=torch.float64)})
    assert float(objective.evaluate(updated).mean_relative_change)==pytest.approx(-.2)


def test_actual_dtype_rounding_and_autograd_match_dense_group_norm():
    base={'weight':torch.randn(3,4,dtype=torch.float64),'bias':torch.tensor([3.,4.],dtype=torch.float64)}
    groups={'matrix':('weight',),'bias':('bias',)}
    objective=GroupedNormObjective(base,groups,chunk_rows=2)
    new={n:(v*.93+.02).clone().requires_grad_() for n,v in base.items()}
    actual=objective.evaluate(new).mean_relative_change
    expected=torch.stack([(new[n].norm()/base[n].norm()-1) for n in base]).mean()
    torch.testing.assert_close(actual,expected,atol=1e-15,rtol=1e-13)
    actual_grad=torch.autograd.grad(actual,tuple(new.values()),retain_graph=True)
    expected_grad=torch.autograd.grad(expected,tuple(new.values()))
    for a,b in zip(actual_grad,expected_grad):torch.testing.assert_close(a,b,atol=1e-15,rtol=1e-13)
    bf16={'weight':torch.ones(2,3,dtype=torch.bfloat16)}
    rounded=GroupedNormObjective(bf16,{'weight':('weight',)})
    replacement={'weight':(bf16['weight'].float()+1e-5).to(torch.bfloat16)}
    assert rounded.evaluate(replacement).mean_relative_change==0


def test_zero_reference_rejected_but_zero_candidate_has_finite_zero_subgradient():
    with pytest.raises(ValueError,match='undefined groups: zero'):
        GroupedNormObjective({'weight':torch.zeros(2,2)},{'zero':('weight',)})
    base={'weight':torch.ones(2,2,dtype=torch.float64)}
    objective=GroupedNormObjective(base,{'weight':('weight',)})
    replacement=torch.zeros(2,2,dtype=torch.float64,requires_grad=True)
    result=objective.evaluate({'weight':replacement})
    assert result.mean_relative_change==-1
    result.mean_relative_change.backward()
    assert torch.equal(replacement.grad,torch.zeros_like(replacement))
    assert torch.isfinite(replacement.grad).all()


def test_group_coverage_and_unknown_roles_fail_closed():
    base={'a.weight':torch.ones(2,2),'b.bias':torch.ones(3)}
    with pytest.raises(ValueError,match='exactly once'):
        GroupedNormObjective(base,{'a':('a.weight',)})
    with pytest.raises(ValueError,match='exactly once'):
        GroupedNormObjective(base,{'a':('a.weight','a.weight','b.bias')})
    with pytest.raises(ValueError,match='unrecognized'):
        functional_groups(base)


def test_online_grouped_objective_is_explicitly_blocked_until_cached_groups_supported():
    actor=torch.nn.Linear(2,2)
    for config in ({'editable_scope':'grouped'},{'norm_objective':'grouped_l2'}):
        with pytest.raises(NotImplementedError,match='cached per-group'):
            OnlineRewriter(actor,config)


def test_reference_mutation_is_detected_before_using_cached_norms():
    base={'weight':torch.ones(2,2)}
    objective=GroupedNormObjective(base,{'weight':('weight',)})
    base['weight'].mul_(2)
    with pytest.raises(RuntimeError,match='modified in place'):
        objective.evaluate(base)
