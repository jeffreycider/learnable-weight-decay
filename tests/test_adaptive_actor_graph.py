import pytest
import torch
from tests.test_controller import make_snapshot
from rewriter.adaptive_coordinate_chisel import AdaptiveCoordinateChisel
from rewriter.adaptive_sensing import query_weight_cores
from rewriter.adaptive_actor_graph import EditableAdaptiveOutputs,AdaptiveActorGraph

@pytest.mark.parametrize('precision',['fp32','fp64'])
def test_live_query_filtered_outputs_exact_values_and_shared_private_vjp(precision):
    torch.set_num_threads(1);torch.manual_seed(17);s=make_snapshot()
    actor=AdaptiveCoordinateChisel(rank=2,trajectory_dim=12,model_dim=128,num_layers=2,latent_tokens=3,fresh_weight_core=True,**({'factor_precision':precision} if precision=='fp64' else {}))
    private=actor.make_private(s,state_key=('gpt',0,6),seed=5)
    weights={n:torch.randn(o.rows.shape[0],o.columns.shape[0]) for n,o in s.matrices.items()}
    with torch.no_grad():
        actor.matrix_output.weight.normal_(std=.03);actor.vector_output.weight.normal_(std=.03)
    editable={next(iter(s.matrices))}
    wrapper=EditableAdaptiveOutputs(actor,private,s,editable,weights,())
    expected=actor(s,private,weight_cores=query_weight_cores(weights,private,embedding_names=()))
    actual=wrapper.action(wrapper(torch.zeros(())));name=next(iter(editable))
    for field in ('U','C','V'):torch.testing.assert_close(getattr(actual.matrices[name],field),getattr(expected.matrices[name],field),atol=0,rtol=0)
    parameters=tuple(actor.parameters())+tuple(private.parameters())
    a=torch.autograd.grad(actual.matrices[name].norm_sq(),parameters,allow_unused=True)
    e=torch.autograd.grad(expected.matrices[name].norm_sq(),parameters,allow_unused=True)
    for x,y in zip(a,e):
        assert (x is None)==(y is None)
        if x is not None:torch.testing.assert_close(x,y,atol=0,rtol=0)
    assert any(x is not None and x.abs().sum()>0 for x in a[len(tuple(actor.parameters())):])
    assert all(not v.requires_grad and not v.count_nonzero() for v in actual.vectors.values())
    assert actual.matrices[name].U.dtype==(torch.float64 if precision=='fp64' else torch.float32)


def test_weight_rebind_pending_layout_mutation_and_address_guards():
    graph=object.__new__(AdaptiveActorGraph);graph.pending=False;graph.embedding_names=()
    graph.weights={'a':torch.ones(3,2)};graph.weight_layout={'a':(torch.Size([3,2]),(2,1),torch.float32,torch.device('cpu'))}
    graph._weights_bound=False
    address=graph.weights['a'].data_ptr();graph.bind_weights({'a':torch.zeros(3,2)},embedding_names=());graph._check_weights()
    assert graph.weights['a'].data_ptr()==address and not graph.weights['a'].count_nonzero()
    graph.pending=True
    with pytest.raises(RuntimeError,match='pending'):graph.bind_weights({'a':torch.ones(3,2)},embedding_names=())
    graph.pending=False
    with pytest.raises(ValueError,match='layout'):graph.bind_weights({'a':torch.ones(2,3)},embedding_names=())
    with pytest.raises(RuntimeError,match='rebind'):graph._check_weights()
    graph.bind_weights({'a':torch.ones(3,2)},embedding_names=());graph.weights['a'].add_(1)
    with pytest.raises(RuntimeError,match='rebind'):graph._check_weights()


def test_native_adaptive_graph_recipe_explicit_precision_and_family_guards():
    from tests.test_coordinate_training_driver import recipe
    from scripts.train_coordinate_chisel import validate_recipe
    r=recipe();r.update(action_family='adaptive_ucv_v1',student_backend='dense',actor_backend='adaptive_actor_graph',private_actor_backend='adaptive_actor_graph',private_inner_steps=8)
    r['architecture'].update(frame_mode='tied_orthonormal',fresh_weight_core=True)
    validate_recipe(r)
    r['architecture']['factor_precision']='fp64'
    validate_recipe(r)
    r['architecture']['factor_precision']='bf16'
    with pytest.raises(ValueError,match='factor_precision'):validate_recipe(r)
    r['architecture']['factor_precision']='fp32';r['action_family']='anchored'
    with pytest.raises(ValueError):validate_recipe(r)


def test_context_rebind_tracks_values_versions_and_stable_cnn_views():
    from types import SimpleNamespace
    g=object.__new__(AdaptiveActorGraph);g.pending=False;g.embedding_names=()
    source=torch.ones(3,2);g.weights={'a':source.clone()};g.weight_layout={'a':(source.shape,source.stride(),source.dtype,source.device)}
    context={'local_id':1,'engine':SimpleNamespace(parameters={'a':source},transpose_storage={})}
    g.bind_context(context);address=g.weights['a'].data_ptr();g._check_weights()
    source.mul_(2)
    with pytest.raises(RuntimeError,match='source checkpoint'):g._check_weights()
    g.bind_context(context);assert g.weights['a'].data_ptr()==address and torch.equal(g.weights['a'],source)
    context['local_id']=2;context['engine'].parameters={'a':torch.full_like(source,3)}
    g.bind_context(context);assert g.weights['a'].eq(3).all()


def test_scoped_snapshot_keeps_observation_identity_but_new_state_does_not():
    from rewriter.edit_scope import scoped_snapshot
    s=make_snapshot()
    scoped=scoped_snapshot(s,torch.tensor(2.),.01,scope='grouped')
    assert scoped is not s
    assert AdaptiveActorGraph._observation_signature(scoped)==AdaptiveActorGraph._observation_signature(s)
    changed=make_snapshot()
    assert AdaptiveActorGraph._observation_signature(changed)!=AdaptiveActorGraph._observation_signature(s)
