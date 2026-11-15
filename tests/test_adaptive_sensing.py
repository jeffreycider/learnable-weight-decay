import pytest
import torch

from rewriter.adaptive_coordinate_chisel import (
    AdaptiveCoordinateChisel, WEIGHT_CORE_ARCHITECTURE, normalized_weight_core)
from rewriter.adaptive_sensing import query_weight_cores
from tests.test_controller import make_snapshot


def fixture(*, fresh=True, latent=3):
    torch.set_num_threads(1);torch.manual_seed(461)
    snapshot=make_snapshot()
    actor=AdaptiveCoordinateChisel(rank=2,trajectory_dim=12,model_dim=128,
                                  num_layers=1,latent_tokens=latent,fresh_weight_core=fresh)
    private=actor.make_private(snapshot,state_key=('gpt',0,6),seed=17)
    # Choose dense representatives consistent with BOTH existing W sketches.
    weights={}
    for name,obs in snapshot.matrices.items():
        a,b=obs.anchors.A,obs.anchors.B;wb,wta=obs.weight_probes
        weights[name]=(wb@b.T+a@wta.T-a@(a.T@wb)@b.T).detach().requires_grad_()
    weights.update({name:obs.weight.clone().requires_grad_() for name,obs in snapshot.vectors.items()})
    return snapshot,actor,private,weights


@pytest.mark.parametrize('latent',[3,8])
def test_fresh_contraction_orientation_private_gradients_and_source_readonly(latent):
    snapshot,actor,private,weights=fixture(latent=latent)
    # Exercise explicit transposed storage on a nonsquare matrix.
    stored={**weights,'first':weights['first'].T.detach().clone().requires_grad_()}
    before={n:(p.detach().clone(),p._version) for n,p in stored.items()}
    cores=query_weight_cores(stored,private,embedding_names=('first',))
    refs=[]
    for i,name in enumerate(private.matrix_names):
        u=private.matrix_rows[i].orthonormal_matrix();v=private.matrix_columns[i].orthonormal_matrix()
        w=stored[name].T if name=='first' else stored[name]
        expected=u.T@w.detach()@v
        torch.testing.assert_close(cores[name],expected,rtol=3e-6,atol=3e-6)
        refs.append((expected*torch.arange(latent*latent).reshape(latent,latent)).sum())
    objective=sum((value*torch.arange(latent*latent).reshape(latent,latent)).sum() for value in cores.values())
    frame_params=[f.raw for f in (*private.matrix_rows,*private.matrix_columns)]
    gradients=torch.autograd.grad(objective,frame_params+list(stored.values()),allow_unused=True)
    expected=torch.autograd.grad(sum(refs),frame_params)
    for g,r in zip(gradients,expected):torch.testing.assert_close(g,r,rtol=2e-5,atol=2e-5)
    assert all(g.norm()>0 for g in gradients[:len(frame_params)])
    assert all(g is None for g in gradients[len(frame_params):])
    for n,p in stored.items():
        torch.testing.assert_close(p,before[n][0],rtol=0,atol=0)
        assert p._version==before[n][1] and p.grad is None


def test_invalid_inventory_shape_and_unrequested_input_fail_closed():
    snapshot,actor,private,weights=fixture()
    with pytest.raises(ValueError,match='nonempty'):query_weight_cores({},private)
    with pytest.raises(ValueError,match='inventory'):query_weight_cores({'first':weights['first']},private)
    with pytest.raises(ValueError,match='shape'):query_weight_cores({**weights,'first':weights['first'].T},private)
    cores=query_weight_cores(weights,private)
    for missing in (None,{}, {'first':cores['first']}):
        with pytest.raises(ValueError,match='mapping'):actor(snapshot,private,weight_cores=missing)
    with pytest.raises(ValueError,match='shape'):
        actor(snapshot,private,weight_cores={**cores,'first':torch.zeros(2,2)})
    _,coreless,old_private,_=fixture(fresh=False)
    with pytest.raises(ValueError,match='coreless'):coreless(snapshot,old_private,weight_cores=cores)
    with pytest.raises(ValueError,match='identity'):actor(snapshot,old_private,weight_cores=cores)


def test_zero_weights_safe_normalization_and_zero_head_can_learn():
    snapshot,actor,private,weights=fixture()
    weights={n:torch.zeros_like(w,requires_grad=True) for n,w in weights.items()}
    cores=query_weight_cores(weights,private)
    assert all(c.count_nonzero()==0 for c in cores.values())
    for c in cores.values():
        normalized,logscale=normalized_weight_core(c)
        assert torch.isfinite(normalized).all() and torch.isfinite(logscale)
    action=actor(snapshot,private,weight_cores=cores)
    assert action.norm_sq()==0
    action.contraction({n:torch.randn_like(f.dense()) for n,f in action.matrices.items()},
                       {n:torch.randn_like(v) for n,v in action.vectors.items()}).backward()
    assert actor.matrix_output.weight.grad.norm()>0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in private.parameters())
    assert all(w.grad is None for w in weights.values())
    # Test normalization itself, including the logscale derivative, away from
    # the protected zero where finite differences cannot resolve the tiny floor.
    core=torch.randn(3,3,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda x:normalized_weight_core(x),(core,))


def test_information_outside_old_probes_changes_core_and_global_edit():
    snapshot,actor,private,weights=fixture()
    with torch.no_grad():
        actor.matrix_output.weight.normal_(std=.05)
        actor.blocks[0].attn.proj.weight.normal_(std=.02)
    obs=snapshot.matrices['first'];a,b=obs.anchors.A,obs.anchors.B
    z=torch.randn_like(weights['first'])
    left=z-a@(a.T@z)
    delta=left-(left@b)@b.T  # no dense square projectors
    changed={**weights,'first':weights['first'].detach()+delta}
    torch.testing.assert_close(changed['first']@b,weights['first']@b,rtol=3e-6,atol=3e-6)
    torch.testing.assert_close(changed['first'].T@a,weights['first'].T@a,rtol=3e-6,atol=3e-6)
    before=query_weight_cores(weights,private);after=query_weight_cores(changed,private)
    assert (before['first']-after['first']).norm()>1e-3
    # Exact same immutable cached snapshot; no fresh gradients or activations.
    first=actor(snapshot,private,weight_cores=before)
    second=actor(snapshot,private,weight_cores=after)
    assert (first.matrices['first'].dense()-second.matrices['first'].dense()).norm()>1e-6
    assert (first.matrices['second'].dense()-second.matrices['second'].dense()).norm()>1e-6
    first.norm_sq().backward()
    assert actor.weight_core_input.weight.grad.norm()>0
    assert private.matrix_rows[0].raw.grad.norm()>0
    assert private.matrix_columns[0].raw.grad.norm()>0
    assert all(w.grad is None for w in weights.values())


def test_token_count_blocks_and_default_state_are_unchanged_but_identity_is_explicit():
    snapshot,actor,private,weights=fixture()
    _,coreless,old_private,_=fixture(fresh=False)
    assert actor.token_counts(snapshot)==coreless.token_counts(snapshot)
    for name,value in coreless.blocks.state_dict().items():
        torch.testing.assert_close(actor.blocks.state_dict()[name],value,atol=0,rtol=0)
    x,reads,count=actor._inputs(snapshot,private)
    y=actor._weight_core_tokens(snapshot,private,x,reads,count,query_weight_cores(weights,private))
    assert x.shape==y.shape
    torch.testing.assert_close(x[:,count:reads],y[:,count:reads],atol=0,rtol=0)
    torch.testing.assert_close(x[:,reads+count:],y[:,reads+count:],atol=0,rtol=0)
    assert actor.configuration()['name']==WEIGHT_CORE_ARCHITECTURE
    assert not actor.configuration()['fresh_gradient_or_momentum_queries']
    with pytest.raises(ValueError,match='architecture'):actor.load_state_dict(coreless.state_dict())
    with pytest.raises(ValueError,match='identity'):private.load_state_dict(old_private.state_dict())
    # Explicit false is the old default: even the metadata and random draw
    # sequence are preserved, so existing coreless tests/checkpoints stay valid.
    torch.manual_seed(461)
    default=AdaptiveCoordinateChisel(rank=2,trajectory_dim=12,model_dim=128,num_layers=1,latent_tokens=3)
    for name,value in coreless.state_dict().items():
        other=default.state_dict()[name]
        if isinstance(value,torch.Tensor):torch.testing.assert_close(value,other,rtol=0,atol=0)
        else:assert value==other
