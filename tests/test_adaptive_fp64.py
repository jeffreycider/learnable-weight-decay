"""Optional factor precision: unchanged backbone/GPT storage, complete credit."""
import copy
import pytest
import torch

from rewriter.adaptive_coordinate_chisel import AdaptiveCoordinateChisel
from rewriter.adaptive_sensing import query_weight_cores
from rewriter.adaptive_preservation import build_adaptive_proposal
from rewriter.tokenwise_reference import TokenwiseConfig, TokenwiseDual, tokenwise_panel
from tests.test_controller import make_snapshot
from tests.test_adaptive_preservation_actor import actual_actor_case


def actor_for(snapshot, *, precision='fp64', fresh=True):
    torch.manual_seed(301)
    actor = AdaptiveCoordinateChisel(rank=2, trajectory_dim=12, model_dim=128,
        num_layers=1, latent_tokens=2, fresh_weight_core=fresh, factor_precision=precision)
    private = actor.make_private(snapshot, state_key=('test',0,6), seed=17)
    return actor, private


def test_fp64_query_heads_frames_and_zero_head_learnability():
    torch.set_num_threads(1)
    snapshot = make_snapshot()
    actor, private = actor_for(snapshot)
    weights = {n:torch.randn(o.rows.shape[0],o.columns.shape[0],requires_grad=True)
               for n,o in snapshot.matrices.items()}
    before = {n:w.detach().clone() for n,w in weights.items()}
    cores = query_weight_cores(weights,private,embedding_names=())
    assert all(c.dtype == torch.float64 for c in cores.values())
    action = actor(snapshot,private,weight_cores=cores)
    assert all(f.U.dtype == f.C.dtype == f.V.dtype == torch.float64 for f in action.matrices.values())
    assert all(v.dtype == torch.float64 for v in action.vectors.values())
    assert all(p.dtype == torch.float32 for p in actor.blocks.parameters())
    assert private.trajectory_input.weight.dtype == actor.weight_core_input.weight.dtype == torch.float32
    assert all(t.count_nonzero()==0 for t in action.dense().values())
    sum((t*torch.randn_like(t)).sum() for t in action.dense().values()).backward()
    assert actor.matrix_output.weight.grad.norm()>0
    assert actor.vector_output.weight.grad.norm()>0
    for n,w in weights.items():
        assert w.grad is None
        torch.testing.assert_close(w,before[n],rtol=0,atol=0)
    actor.zero_grad();private.zero_grad()
    with torch.no_grad():
        actor.matrix_output.weight.normal_(std=.002)
        actor.vector_output.weight.normal_(std=.002)
    action=actor(snapshot,private,weight_cores=query_weight_cores(weights,private,embedding_names=()))
    sum((t*torch.randn_like(t)).sum() for t in action.dense().values()).backward()
    assert private.matrix_rows[0].raw.grad.norm()>0
    assert private.matrix_columns[0].raw.grad.norm()>0
    assert actor.weight_core_input.weight.grad.norm()>0


@pytest.mark.parametrize('radius',[1e-6,1.])
def test_actual_fp32_gpt_fp64_direct_and_detached_credit(radius):
    results, gradients = [], []
    for detached in (False,True):
        engine,_,_,snapshot,grouped,inputs=actual_actor_case()
        actor,private=actor_for(snapshot)
        with torch.no_grad():
            actor.matrix_output.weight.normal_(std=.001)
            actor.vector_output.weight.normal_(std=.001)
            actor.blocks[0].attn.proj.weight.normal_(std=.001)
        before={n:p.clone() for n,p in engine.parameters.items()}
        model_before={n:p.clone() for n,p in engine.view.gpt.state_dict().items()}
        cached=[engine.cache_teacher(x[None],data_role='synthetic')[0] for x in inputs]
        config=TokenwiseConfig(mean_tolerance=1e-10,tail_tolerance=1e-10,scale=1e-5,
            rho=.01,ranking_weight=.01,ranking_k=10,ranking_tie_policy='exact_teacher_ties',ranking_loss_mode='relative_gap')
        dual=TokenwiseDual([6],config=config,dual_lr=.01);dual.multipliers[6]=.1
        parameters,regularizer,stats,editable,credit=build_adaptive_proposal(
            lambda s:actor(s,private,weight_cores=query_weight_cores(engine.parameters,private)),
            snapshot,engine,grouped,radius_fraction=radius,detach_credit=detached)
        assert stats['factor_precision']=='fp64' and stats['radius_saturated']==(radius<1e-5)
        assert all(p.dtype==torch.float32 for p in parameters.values())
        results.append(tokenwise_panel(engine,parameters,regularizer,inputs,config=config,dual=dual,
            state_id=6,actor_credit=credit,teacher_provider=lambda a,b:cached[a]))
        gradients.append({prefix+n:None if p.grad is None else p.grad.clone()
                          for prefix,m in [('actor.',actor),('private.',private)] for n,p in m.named_parameters()})
        for n,p in engine.parameters.items():
            torch.testing.assert_close(p,before[n],rtol=0,atol=0);assert p.grad is None
        for n,p in engine.view.gpt.state_dict().items():torch.testing.assert_close(p,model_before[n],rtol=0,atol=0)
        assert engine.compute.student_forwards==engine.compute.student_backwards==2
    assert results[0]==results[1]
    for name,a in gradients[0].items():
        b=gradients[1][name];assert (a is None)==(b is None),name
        if a is not None:torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6,msg=name)
    assert gradients[1]['private.matrix_rows.0.raw'].norm()>0
    assert gradients[1]['actor.weight_core_input.weight'].norm()>0


def test_distinct_precision_identity_and_no_silent_tensor_cast_loading():
    snapshot=make_snapshot();old,old_private=actor_for(snapshot,precision='fp32')
    new,new_private=actor_for(snapshot)
    assert old.configuration()['name']!=new.configuration()['name']
    for target,state in [(new,old.state_dict()),(old,new.state_dict()),
                         (new_private,old_private.state_dict()),(old_private,new_private.state_dict())]:
        with pytest.raises(ValueError):target.load_state_dict(state)
    for target,key in [(new,'matrix_output.weight'),(new_private,'matrix_rows.0.raw')]:
        state=copy.deepcopy(target.state_dict());state[key]=state[key].float()
        with pytest.raises(ValueError,match='precision'):target.load_state_dict(state)
    clone,clone_private=actor_for(snapshot)
    clone.load_state_dict(new.state_dict());clone_private.load_state_dict(new_private.state_dict())
    new.float()
    with pytest.raises(ValueError,match='precision'):new(snapshot,new_private,weight_cores={})


def test_legacy_default_is_explicit_fp32_identical():
    snapshot=make_snapshot();torch.manual_seed(41)
    default=AdaptiveCoordinateChisel(rank=2,trajectory_dim=12,model_dim=128,num_layers=1,latent_tokens=2)
    torch.manual_seed(41)
    explicit=AdaptiveCoordinateChisel(rank=2,trajectory_dim=12,model_dim=128,num_layers=1,latent_tokens=2,factor_precision='fp32')
    assert default.configuration()==explicit.configuration()
    assert 'factor_precision' not in default.configuration()
    for n,p in default.named_parameters():torch.testing.assert_close(p,dict(explicit.named_parameters())[n],rtol=0,atol=0)


def test_cifar_fp64_detached_credit_and_single_storage_cast():
    from tests.test_adaptive_preservation_cifar import case
    from rewriter.image_tokenwise_panel import image_tokenwise_panel
    from rewriter.adaptive_preservation import group_rms_scales
    outcomes=[];gradients=[]
    for detach in (False,True):
        engine,images,labels,snapshot,_,_,_,grouped=case()
        torch.manual_seed(811)
        actor=AdaptiveCoordinateChisel(rank=2,trajectory_dim=100,model_dim=128,
            num_layers=1,latent_tokens=2,fresh_weight_core=True,factor_precision='fp64')
        private=actor.make_private(snapshot,state_key=('cifar10',0,1),seed=81)
        with torch.no_grad():
            actor.matrix_output.weight.normal_(std=.002);actor.vector_output.weight.normal_(std=.002)
        observed={n:p.flatten(1) if p.ndim==4 else p for n,p in engine._all_parameters.items()}
        forward=lambda s:actor(s,private,weight_cores=query_weight_cores(observed,private))
        teacher=engine._forward(engine.parameters,images).detach()
        before={n:p.clone() for n,p in engine._all_parameters.items()}
        raw=forward(snapshot);scales=group_rms_scales(grouped,dtype=torch.float64)
        # Fail if the new branch accidentally uses CNN's early delta.float().
        engine.edited_parameters=lambda *a,**k:(_ for _ in ()).throw(AssertionError('legacy materialization'))
        params,regularizer,stats,editable,credit=build_adaptive_proposal(forward,snapshot,engine,grouped,
            radius_fraction=1.,detach_credit=detach)
        assert not stats['radius_saturated']
        for n,base in engine.parameters.items():
            delta=raw.matrices[n].dense() if base.ndim==2 else raw.vectors[n]
            expected=(base.double()+scales[n]*delta).to(base.dtype)
            torch.testing.assert_close(params[n],expected,rtol=0,atol=0)
            assert params[n].dtype==torch.float32
        config=TokenwiseConfig(mean_tolerance=1e-10,tail_tolerance=1e-10,scale=1e-5,rho=.01,
            ranking_weight=.01,ranking_k=3,ranking_tie_policy='exact_teacher_ties',ranking_loss_mode='relative_gap')
        dual=TokenwiseDual([1],config=config,dual_lr=.01);dual.multipliers[1]=.1
        outcomes.append(image_tokenwise_panel(engine,params,regularizer,images,labels,config=config,
            dual=dual,state_id=1,actor_credit=credit,
            teacher_provider=lambda a,b:engine.teacher_batch(images[a:b],teacher[a:b],range(a,b))))
        gradients.append({prefix+n:None if p.grad is None else p.grad.clone()
            for prefix,m in [('actor.',actor),('private.',private)] for n,p in m.named_parameters()})
        for n,p in engine._all_parameters.items():torch.testing.assert_close(p,before[n],rtol=0,atol=0)
    assert outcomes[0]==outcomes[1]
    for n,a in gradients[0].items():
        b=gradients[1][n];assert (a is None)==(b is None)
        if a is not None:torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6,msg=n)
