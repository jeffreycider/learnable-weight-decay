"""A real CNN's complete observations and smaller finite write inventory."""
import pytest
import torch

from rewriter.adaptive_action import AdaptiveAction
from rewriter.adaptive_coordinate_chisel import AdaptiveCoordinateChisel
from rewriter.adaptive_preservation import build_adaptive_proposal
from rewriter.adaptive_sensing import query_weight_cores
from rewriter.cifar_adapter import observe_cifar_terminal
from rewriter.grouped_norm import GroupedNormObjective
from rewriter.image_tokenwise_panel import image_tokenwise_panel
from rewriter.tokenwise_reference import TokenwiseConfig,TokenwiseDual
from tests.test_cifar_adapter import fixture


def case():
    _,engine,images,labels=fixture()
    engine.microbatch_sequences=2
    snapshot,_=observe_cifar_terminal(engine,images,labels,momentum={},anchor_seed=87,
                                      rank=2,trajectory_images=4)
    actor=AdaptiveCoordinateChisel(rank=2,trajectory_dim=100,model_dim=128,
                                  num_layers=1,latent_tokens=2,fresh_weight_core=True)
    private=actor.make_private(snapshot,state_key=('cifar10',0,1),seed=71)
    with torch.no_grad():
        actor.matrix_output.weight.normal_(std=.002)
        actor.vector_output.weight.normal_(std=.002)
        actor.blocks[0].attn.proj.weight.normal_(std=.002)
    full_weights={n:p.flatten(1) if p.ndim==4 else p for n,p in engine._all_parameters.items()}
    def forward(snap):
        return actor(snap,private,weight_cores=query_weight_cores(full_weights,private))
    grouped=GroupedNormObjective(engine.parameters,groups=engine.groups,zero_reference_policy='freeze')
    return engine,images,labels,snapshot,actor,private,forward,grouped


@pytest.mark.parametrize('radius',[1e-6,1.])
def test_cnn_direct_detached_full_observations_writable_subset_exact_credit(radius):
    outcomes=[];gradients=[]
    for detach in (False,True):
        engine,images,labels,snapshot,actor,private,forward,grouped=case()
        before={n:p.clone() for n,p in {**engine._all_parameters,**engine.buffers}.items()}
        with torch.no_grad():teacher_logits=engine._forward(engine.parameters,images)
        params,regularizer,stats,editable,credit=build_adaptive_proposal(
            forward,snapshot,engine,grouped,radius_fraction=radius,detach_credit=detach)
        assert params.keys()==engine.parameters.keys()
        assert stats['radius_saturated']==(radius<1e-5)
        excluded=set(engine._all_parameters)-set(engine.parameters)
        assert set(stats['readonly_observation_only_parameters'])==excluded
        assert 'whiten.weight' in excluded and 'layers.1.norm1.weight' in excluded
        if detach:
            assert len(credit[0])==3*sum(p.ndim==2 for p in engine.parameters.values())+sum(p.ndim==1 for p in engine.parameters.values())
        config=TokenwiseConfig(mean_tolerance=1e-10,tail_tolerance=1e-10,scale=1e-5,
            rho=.01,ranking_weight=.01,ranking_k=3,ranking_tie_policy='exact_teacher_ties',ranking_loss_mode='relative_gap')
        dual=TokenwiseDual([1],config=config,dual_lr=.01);dual.multipliers[1]=.1
        outcomes.append(image_tokenwise_panel(engine,params,regularizer,images,labels,
            config=config,dual=dual,state_id=1,actor_credit=credit,
            teacher_provider=lambda a,b:engine.teacher_batch(images[a:b],teacher_logits[a:b],range(a,b))))
        gradients.append({prefix+n:None if p.grad is None else p.grad.clone()
                          for prefix,module in [('shared.',actor),('private.',private)]
                          for n,p in module.named_parameters()})
        for n,p in {**engine._all_parameters,**engine.buffers}.items():
            torch.testing.assert_close(p,before[n],rtol=0,atol=0)
            assert p.grad is None and not p.requires_grad
    assert outcomes[0]==outcomes[1]
    assert outcomes[0]['compute']['student_forwards']==outcomes[0]['compute']['student_backwards']==3
    for n,g in gradients[0].items():
        actual=gradients[1][n]
        assert (actual is None)==(g is None),n
        if g is not None:torch.testing.assert_close(actual,g,rtol=2e-4,atol=2e-5,msg=n)
    assert gradients[1]['shared.weight_core_input.weight'].norm()>0
    assert gradients[1]['private.matrix_rows.0.raw'].norm()>0  # frozen whitening may still be useful to READ


def test_missing_writable_or_unobserved_extra_action_rejected():
    engine,_,_,snapshot,_,_,forward,grouped=case()
    action=forward(snapshot)
    missing=AdaptiveAction({n:f for n,f in action.matrices.items() if n!='head.weight'},action.vectors)
    arbitrary=AdaptiveAction({**action.matrices,'unobserved.weight':action.matrices['head.weight']},action.vectors)
    for invalid in (missing,arbitrary):
        with pytest.raises(ValueError,match='inventory'):
            build_adaptive_proposal(lambda _:invalid,snapshot,engine,grouped,radius_fraction=.01)
