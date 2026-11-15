import pytest
import torch
from rewriter.adaptive_preservation import build_adaptive_proposal
from rewriter.experimental_factor_panel import FactorPanelCapture
from rewriter.tokenwise_reference import TokenwiseConfig,TokenwiseDual,tokenwise_panel
from tests.test_adaptive_preservation_actor import actual_actor_case


@pytest.mark.parametrize('radius',[1e-6,1.])
@pytest.mark.parametrize('zero',[False,True])
@pytest.mark.parametrize('factor_precision',['fp32','fp64'])
def test_actual_panel_matches_dense_actor_private_credit_counts_and_stats(radius,zero,factor_precision):
    torch.set_num_threads(1)
    results=[];gradients=[];counts=[]
    for direct in (False,True):
        engine,actor,private,snapshot,grouped,inputs=actual_actor_case()
        if factor_precision=='fp64':
            from rewriter.adaptive_coordinate_chisel import AdaptiveCoordinateChisel
            torch.manual_seed(133)
            actor=AdaptiveCoordinateChisel(rank=2,trajectory_dim=12,model_dim=128,num_layers=1,latent_tokens=2,factor_precision='fp64',residual_precision='fp32')
            private=actor.make_private(snapshot,state_key=('test',0,6),seed=77)
            with torch.no_grad():
                for head in actor.output_heads:head.weight.normal_(std=.001)
                actor.blocks[0].attn.proj.weight.normal_(std=.001)
        inputs=inputs[:,:4]  # two actual one-sequence GPT microbatches
        if zero:
            with torch.no_grad():
                for head in actor.output_heads:head.weight.zero_();head.bias.zero_()
        cached=[engine.cache_teacher(x[None],data_role='synthetic')[0] for x in inputs]
        config=TokenwiseConfig(mean_tolerance=1e-10,tail_tolerance=1e-10,scale=1e-5,rho=.01,
            ranking_weight=.01,ranking_k=10,ranking_tie_policy='exact_teacher_ties',ranking_loss_mode='relative_gap')
        dual=TokenwiseDual([6],config=config,dual_lr=.01);dual.multipliers[6]=.1
        capture=FactorPanelCapture(chunk_tokens=3)
        parameters,regularizer,stats,editable,credit=build_adaptive_proposal(
            lambda snap:actor(snap,private),snapshot,engine,grouped,radius_fraction=radius,
            execution_capture=capture if direct else None)
        assert stats['radius_saturated']==(radius<1e-5 and not zero)
        kwargs=dict(config=config,dual=dual,state_id=6,state_weight=.5,actor_credit=credit,
                    teacher_provider=lambda a,b:cached[a])
        result=capture.run(engine,parameters,regularizer,inputs,**kwargs) if direct else tokenwise_panel(engine,parameters,regularizer,inputs,**kwargs)
        results.append(result);counts.append(engine.compute.as_dict())
        gradients.append({prefix+n:None if p.grad is None else p.grad.clone() for prefix,m in [('actor.',actor),('private.',private)] for n,p in m.named_parameters()})
        assert engine.compute.student_forwards==engine.compute.student_backwards==2
        assert all(p.grad is None and not p.requires_grad for p in engine.parameters.values())
        if direct:
            assert capture.phase=='consumed' and not capture.credit_leaves
            with pytest.raises(RuntimeError,match='fresh'):capture.run(engine,parameters,regularizer,inputs,**kwargs)
            with pytest.raises(RuntimeError,match='one-shot'):capture(None,None)
    assert results[0]==results[1] and counts[0]==counts[1]
    for name in gradients[0]:
        a,b=gradients[0][name],gradients[1][name]
        assert (a is None)==(b is None),name
        if a is not None:torch.testing.assert_close(a,b,rtol=5e-4,atol=3e-5,msg=name)
    assert gradients[1]['actor.matrix_output.weight'].norm()>0


def test_capture_mutation_rejected_and_failure_consumes():
    engine,actor,private,snapshot,grouped,inputs=actual_actor_case()
    capture=FactorPanelCapture()
    parameters,regularizer,_,_,credit=build_adaptive_proposal(lambda snap:actor(snap,private),snapshot,engine,grouped,radius_fraction=.1,execution_capture=capture)
    with torch.no_grad():next(iter(capture.action.matrices.values())).U.add_(.1)
    with pytest.raises(RuntimeError,match='mutated'):capture.run(engine,parameters,regularizer,inputs)
    assert capture.phase=='consumed' and capture.action is None
    assert all('forward' not in m.__dict__ for m in engine.view.gpt.modules())


def test_mid_panel_failure_consumes_capture_and_restores_forward_methods():
    engine,actor,private,snapshot,grouped,inputs=actual_actor_case()
    cached=engine.cache_teacher(inputs[:1],data_role='synthetic')[0]
    capture=FactorPanelCapture()
    parameters,regularizer,_,_,credit=build_adaptive_proposal(lambda snap:actor(snap,private),snapshot,engine,grouped,radius_fraction=.1,execution_capture=capture)
    config=TokenwiseConfig(mean_tolerance=1e-10,tail_tolerance=1e-10,scale=1e-5,rho=.01,ranking_k=10);dual=TokenwiseDual([6],config=config,dual_lr=.01)
    def teacher(start,stop):
        if start:raise RuntimeError('test second microbatch failure')
        return cached
    with pytest.raises(RuntimeError,match='second microbatch'):
        capture.run(engine,parameters,regularizer,inputs,config=config,dual=dual,state_id=6,teacher_provider=teacher,actor_credit=credit)
    assert capture.phase=='consumed' and not capture.credit_leaves and capture.action is None
    assert engine.compute.student_forwards==engine.compute.student_backwards==1
    assert all('forward' not in m.__dict__ and not hasattr(m,'_factor_forward_active') for m in engine.view.gpt.modules())
    assert all(p.grad is None for p in actor.parameters())
