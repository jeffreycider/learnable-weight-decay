import pytest
import torch
from rewriter.adaptive_preservation import build_adaptive_proposal
from rewriter.adaptive_thin_student import AdaptiveExecutionCapture, adaptive_thin_tokenwise_panel
from rewriter.tokenwise_reference import TokenwiseConfig, TokenwiseDual, tokenwise_panel
from tests.test_adaptive_preservation_actor import actual_actor_case
from tests.test_adaptive_fp64 import actor_for


@pytest.mark.parametrize('precision', ['fp32','fp64'])
@pytest.mark.parametrize('radius', [1e-6,1.])
@pytest.mark.parametrize('weight', [.5,1.])
def test_multimicro_shared_private_credit_and_guard_match_dense(precision,radius,weight):
    outputs, gradients = [], []
    for thin in (False,True):
        engine,actor,private,snapshot,grouped,inputs=actual_actor_case()
        if precision=='fp64':
            actor,private=actor_for(snapshot,precision='fp64',fresh=False)
            with torch.no_grad():
                actor.matrix_output.weight.normal_(std=.001)
                actor.vector_output.weight.normal_(std=.001)
        teachers=[engine.cache_teacher(x[None],data_role='synthetic')[0] for x in inputs]
        config=TokenwiseConfig(mean_tolerance=1e-10,tail_tolerance=1e-10,scale=1e-5,
            rho=.01,ranking_weight=.01,ranking_k=10,ranking_tie_policy='exact_teacher_ties',ranking_loss_mode='relative_gap')
        dual=TokenwiseDual([6],config=config,dual_lr=.01);dual.multipliers[6]=.1
        capture=AdaptiveExecutionCapture()
        parameters,regularizer,stats,editable,credit=build_adaptive_proposal(
            lambda s:actor(s,private),snapshot,engine,grouped,radius_fraction=radius,
            execution_capture=capture)
        args=dict(config=config,dual=dual,state_id=6,state_weight=weight,actor_credit=credit,
                  teacher_provider=lambda a,b:teachers[a],backward=True)
        result=(adaptive_thin_tokenwise_panel(engine,capture,parameters,regularizer,inputs,**args)
                if thin else tokenwise_panel(engine,parameters,regularizer,inputs,**args))
        outputs.append(result)
        gradients.append({prefix+n:None if p.grad is None else p.grad.clone()
                          for prefix,module in [('shared.',actor),('private.',private)]
                          for n,p in module.named_parameters()})
        assert engine.compute.student_forwards==engine.compute.student_backwards==2
        assert all(p.grad is None for p in engine.parameters.values())
    assert outputs[0]==outputs[1]
    for name,dense in gradients[0].items():
        thin=gradients[1][name]
        assert (dense is None)==(thin is None),name
        if dense is not None:
            torch.testing.assert_close(dense,thin,rtol=5e-4,atol=2e-5,msg=name)


@pytest.mark.parametrize('bad_orientation', ['embedding','square_linear'])
def test_reject_unsupported_storage_orientation_before_forward(bad_orientation):
    engine,actor,private,snapshot,grouped,inputs=actual_actor_case()
    capture=AdaptiveExecutionCapture()
    parameters,regularizer,*_=build_adaptive_proposal(lambda s:actor(s,private),
        snapshot,engine,grouped,radius_fraction=.01,execution_capture=capture)
    engine.transpose_storage=dict(engine.transpose_storage)
    engine.transpose_storage['embed.weight' if bad_orientation=='embedding' else 'blocks.0.attn.q.weight'] = bad_orientation!='embedding'
    with pytest.raises(ValueError,match='transposed embedding storage only'):
        adaptive_thin_tokenwise_panel(engine,capture,parameters,regularizer,inputs)
