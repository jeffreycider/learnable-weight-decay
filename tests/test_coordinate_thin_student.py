import pytest
import torch
from types import SimpleNamespace
from tests.test_factorized_fp32 import setup
from rewriter.algebra import RawAction,Factors
from rewriter.coordinate_thin_student import ProposalCapture,thin_tokenwise_panel
from rewriter.tokenwise_reference import TokenwiseConfig,TokenwiseDual,tokenwise_panel


@pytest.mark.parametrize('saturated',[False,True])
@pytest.mark.parametrize('state_weight',[1.,1/200])
def test_multimicro_thin_panel_matches_dense_shared_private_credit(saturated,state_weight):
    torch.set_num_threads(1)
    engine,template,_,inputs=setup();engine.microbatch_sequences=1
    config=TokenwiseConfig(mean_tolerance=1e-10,tail_tolerance=1e-10,scale=1e-5,rho=.01,ranking_weight=.01,
        ranking_k=10,ranking_tie_policy='exact_teacher_ties',ranking_loss_mode='relative_gap')
    dual=TokenwiseDual([3],config=config,dual_lr=.01);dual.multipliers[3]=.1
    teachers=[engine.cache_teacher(x[None],data_role='synthetic')[0] for x in inputs]
    outputs=[];gradients=[]
    for thin in (False,True):
        shared=torch.tensor(.3,requires_grad=True);private=torch.tensor(.7,requires_grad=True)
        action=RawAction({n:Factors(f.P.detach()*shared.sin(),f.Q.detach()*private.square(),f.anchors) for n,f in template.matrices.items()},
                         {n:v.detach()*shared*private for n,v in template.vectors.items()})
        scale=.01/action.norm_sq().sqrt() if saturated else action.norm_sq().new_ones(())
        capture=ProposalCapture(engine);parameters=capture.edited_parameters(action,scale)
        norm=sum(v.double().square().mean() for v in parameters.values())*.001
        args=dict(config=config,dual=dual,state_id=3,state_weight=state_weight,teacher_provider=lambda a,b:teachers[a],backward=True)
        result=thin_tokenwise_panel(capture,parameters,norm,inputs,**args) if thin else tokenwise_panel(engine,parameters,norm,inputs,**args)
        outputs.append(result);gradients.append(torch.stack((shared.grad,private.grad)))
    assert outputs[0]==outputs[1]
    torch.testing.assert_close(gradients[0],gradients[1],rtol=3e-5,atol=1e-8)
    assert all(p.grad is None for p in engine.parameters.values())


def test_capture_fails_closed_and_does_not_mutate_engine():
    engine,action,_,_=setup();original=engine.edited_parameters
    capture=ProposalCapture(engine);capture.edited_parameters(action,torch.ones(()))
    with pytest.raises(RuntimeError,match='one proposal'):capture.edited_parameters(action,torch.ones(()))
    assert engine.edited_parameters==original


def test_qualifier_actor_snapshot_keeps_frozen_buffers_for_endpoint_restore():
    from scripts.qualify_coordinate_thin_student import actor_state
    module=torch.nn.Linear(3,2)
    module.register_buffer('rotary_frequency',torch.tensor([.1,.2]))
    saved=actor_state(module)
    with torch.no_grad():module.weight.add_(1);module.rotary_frequency.zero_()
    module.load_state_dict(saved,strict=True)
    assert torch.equal(module.rotary_frequency,torch.tensor([.1,.2]))
