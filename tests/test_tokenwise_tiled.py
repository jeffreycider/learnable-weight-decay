import pytest
import torch
from rewriter.tokenwise_reference import TokenwiseConfig,logit_terms_and_cotangent as reference
from rewriter.tokenwise_tiled import logit_terms_and_cotangent as tiled

@pytest.mark.parametrize('backward,balance',[(True,False),(True,True),(False,False),(False,True)])
@pytest.mark.parametrize('multiplier',[0.,.37])
def test_tiled_tokenwise_matches_reference_with_all_diagnostics(backward,balance,multiplier):
    torch.manual_seed(15);t=torch.randn(2,3,37).double().log_softmax(-1);s=torch.randn(2,3,37)
    cfg=TokenwiseConfig(mean_tolerance=.01,tail_tolerance=.02,scale=.1,rho=.01,ranking_weight=3.7)
    inputs=torch.randint(37,(2,3));kwargs=dict(config=cfg,multiplier=multiplier,panel_tokens=11,chunk_tokens=2,backward=backward,balance=balance,inputs=inputs)
    ga,sa,ka=tiled(s,t,**kwargs);gr,sr,kr=reference(s,t,**kwargs)
    torch.testing.assert_close(ka,kr,atol=5e-14,rtol=1e-12)
    torch.testing.assert_close(sa,sr,atol=1e-10,rtol=1e-6)
    if backward:torch.testing.assert_close(ga,gr,atol=1e-7,rtol=3e-6)
    else:assert ga is None and gr is None


def test_explicit_backend_and_strict_resume_identity():
    from dataclasses import asdict,replace
    from rewriter.tokenwise_reference import TokenwiseDual,validate_objective_resume
    config=TokenwiseConfig(.01,.01,.1,.01)
    legacy=asdict(config);legacy.pop('loss_backend')
    validate_objective_resume({'tokenwise_objective':legacy},config)
    accelerated=replace(config,loss_backend='tiled_fp64_v1')
    with pytest.raises(ValueError):validate_objective_resume({'tokenwise_objective':legacy},accelerated)
    dual=TokenwiseDual([1],config=config,dual_lr=.1);state=dual.state_dict();state['config'].pop('loss_backend');dual.load_state_dict(state)
    with pytest.raises(ValueError):TokenwiseDual([1],config=accelerated,dual_lr=.1).load_state_dict(state)
    with pytest.raises(ValueError):replace(config,loss_backend='automatic')


def test_explicit_backend_dispatch_and_default_unchanged():
    from dataclasses import replace
    torch.manual_seed(123);student=torch.randn(2,3,37);teacher=torch.randn_like(student).double().log_softmax(-1)
    config=TokenwiseConfig(.01,.01,.1,.01,ranking_weight=2.)
    kwargs=dict(multiplier=.3,panel_tokens=6,backward=True)
    expected=tiled(student,teacher,config=config,**kwargs)
    got=reference(student,teacher,config=replace(config,loss_backend='tiled_fp64_v1'),**kwargs)
    for a,b in zip(got,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
