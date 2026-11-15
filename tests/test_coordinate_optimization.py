import copy
import pytest
import torch
from rewriter.coordinate_optimization import joint_state_backward,clip_private_gradients


def test_one_vjp_matches_weighted_shared_unweighted_private_and_local_clocks():
    shared=torch.nn.Parameter(torch.tensor([.3,-.8],dtype=torch.float64))
    private=[torch.nn.Parameter(torch.tensor([1.,-.2],dtype=torch.float64)),torch.nn.Parameter(torch.tensor([-.5,.4],dtype=torch.float64))]
    optim=[torch.optim.Adam([p],lr=.01) for p in private]
    oracle_shared=torch.zeros_like(shared)
    for index,weight in enumerate([.125,.375]):
        loss=((shared*private[index]+shared.square())**2).sum()
        expected=torch.autograd.grad(loss,(shared,private[index]),retain_graph=True)
        oracle_shared.add_(expected[0],alpha=weight)
        other=private[1-index].detach().clone()
        joint_state_backward(loss,state_weight=weight,shared_parameters=[shared],private_parameters=[private[index]])
        torch.testing.assert_close(private[index].grad,expected[1],atol=1e-15,rtol=1e-15)
        before_shared=shared.grad.clone();clip_private_gradients([private[index]],.01)
        torch.testing.assert_close(shared.grad,before_shared,atol=0,rtol=0)
        optim[index].step();optim[index].zero_grad(set_to_none=True)
        torch.testing.assert_close(private[1-index],other,atol=0,rtol=0)
    torch.testing.assert_close(shared.grad,oracle_shared,atol=1e-15,rtol=1e-15)
    assert [int(o.state[p]['step']) for o,p in zip(optim,private)]==[1,1]


def test_local_update_does_not_depend_on_global_state_weight():
    phi=torch.nn.Parameter(torch.tensor([.6,-.3]));p=torch.nn.Parameter(torch.tensor([.2,.7]))
    states=[]
    for weight in [1.,1/16,1/200]:
        q=torch.nn.Parameter(p.detach().clone());optimizer=torch.optim.Adam([q],lr=.001)
        joint_state_backward(((phi*q)**2).sum(),state_weight=weight,shared_parameters=[phi],private_parameters=[q])
        clip_private_gradients([q],1.);optimizer.step();states.append(q.detach())
    for state in states[1:]:torch.testing.assert_close(state,states[0],atol=0,rtol=0)


def test_stale_private_gradient_and_overlapping_ownership_fail_before_backward():
    p=torch.nn.Parameter(torch.ones(2));q=torch.nn.Parameter(torch.ones(2));loss=(p*q).sum()
    with pytest.raises(ValueError,match='overlaps'):joint_state_backward(loss,state_weight=1,shared_parameters=[p],private_parameters=[p])
    q.grad=torch.zeros_like(q)
    with pytest.raises(ValueError,match='clear private'):joint_state_backward(loss,state_weight=1,shared_parameters=[p],private_parameters=[q])
    assert p.grad is None


def test_shared_transition_retains_moments_clocks_and_head_hyperparameters():
    from rewriter.coordinate_optimization import configure_shared_continuation,clip_shared_continuation
    head=torch.nn.Parameter(torch.tensor([1.,2.]));body=torch.nn.Parameter(torch.tensor([.2,.3]))
    optimizer=torch.optim.Adam([dict(params=[head],role='action_heads',lr=1e-7,lr_multiplier=1.),dict(params=[body],role='representation',lr=1e-6,lr_multiplier=10.)],eps=1e-8)
    (head.square().sum()+body.square().sum()).backward();optimizer.step()
    before=copy.deepcopy(optimizer.state_dict());configure_shared_continuation(optimizer,'separate_representation');after=optimizer.state_dict()
    for key,state in before['state'].items():
        for name,value in state.items():torch.testing.assert_close(after['state'][key][name],value,atol=0,rtol=0)
    assert after['param_groups'][0]==before['param_groups'][0]
    assert after['param_groups'][1]['lr']==1e-4 and after['param_groups'][1]['eps']==1e-12
    head.grad=torch.tensor([1000.,0.]);body.grad=torch.tensor([0.,.5])
    clip_shared_continuation(optimizer,'separate_representation')
    torch.testing.assert_close(body.grad,torch.tensor([0.,.5]),atol=0,rtol=0)
    torch.testing.assert_close(head.grad,torch.tensor([1.,0.]))


def test_effective_body_motion_distinguishes_master_only_changes():
    from baseline.model import RMSNorm
    from rewriter.coordinate_optimization import effective_bfloat16_motion
    module=torch.nn.Module();module.norm=RMSNorm(4);module.fp32_head=torch.nn.Linear(4,2)
    before={n:p.detach().clone() for n,p in module.named_parameters()}
    with torch.no_grad():module.norm.gains.add_(1e-5);module.fp32_head.weight.add_(1.)
    first=effective_bfloat16_motion(before,module)
    assert first['elements']==4 and first['changed_fraction']==0 and first['cast_change_rms']==0 and first['master_change_rms']>0
    with torch.no_grad():module.norm.gains.add_(.01)
    second=effective_bfloat16_motion(before,module)
    assert second['changed_fraction']==1 and second['cast_change_rms']>0
