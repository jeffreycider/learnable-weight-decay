import copy
import pytest
import torch
from rewriter.persistent_offline import export_torch_adam, PersistentActorOptimizer


def fixture(device):
    torch.manual_seed(812)
    actor=torch.nn.ParameterList([torch.nn.Parameter(torch.randn(11,7,device=device)),
                                  torch.nn.Parameter(torch.randn(13,device=device))])
    optimizer=torch.optim.Adam([{'params':[actor[1]],'lr':.001},
                                {'params':[actor[0]],'lr':.01}],betas=(.9,.99))
    for _ in range(3):
        for p in actor.parameters():p.grad=torch.randn_like(p)
        optimizer.step()
    return actor,optimizer


def test_saved_adam_export_preserves_reordered_groups_and_independent_buffers():
    actor,optimizer=fixture('cpu')
    state,rates,betas,eps=export_torch_adam(actor,optimizer)
    assert rates=={'0':.01,'1':.001} and int(state['step'])==3
    assert betas==(.9,.99) and eps==1e-8
    for i,p in enumerate(actor.parameters()):
        torch.testing.assert_close(state['exp_avg'][i],optimizer.state[p]['exp_avg'],rtol=0,atol=0)
        state['exp_avg'][i].zero_()
        assert torch.count_nonzero(optimizer.state[p]['exp_avg'])>0


@pytest.mark.skipif(not torch.cuda.is_available(),reason='cooperative optimizer CUDA kernel')
def test_literal_saved_adam_matches_reordered_group_continuation():
    actor,optimizer=fixture('cuda');actual=copy.deepcopy(actor)
    state,rates,betas,eps=export_torch_adam(actor,optimizer)
    gradients={n:torch.empty_like(p) for n,p in actual.named_parameters()}
    program=PersistentActorOptimizer(actual,gradients,optimizer_state=state,
                                     parameter_lrs=rates,betas=betas,eps=eps)
    for _ in range(3):
        for name,p in actor.named_parameters():
            p.grad=torch.randn_like(p);gradients[name].copy_(p.grad)
        torch.nn.utils.clip_grad_norm_(actor.parameters(),1.)
        optimizer.step();program.step();torch.cuda.synchronize()
        for index,(reference,observed) in enumerate(zip(actor.parameters(),actual.parameters())):
            torch.testing.assert_close(observed,reference,rtol=2e-6,atol=2e-7)
            torch.testing.assert_close(program.plan.moments[index],optimizer.state[reference]['exp_avg'],rtol=2e-6,atol=1e-7)
