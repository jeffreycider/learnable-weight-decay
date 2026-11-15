import torch
from chisel.online import Consolidation
from chisel.muonh import MuonH
from chisel.recording import cpu_copy

def config():
    return dict(seed=4,model=dict(_target_='chisel.network.Chisel',rank=2,width=16,depth=1,head_dim=4,causal=False,expansion=2),optimizer=dict(_target_='torch.optim.Adam',lr=.001),radius_floor=1e-6,penalty=.01,gradient_clip=1)

def test_identity_and_pending_credit_resume():
    torch.manual_seed(11)
    p=torch.nn.Parameter(torch.randn(8,8));baseline=torch.nn.Parameter(p.detach().clone())
    opt=MuonH([p],lr=.018,mu=.95,ns_steps=12);base=MuonH([baseline],lr=.018,mu=.95,ns_steps=12)
    state=Consolidation({'weight':p},config())
    p.grad=torch.randn_like(p);baseline.grad=p.grad.clone()
    state.prepare(opt,{},True);opt.step();state.finish();base.step()
    torch.testing.assert_close(p,baseline,rtol=0,atol=0)
    resumed=Consolidation({'weight':p},config());resumed.load_state_dict(cpu_copy(state.state_dict()))
    p.grad=torch.randn_like(p)
    state.consume(1);resumed.consume(1)
    for a,b in zip(state.actor.parameters(),resumed.actor.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert state.actor.proj.weight.norm()>0

def test_nonzero_transfer_uses_same_executed_update_as_credit():
    p=torch.nn.Parameter(torch.randn(8,8));opt=MuonH([p],lr=.018,mu=.95,ns_steps=12)
    state=Consolidation({'weight':p},config())
    with torch.no_grad():state.actor.proj.weight.normal_(std=.03)
    p.grad=torch.randn_like(p);state.prepare(opt,{},True)
    expected=state.pending_graph[1]['weight'].detach().clone()
    opt.step();state.finish()
    torch.testing.assert_close(p,expected,rtol=1e-6,atol=1e-7)
