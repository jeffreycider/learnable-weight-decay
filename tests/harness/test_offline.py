import torch
from chisel.offline import next_loss_gradient, ProbeStore
from chisel.harness import RunState

def test_chunked_task_gradient_matches_full_batch():
    model=torch.nn.Linear(4,3)
    parameter=model.weight.detach().clone().requires_grad_();params=dict(weight=parameter,bias=model.bias.detach())
    batch=dict(inputs=torch.randn(7,4),targets=torch.randint(3,(7,)))
    loss=torch.nn.functional.cross_entropy(torch.func.functional_call(model,params,(batch['inputs'],)),batch['targets'])
    expected=torch.autograd.grad(loss,parameter)[0]
    state=RunState(extras=dict(device=torch.device('cpu')))
    result,gradient=next_loss_gradient(model,params,dict(weight=parameter),batch,dict(name='torch.nn.functional.cross_entropy',kwargs={}),3,dict(runtime=dict(autocast=None)),state)
    torch.testing.assert_close(result,loss);torch.testing.assert_close(gradient['weight'],expected)
    assert state.extras['training_forwards']==3

def test_private_probe_and_adam_state_survive_reload(tmp_path):
    config=dict(seed=2,rank=2,optimizer=dict(_target_='torch.optim.Adam',lr=.001))
    store=ProbeStore(tmp_path,config,'cpu');key='a'*64;snapshot=dict(w=dict(raw=torch.randn(8,8)))
    probes,opt=store.load(key,snapshot)
    sum(p.square().sum() for p in probes.parameters()).backward();opt.step()
    store.save(key,0,probes,opt)
    restored=ProbeStore(tmp_path,config,'cpu');restored.load_state_dict(store.state_dict())
    actual,adam=restored.load(key,snapshot)
    for a,b in zip(actual.parameters(),probes.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert all(s['step']==1 for s in adam.state.values())
