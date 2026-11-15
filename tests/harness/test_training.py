import torch
from chisel.harness import RunState
from chisel.training import backward, optimizers

def test_microbatch_gradient_matches_full_batch():
    model=torch.nn.Linear(4,3)
    x=torch.randn(7,4);y=torch.randint(3,(7,))
    full=torch.nn.functional.cross_entropy(model(x),y)
    expected=torch.autograd.grad(full,list(model.parameters()))
    cfg=dict(microbatch_size=3,gradient_units='mean',runtime=dict(autocast=None),loss=dict(name='torch.nn.functional.cross_entropy',kwargs={}))
    state=RunState(model=model,train_batches=iter([(x,y)]),extras=dict(device=torch.device('cpu'),world_size=1,forward=model))
    backward(cfg,state)
    for p,g in zip(model.parameters(),expected):torch.testing.assert_close(p.grad,g)
    assert state.extras['training_forwards']==3

def test_optimizer_partition_is_total_and_disjoint():
    import pytest
    model=torch.nn.Linear(4,3)
    spec=dict(parameters='weight',tensor_lr=False,optimizer=dict(_target_='torch.optim.SGD',lr=.1))
    config=dict(optimizers=dict(first=spec))
    state=RunState(model=model,extras=dict(device='cpu'))
    with pytest.raises(ValueError,match='without'):optimizers()(config,state)
    config['optimizers']['second']=spec
    with pytest.raises(ValueError,match='overlapping'):optimizers()(config,state)
