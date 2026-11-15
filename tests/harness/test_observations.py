import torch
from chisel.observations import Observations

def test_hooks_preserve_gradients_and_accumulate_rms():
    model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Linear(4,2))
    data=torch.randn(5,3)
    expected=torch.autograd.grad(model(data).square().sum(),list(model.parameters()))
    sensor=Observations(model,r'0.weight');sensor.active=True
    for part in data.split(2):model(part).square().sum().backward()
    result=sensor.finalize(1)
    torch.testing.assert_close(result['0.weight']['input'],data.square().mean(0).sqrt())
    assert result['0.weight']['adjoint'].shape==(4,)
    for p,g in zip(model.parameters(),expected):torch.testing.assert_close(p.grad,g)
    sensor.close()
