import pytest
import torch
from chisel.fused_relu_square import fused_relu_square
from baseline.model import Block


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_fused_values_and_gradients(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA qualification')
    x=torch.randn(4097,device=device,requires_grad=True)
    with torch.no_grad():x[:7]=torch.tensor([-float('inf'),-0.,0.,float('inf'),float('nan'),1e-10,-1e-10],device=device)
    gradient=torch.randn_like(x)
    actual=fused_relu_square(x);expected=x.relu().square()
    torch.testing.assert_close(actual,expected,rtol=0,atol=0,equal_nan=True)
    a=torch.autograd.grad(actual,x,gradient)[0];b=torch.autograd.grad(expected,x,gradient)[0]
    torch.testing.assert_close(a,b,rtol=0,atol=0,equal_nan=True)


def test_fused_block_preserves_default_forward_and_gradient():
    torch.manual_seed(8);native=Block(16,head_dim=4)
    fused=Block(16,head_dim=4,activation_backend='fused_fp32');fused.load_state_dict(native.state_dict())
    x=torch.randn(2,8,16)
    a=native(x);b=fused(x)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    a.sum().backward();b.sum().backward()
    for left,right in zip(native.parameters(),fused.parameters()):torch.testing.assert_close(left.grad,right.grad,rtol=0,atol=0)
