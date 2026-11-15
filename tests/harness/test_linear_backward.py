import pytest
import torch
from baseline.model import Block
from chisel.linear_backward import linear


def test_compensated_backward_is_explicit_and_rejects_cpu():
    with pytest.raises(ValueError,match='CUDA FP32'):
        linear(torch.randn(2,3),torch.randn(4,3),torch.zeros(4))
    from chisel.offline_data import Replay
    with pytest.raises(ValueError,match='architecture'):
        Replay('cpu',model_options={'width':64})


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA numerical qualification')
def test_compensated_linear_forward_exact_and_gradient_close_with_compile_reuse():
    torch.set_float32_matmul_precision('highest');torch.manual_seed(12)
    x=torch.randn(3,17,48,device='cuda',requires_grad=True)
    w=torch.randn(64,48,device='cuda',requires_grad=True)
    b=torch.randn(64,device='cuda',requires_grad=True)
    cotangent=torch.randn(3,17,64,device='cuda')
    expected=torch.nn.functional.linear(x,w,b)
    reference=torch.autograd.grad(expected,(x,w,b),cotangent)
    forward=torch.compile(linear,fullgraph=True,dynamic=False)
    for _ in range(3):
        actual=forward(x,w,b)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        gradient=torch.autograd.grad(actual,(x,w,b),cotangent)
        for a,e in zip(gradient,reference):
            assert (a-e).norm()/e.norm()<3e-6
        if _==0:
            graphs=torch._dynamo.utils.counters['stats']['unique_graphs']
        assert torch._dynamo.utils.counters['stats']['unique_graphs']==graphs


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA numerical qualification')
def test_compensated_block_forward_and_backward_match_native():
    torch.set_float32_matmul_precision('highest');torch.manual_seed(14)
    native=Block(32,head_dim=8).cuda()
    alternative=Block(32,head_dim=8,linear_backward='tf32x3').cuda()
    alternative.load_state_dict(native.state_dict())
    x=torch.randn(2,16,32,device='cuda')
    expected=native(x);actual=alternative(x)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    expected.square().mean().backward();actual.square().mean().backward()
    for a,e in zip(alternative.parameters(),native.parameters()):
        assert (a.grad-e.grad).norm()/e.grad.norm().clamp_min(1e-20)<1e-5
