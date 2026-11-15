import pytest
import torch
from rewriter.fused_relu_square import fused_relu_square, reference_relu_square


def _inputs(device, shape=(3, 5, 3072), scale=2.):
    torch.manual_seed(17)
    x = torch.randn(*shape, device=device) * scale
    x.view(-1)[:9] = torch.tensor([0., -0., 1e-30, -1e-30, 3e4, -3e4, float('nan'), float('inf'), float('-inf')], device=device)
    return x


def _same_bits(a, b):
    return torch.equal(a.view(torch.int32), b.view(torch.int32))  # NaN payloads and signed zeros included


@pytest.mark.parametrize('scale', [0.05, 2., 500.])
def test_fused_relu_square_bit_identical_to_eager_on_cpu(scale):
    torch.set_num_threads(1)
    x = _inputs('cpu', scale=scale)
    a, b = x.clone().requires_grad_(), x.clone().requires_grad_()
    za, zb = fused_relu_square(a), reference_relu_square(b)
    assert _same_bits(za, zb)
    g = torch.randn_like(x) * .7
    g.view(-1)[:2] = torch.tensor([-1., 1.])  # gradient at x == 0 / -0 must be exactly zero like relu's threshold backward
    za.backward(g); zb.backward(g)
    assert _same_bits(a.grad, b.grad)
    assert float(a.grad.view(-1)[0]) == 0. and float(a.grad.view(-1)[1]) == 0.
    assert torch.isnan(a.grad.view(-1)[6]) and torch.isnan(b.grad.view(-1)[6])  # NaN propagates like eager
    assert torch.isinf(a.grad.view(-1)[7]) and float(a.grad.view(-1)[8]) == 0.


def test_fused_relu_square_rejects_non_fp32_and_keeps_shape():
    with pytest.raises(ValueError, match='FP32'):
        fused_relu_square(torch.zeros(4, dtype=torch.bfloat16))
    x = torch.randn(2, 3, 8)
    assert fused_relu_square(x).shape == x.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA Triton kernels')
@pytest.mark.parametrize('shape', [(3, 5, 3072), (16, 1024, 3072)])
def test_fused_relu_square_cuda_matches_eager_bitwise_or_reports_ulps(shape):
    x = _inputs('cuda', shape=shape)
    a, b = x.clone().requires_grad_(), x.clone().requires_grad_()
    za, zb = fused_relu_square(a), reference_relu_square(b)
    g = torch.randn_like(x) * .7
    za.backward(g); zb.backward(g)
    # bit-exact is the pass condition (owner requires explicit 0 ULP; no xfail)
    assert torch.equal(za.view(torch.int32), zb.view(torch.int32)), 'forward not bit-identical to eager'
    assert torch.equal(a.grad.view(torch.int32), b.grad.view(torch.int32)), 'backward not bit-identical to eager'
