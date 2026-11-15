import pytest
import torch
from rewriter.fused_softcap import fused_softcap, reference_softcap


def _inputs(device, shape=(3, 5, 777), scale=4.):
    torch.manual_seed(11)
    x = torch.randn(*shape, device=device) * scale
    # include exact zeros, huge and tiny magnitudes, negative values
    x.view(-1)[:4] = torch.tensor([0., -0., 3e4, -3e4], device=device)
    x.view(-1)[4:8] = torch.tensor([1e-30, -1e-30, 1e-3, 123.456], device=device)
    return x


@pytest.mark.parametrize('scale', [0.1, 4., 300.])
def test_fused_softcap_is_bit_identical_to_eager_on_cpu(scale):
    torch.set_num_threads(1)
    x = _inputs('cpu', scale=scale)
    a = x.clone().requires_grad_()
    b = x.clone().requires_grad_()
    ya, yb = fused_softcap(a), reference_softcap(b)
    assert torch.equal(ya, yb)
    g = torch.randn_like(x) * 0.37
    ya.backward(g)
    yb.backward(g)
    assert torch.equal(a.grad, b.grad)


def test_fused_softcap_rejects_non_fp32_and_keeps_shape():
    with pytest.raises(ValueError, match='FP32'):
        fused_softcap(torch.zeros(3, dtype=torch.float64))
    x = torch.randn(2, 3, 4)
    assert fused_softcap(x).shape == x.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA Triton kernels')
@pytest.mark.parametrize('shape', [(3, 5, 777), (16, 1024, 50304)])
def test_fused_softcap_cuda_matches_eager_bitwise_or_reports_ulps(shape):
    x = _inputs('cuda', shape=shape)
    a = x.clone().requires_grad_()
    b = x.clone().requires_grad_()
    ya, yb = fused_softcap(a), reference_softcap(b)
    g = torch.randn_like(x) * 0.37
    ya.backward(g)
    yb.backward(g)
    # Bit-exact is the pass condition (owner measured 0 ULP at full shape; a nonzero ULP is a failure, not an xfail).
    assert torch.equal(ya.view(torch.int32), yb.view(torch.int32)), 'forward not bit-identical to eager'
    assert torch.equal(a.grad.view(torch.int32), b.grad.view(torch.int32)), 'backward not bit-identical to eager'
