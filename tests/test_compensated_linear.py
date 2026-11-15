import pytest
import torch
from unittest.mock import patch
from rewriter.anchored_linear import AnchoredLinear
from rewriter.compensated_linear import compensated_linear,bf16_six_product_linear


def test_compensated_forward_requires_explicit_backward_and_cuda():
    with pytest.raises(ValueError,match='custom backward'):
        compensated_linear(torch.ones(2,3,requires_grad=True),torch.ones(4,3))
    with pytest.raises(ValueError,match='CUDA'):
        compensated_linear(torch.ones(2,3),torch.ones(4,3))


def test_six_product_rejects_unqualified_storage_without_cuda_import():
    with pytest.raises(TypeError,match='FP32'):
        bf16_six_product_linear(torch.ones(2,3,dtype=torch.bfloat16),torch.ones(4,3,dtype=torch.bfloat16))
    with pytest.raises(ValueError,match='CUDA'):
        bf16_six_product_linear(torch.ones(2,3),torch.ones(4,3))


def test_six_product_expansion_preserves_small_fp32_edits_before_gpu_rounding():
    # Algebraic oracle, not a GPU error certificate: summation here is FP64.
    generator=torch.Generator().manual_seed(81)
    x=torch.randn(23,37,generator=generator)
    w=torch.randn(31,37,generator=generator)
    edit=1e-5*torch.randn(31,37,generator=generator)
    def pieces(v):
        high=v.bfloat16().float();residual=v-high
        middle=residual.bfloat16().float();low=(residual-middle).bfloat16().float()
        return high.double(),middle.double(),low.double()
    def six(v):
        a,b,c=pieces(x);d,e,f=pieces(v)
        return a@d.T+b@d.T+a@e.T+c@d.T+b@e.T+a@f.T
    ref=x.double()@(w+edit).double().T-x.double()@w.double().T
    response=six(w+edit)-six(w)
    assert (response-ref).norm()/ref.norm()<.003
    # A plain single BF16 cast is deliberately not this arithmetic.
    single=x.bfloat16().double()@((w+edit).bfloat16().double()-w.bfloat16().double()).T
    assert (single-ref).norm()/ref.norm()>.1


def test_optional_linear_forward_keeps_thin_vjp_and_default_unchanged():
    torch.manual_seed(18)
    x=torch.randn(2,3,requires_grad=True);w=torch.randn(4,3);bias=torch.randn(4)
    p=torch.randn(4,2,requires_grad=True);q=torch.randn(3,2,requires_grad=True)
    a=torch.randn(4,2);b=torch.randn(3,2);v=torch.randn(4,requires_grad=True)
    scale=torch.tensor(.1,requires_grad=True);args=(x,w,bias,p,q,a,b,v,scale)
    ref=AnchoredLinear.apply(*args);cotangent=torch.randn_like(ref)
    expected=torch.autograd.grad(ref,(x,p,q,v,scale),cotangent)
    with patch('rewriter.compensated_linear.compensated_linear',side_effect=lambda x,w,b:torch.nn.functional.linear(x,w,b)) as call:
        actual=AnchoredLinear.apply(*args,'tf32x3')
    call.assert_called_once();assert torch.equal(actual,ref)
    observed=torch.autograd.grad(actual,(x,p,q,v,scale),cotangent)
    for got,want in zip(observed,expected):assert torch.equal(got,want)
