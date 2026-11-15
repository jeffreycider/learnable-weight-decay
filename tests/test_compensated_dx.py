import sys
from unittest.mock import patch

import pytest
import torch

from rewriter.anchored_linear import AnchoredLinear
from rewriter.compensated_dx import (
    compensated_dx, current_dx_backend, experimental_dx_backend,
)


def test_context_is_explicit_nested_and_exception_safe():
    assert current_dx_backend() == 'ieee'
    with pytest.raises(RuntimeError):
        with experimental_dx_backend():
            assert current_dx_backend() == 'tf32x3'
            with experimental_dx_backend('ieee'):
                assert current_dx_backend() == 'ieee'
            assert current_dx_backend() == 'tf32x3'
            raise RuntimeError('test cleanup')
    assert current_dx_backend() == 'ieee'
    with pytest.raises(ValueError):
        with experimental_dx_backend('tf32'):
            pass


def test_cpu_import_and_rejection_do_not_import_triton_kernel():
    assert 'rewriter.kernels.compensated_dx_triton' not in sys.modules
    with pytest.raises(ValueError, match='CUDA'):
        compensated_dx(torch.ones(3, 4), torch.ones(4, 2))
    with pytest.raises(TypeError, match='FP32'):
        compensated_dx(torch.ones(3, 4).double(), torch.ones(4, 2).double())
    with pytest.raises(ValueError, match='compatible'):
        compensated_dx(torch.ones(3, 4), torch.ones(3, 2))


def test_only_input_gradient_uses_captured_opt_in():
    torch.manual_seed(1)
    x = torch.randn(2, 3, requires_grad=True)
    w = torch.randn(5, 3)
    bias = torch.randn(5)
    p = torch.randn(5, 2, requires_grad=True)
    q = torch.randn(3, 2, requires_grad=True)
    a = torch.randn(5, 2)
    b = torch.randn(3, 2)
    v = torch.randn(5, requires_grad=True)
    scale = torch.tensor(.1, requires_grad=True)
    args = (x, w, bias, p, q, a, b, v, scale)
    upstream = torch.randn(2, 5)
    reference = AnchoredLinear.apply(*args)
    expected = torch.autograd.grad(reference, (x, p, q, v, scale), upstream)
    with experimental_dx_backend():
        actual = AnchoredLinear.apply(*args)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    # Context has ended before backward: it must retain its forward choice.
    with patch('rewriter.anchored_linear.compensated_dx', side_effect=lambda g,w: 2*(g@w)) as call:
        observed = torch.autograd.grad(actual, (x, p, q, v, scale), upstream)
    call.assert_called_once()
    torch.testing.assert_close(observed[0], 2*expected[0])
    for got, want in zip(observed[1:], expected[1:]):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_selective_cublas_dx_keeps_small_shapes_ieee():
    torch.manual_seed(4)
    x=torch.randn(2,3,requires_grad=True);w=torch.randn(5,3);bias=torch.randn(5)
    p=torch.randn(5,2,requires_grad=True);q=torch.randn(3,2,requires_grad=True)
    a=torch.randn(5,2);b=torch.randn(3,2);v=torch.randn(5,requires_grad=True);s=torch.tensor(.1,requires_grad=True)
    with experimental_dx_backend('cublas3_large'):
        out=AnchoredLinear.apply(x,w,bias,p,q,a,b,v,s)
    with patch('rewriter.compensated_linear.cublas_three_product_linear') as call:
        got=torch.autograd.grad(out,x,torch.ones_like(out))[0]
    call.assert_not_called()
    assert torch.equal(got,torch.ones_like(out)@w)

def test_selective_cublas_dx_long_k_includes_transpose():
    torch.manual_seed(4)
    x=torch.randn(2,3,requires_grad=True);w=torch.randn(3072,3);bias=torch.randn(3072)
    p=torch.randn(3072,2,requires_grad=True);q=torch.randn(3,2,requires_grad=True)
    a=torch.randn(3072,2);b=torch.randn(3,2);v=torch.randn(3072,requires_grad=True);s=torch.tensor(.1,requires_grad=True)
    with experimental_dx_backend('cublas3_large'):
        out=AnchoredLinear.apply(x,w,bias,p,q,a,b,v,s)
    def linear(g,weight):
        assert weight.is_contiguous()
        return g@weight.T
    with patch('rewriter.compensated_linear.cublas_three_product_linear',side_effect=linear) as call:
        got=torch.autograd.grad(out,x,torch.ones_like(out))[0]
    call.assert_called_once()
    assert torch.equal(got,torch.ones_like(out)@w)


def test_cached_dx_requires_current_parts_and_detects_mutation():
    torch.manual_seed(7)
    x=torch.randn(2,3,requires_grad=True);w=torch.randn(3072,3);bias=torch.randn(3072)
    p=torch.randn(3072,2,requires_grad=True);q=torch.randn(3,2,requires_grad=True)
    a=torch.randn(3072,2);b=torch.randn(3,2);v=torch.randn(3072,requires_grad=True);s=torch.tensor(.1,requires_grad=True)
    high=w.T.contiguous();low=torch.zeros_like(high)
    with experimental_dx_backend('cublas3_cached'):
        with pytest.raises(ValueError,match='current realized'):
            AnchoredLinear.apply(x,w,bias,p,q,a,b,v,s)
        out=AnchoredLinear.apply(x,w,bias,p,q,a,b,v,s,'ieee',high,low)
    def cached(g,weight,*,split_weight):
        assert split_weight[0] is high and split_weight[1] is low
        return g@weight.T
    with patch('rewriter.compensated_linear.cublas_three_product_linear',side_effect=cached) as call:
        got=torch.autograd.grad(out,x,torch.ones_like(out))[0]
    call.assert_called_once();assert torch.equal(got,torch.ones_like(out)@w)
    with experimental_dx_backend('cublas3_cached'):
        out=AnchoredLinear.apply(x,w,bias,p,q,a,b,v,s,'ieee',high,low)
    high.add_(1)
    with pytest.raises(RuntimeError,match='modified by an inplace'):
        torch.autograd.grad(out,x,torch.ones_like(out))
