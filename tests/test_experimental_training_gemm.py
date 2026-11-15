import pytest
import torch
from unittest.mock import patch
from rewriter.experimental_training_gemm import ExperimentalTrainingLinear,experimental_training_gemms


def test_full_gradient_ieee_matches_all_three_gradients():
    torch.manual_seed(151)
    x=torch.randn(2,7,5,requires_grad=True);w=torch.randn(11,5,requires_grad=True);b=torch.randn(11,requires_grad=True)
    reference=torch.nn.functional.linear(x,w,b);g=torch.randn_like(reference)
    expected=torch.autograd.grad(reference,(x,w,b),g)
    candidate=ExperimentalTrainingLinear.apply(x,w,b,'ieee','ieee')
    assert torch.equal(candidate,reference)
    actual=torch.autograd.grad(candidate,(x,w,b),g)
    for a,e in zip(actual,expected):torch.testing.assert_close(a,e,rtol=0,atol=0)


def test_full_weight_gradient_has_correct_transposed_geometry():
    torch.manual_seed(152)
    x=torch.randn(2,7,5,requires_grad=True);w=torch.randn(11,5,requires_grad=True)
    out=ExperimentalTrainingLinear.apply(x,w,None,'ieee','tf32x3');g=torch.randn_like(out)
    def multiply(a,b):
        assert a.shape==(11,14) and b.shape==(14,5)
        return 2*(a@b)
    with patch('rewriter.experimental_training_gemm.compensated_dx',side_effect=multiply) as call:
        dx,dw=torch.autograd.grad(out,(x,w),g)
    call.assert_called_once()
    assert torch.equal(dx,g@w)
    assert torch.equal(dw,2*(g.reshape(-1,11).T@x.reshape(-1,5)))


def test_instance_installation_restores_forward_and_parameter_identity_after_error():
    model=torch.nn.Sequential(torch.nn.Linear(5,7),torch.nn.ReLU(),torch.nn.Linear(7,3))
    original=[m.forward for m in model if isinstance(m,torch.nn.Linear)]
    addresses=[p.data_ptr() for p in model.parameters()]
    with pytest.raises(RuntimeError,match='intentional'):
        with experimental_training_gemms(model,dx_backend='ieee',dw_backend='ieee'):
            with pytest.raises(ValueError,match='nested'):
                with experimental_training_gemms(model):pass
            model(torch.ones(2,5)).sum().backward()
            raise RuntimeError('intentional')
    assert original==[m.forward for m in model if isinstance(m,torch.nn.Linear)]
    assert addresses==[p.data_ptr() for p in model.parameters()]
