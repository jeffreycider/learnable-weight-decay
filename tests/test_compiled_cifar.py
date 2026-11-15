import pytest
import torch

from rewriter.cifar_adapter import CifarPreservation
from rewriter.compiled_cifar import CompiledCifarForward
from tests.test_cifar_adapter import TinyAirbench


def engine(seed):
    torch.manual_seed(seed)
    return CifarPreservation(TinyAirbench().eval(), microbatch_images=4)


def test_one_compilation_reuses_changed_model_tensors_and_preserves_full_parameter_vjp():
    torch.set_num_threads(1)
    compilations=[]
    def compiler(gm, example_inputs):
        compilations.append(gm)
        return gm.forward
    first,second=engine(17),engine(19)
    compiled=CompiledCifarForward(first,backend=compiler)
    images=torch.randn(4,3,8,8).contiguous(memory_format=torch.channels_last)
    outputs=[]
    for current in (first,second,first):
        actual_parameters={n:p.detach().clone().requires_grad_(True) for n,p in current.parameters.items()}
        reference_parameters={n:p.detach().clone().requires_grad_(True) for n,p in current.parameters.items()}
        actual=compiled(current,actual_parameters,images)
        expected=current._forward(reference_parameters,images)
        weights=torch.randn_like(actual)
        ga=torch.autograd.grad((actual*weights).sum(),tuple(actual_parameters.values()))
        ge=torch.autograd.grad((expected*weights).sum(),tuple(reference_parameters.values()))
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        for a,b in zip(ga,ge):torch.testing.assert_close(a,b,rtol=0,atol=0)
        outputs.append(actual.detach())
    assert len(compilations)==1
    assert not torch.equal(outputs[0],outputs[1]) and torch.equal(outputs[0],outputs[2])


def test_layout_partial_batch_and_mutated_frozen_state_fail_before_execution():
    torch.set_num_threads(1)
    current=engine(1)
    compiled=CompiledCifarForward(current,backend='eager')
    images=torch.randn(4,3,8,8).contiguous(memory_format=torch.channels_last)
    compiled(current,current.parameters,images)
    with pytest.raises(ValueError,match='complete image microbatch'):
        compiled(current,current.parameters,images[:3])
    with pytest.raises(ValueError,match='image layout'):
        compiled(current,current.parameters,images.contiguous())
    next(iter(current.buffers.values())).add_(1)
    with pytest.raises(RuntimeError,match='mutated'):
        compiled(current,current.parameters,images)
