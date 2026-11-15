"""Graph-owned address storage qualification; CPU import remains supported."""
import pytest
import torch
from rewriter.graph_offline import PinnedAddressTables


def test_early_failed_checkpoint_verification_invalidates_replay():
    from types import SimpleNamespace
    from unittest.mock import Mock
    from rewriter.graph_offline import NativeThinGradientGraph
    graph=NativeThinGradientGraph.__new__(NativeThinGradientGraph)
    graph.binding_valid=True;graph.teacher_features=torch.empty(0)
    graph.graph=Mock()
    record=SimpleNamespace(assert_verified=Mock(side_effect=ValueError('bad checkpoint SHA')))
    with pytest.raises(ValueError,match='bad checkpoint SHA'):
        graph.load_checkpoint(None,record,None,None)
    assert not graph.binding_valid
    with pytest.raises(RuntimeError,match='verified graph binding'):graph.replay()
    graph.graph.replay.assert_not_called()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA graph and Triton norm')
def test_captured_norm_survives_pinned_allocator_churn_and_updates():
    from baseline.model import GPT
    from rewriter.accelerated_grouped_norm import AcceleratedGroupedNormObjective
    torch.manual_seed(55)
    model=GPT(128,2,128).cuda()
    base={n:v.detach().clone() for n,v in model.named_parameters()}
    grouped=AcceleratedGroupedNormObjective(base,zero_reference_policy='freeze')
    # Always preserve exact-zero readonly groups, including the initial biases.
    values={n:(v if n in grouped.frozen_parameter_names else (v+1e-5).requires_grad_()) for n,v in base.items()}
    leaves=[values[n] for n in grouped.names if values[n].requires_grad]
    def body():
        result=grouped.evaluate(values)
        return result.mean_relative_change.detach(),torch.autograd.grad(result.mean_relative_change,leaves)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):body()
    torch.cuda.current_stream().wait_stream(stream)
    addresses=PinnedAddressTables();grouped.address_table=addresses;graph=torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):output,gradients=body()
    finally:del grouped.address_table
    assert len(addresses.host_tables)==2
    for magnitude in (1e-5,2e-4,0.):
        with torch.no_grad():
            for name,value in values.items():
                if value.requires_grad:value.copy_(base[name]+magnitude)
        expected,expected_gradients=body()
        churn=[torch.full((4096,),-1,dtype=torch.int64,pin_memory=True) for _ in range(100)]
        del churn
        graph.replay()
        assert torch.equal(output,expected)
        for actual,reference in zip(gradients,expected_gradients):assert torch.equal(actual,reference)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA graph and Triton norm')
def test_dynamic_norm_capture_can_unfreeze_and_refreeze_initial_zero_group():
    """An initial frozen-name list must not become an unconditional graph assert."""
    from rewriter.accelerated_grouped_norm import AcceleratedGroupedNormObjective
    base={'w':torch.zeros(4,4,device='cuda')}
    values={'w':torch.zeros_like(base['w'],requires_grad=True)}
    groups={'weight':('w',)}
    grouped=AcceleratedGroupedNormObjective(base,groups,zero_reference_policy='freeze')
    masks={'w':torch.zeros((),dtype=torch.bool,device='cuda')}
    def body():
        result=grouped.evaluate(values,edit_masks=masks)
        return result.mean_relative_change.detach(),torch.autograd.grad(result.mean_relative_change,values['w'])[0]
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):body()
    torch.cuda.current_stream().wait_stream(stream)
    addresses=PinnedAddressTables();grouped.address_table=addresses;graph=torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):output,gradient=body()
    finally:del grouped.address_table
    for magnitude in (1.,0.,2.):
        with torch.no_grad():
            base['w'].fill_(magnitude)
            values['w'].fill_(magnitude*.99)
            masks['w'].fill_(magnitude>0)
            reference=AcceleratedGroupedNormObjective(base,groups,zero_reference_policy='freeze')
            grouped.reference_vector.copy_(reference.reference_vector)
        # Fresh reference uses the ordinary non-dynamic production validator.
        expected=reference.evaluate(values).mean_relative_change
        expected_gradient,=torch.autograd.grad(expected,values['w'])
        graph.replay()
        assert torch.equal(output,expected.detach())
        assert torch.equal(gradient,expected_gradient)
