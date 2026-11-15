import pytest
import torch

from rewriter.teacher_model_staging import stage_fp32_teacher_model


@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float16,torch.float32,torch.float64])
def test_promotion_matches_existing_loader_bits_and_preserves_buffers(dtype):
    raw=torch.tensor([0.,-0.,1.,-2.,float('inf'),-float('inf'),float('nan')],dtype=dtype)
    model={'weight':raw,'gain':torch.arange(12,dtype=dtype).reshape(3,4).T,
           'float_buffer':raw.clone(),'integer_buffer':torch.arange(3)}
    result=stage_fp32_teacher_model(model,parameter_names={'weight','gain'})
    for name,value in model.items():
        expected=value.float() if name in {'weight','gain'} else value
        assert result[name].dtype==expected.dtype
        assert torch.equal(result[name].contiguous().view(torch.uint8),expected.contiguous().view(torch.uint8))
        assert result[name].data_ptr()!=value.data_ptr()
    assert result['gain'].stride()==model['gain'].stride()
    result['weight'].zero_()
    assert model['weight'][2]==1


def test_staged_teacher_matches_actual_fp32_loader_and_forward():
    from baseline.model import GPT
    from rewriter.function_preservation import FunctionalPreservation
    torch.manual_seed(38);torch.set_num_threads(1)
    engine=FunctionalPreservation(GPT(16,1,128),gpt_precision='fp32',microbatch_sequences=2)
    names=set(engine.parameters)
    # Stored checkpoint contains a mixture of BF16 matrices and FP32 vectors.
    raw={n:v.detach().to(torch.bfloat16 if v.ndim==2 else torch.float32).clone()
         for n,v in engine.parameters.items()}
    raw.update({n:v.detach().clone() for n,v in engine.buffers.items()})
    inputs=torch.arange(8).reshape(2,4)
    engine.load_teacher_state({n:raw[n] for n in names},{n:raw[n] for n in engine.buffers})
    reference={n:v.clone() for n,v in engine.parameters.items()}
    before=engine._forward(engine.parameters,inputs).detach().clone()
    staged=stage_fp32_teacher_model(raw,parameter_names=names)
    engine.load_teacher_state({n:staged[n] for n in names},{n:staged[n] for n in engine.buffers})
    assert all(torch.equal(reference[n],v) for n,v in engine.parameters.items())
    assert torch.equal(before,engine._forward(engine.parameters,inputs))


def test_rejects_live_parameters_and_incomplete_declaration():
    with pytest.raises(ValueError,match='nonempty'):
        stage_fp32_teacher_model({'p':torch.ones(2)},parameter_names=set())
    with pytest.raises(ValueError,match='exist'):
        stage_fp32_teacher_model({'p':torch.ones(2)},parameter_names={'missing'})
    with pytest.raises(ValueError,match='detached'):
        stage_fp32_teacher_model({'p':torch.ones(2,requires_grad=True)},parameter_names={'p'})
    with pytest.raises(ValueError,match='floating'):
        stage_fp32_teacher_model({'p':torch.ones(2,dtype=torch.int64)},parameter_names={'p'})


@pytest.mark.skipif(not torch.cuda.is_available(),reason='pinned allocator requires CUDA runtime')
def test_promoted_storage_remains_pinned_through_loader_float_conversion():
    result=stage_fp32_teacher_model({'p':torch.arange(32,dtype=torch.bfloat16)},
                                   parameter_names={'p'},pin_memory=True)
    assert result['p'].is_pinned()
    assert result['p'].float().data_ptr()==result['p'].data_ptr()
    assert result['p'].float().is_pinned()
