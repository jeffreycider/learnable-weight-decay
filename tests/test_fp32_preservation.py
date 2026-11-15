import copy
import dataclasses

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from baseline.model import GPT
from rewriter.function_preservation import FunctionalPreservation, gpt_logits
from rewriter.gpt_precision import RMS_EPSILON, precision_metadata
from rewriter.teacher_cache import write_panel, TeacherCacheReader, bind_engine
from tests.test_transformer_chisel import model as chisel_model, flatten
from tests.test_controller import make_snapshot


def engine():
    torch.manual_seed(187)
    return FunctionalPreservation(GPT(vocab_size=32,num_layers=2,model_dim=128),
                                  kl_dtype=torch.float64,gpt_precision='fp32')


def test_fp32_matches_unmodified_baseline_equations():
    e=engine();inputs=torch.randint(32,(1,7));observed=[]
    for module in e.view.gpt.modules():
        if isinstance(module,(torch.nn.Linear,torch.nn.Embedding)):
            module.register_forward_hook(lambda m,a,o:observed.append(o.dtype))
    actual=e._forward(e.parameters,inputs)
    with sdpa_kernel(SDPBackend.MATH):expected=gpt_logits(e.view.gpt,inputs)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert observed and set(observed)=={torch.float32}
    assert {p.dtype for p in e.parameters.values()}=={torch.float32}


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
@pytest.mark.parametrize('shape', [(2, 3, 128), (2, 3, 2, 128)])
def test_baseline_default_epsilon_is_opmath_float32(dtype, shape):
    # Small values distinguish the two epsilon choices after BF16 rounding.
    # The reference deliberately does not use or monkeypatch RMS_EPSILON.
    x=torch.linspace(-.003,.003,torch.tensor(shape).prod().item()).reshape(shape).to(dtype)
    actual=F.rms_norm(x,(128,))
    expected=F.rms_norm(x,(128,),eps=2.0**-23)
    incorrect=F.rms_norm(x,(128,),eps=2.0**-7)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert (actual.float()-incorrect.float()).abs().max()>.1
    assert RMS_EPSILON==2.0**-23


def test_checkpoint_promotion_never_mutates_source_or_erases_embedding_edits():
    original=GPT(vocab_size=32,num_layers=2,model_dim=128)
    params={n:p.detach().clone() for n,p in original.named_parameters()}
    buffers={n:p.detach().clone() for n,p in original.named_buffers()}
    before=params['embed.weight'].clone();e=engine();e.load_teacher_state(params,buffers)
    constructed=FunctionalPreservation(copy.deepcopy(original),parameter_state=params,buffer_state=buffers,gpt_precision='fp32')
    torch.testing.assert_close(constructed.parameters['embed.weight'],before.float(),rtol=0,atol=0)
    assert params['embed.weight'].dtype==torch.bfloat16
    assert torch.equal(params['embed.weight'],before)
    assert e.parameters['embed.weight'].dtype==torch.float32
    tiny=e.parameters['embed.weight']+1e-5
    assert (tiny!=e.parameters['embed.weight']).all()
    assert e.parameters['embed.weight'].data_ptr()!=params['embed.weight'].data_ptr()


def test_matching_fp32_cache_zero_kl_and_gradient(tmp_path):
    e=engine();inputs=torch.randint(32,(1,7));sha='a'*64
    write_panel(tmp_path,1,'train',e,inputs,[13],state_sha256=sha,execution={'gpt_precision':'fp32'})
    panel=TeacherCacheReader(tmp_path,require_complete=False).read(1,'train')
    assert panel.features.dtype==torch.float32
    teacher=panel.teacher_batch(e,0,1)
    assert e.student_parameter_loss(e.parameters,teacher).item()==0
    direction=torch.randn_like(e.parameters['proj.weight'])
    def loss(t):
        params=dict(e.parameters);params['proj.weight']=params['proj.weight']+t*direction
        return e.student_parameter_loss(params,teacher)
    t=torch.tensor(.002,requires_grad=True);value=loss(t);value.backward()
    assert value>0 and torch.isfinite(t.grad) and t.grad!=0
    eps=1e-5;numeric=(loss(t.detach()+eps)-loss(t.detach()-eps))/(2*eps)
    torch.testing.assert_close(t.grad.double(),numeric,rtol=.002,atol=1e-6)
    bad=dataclasses.replace(panel,metadata={**panel.metadata,'precision':{**precision_metadata('fp32'),'rms_epsilon':1e-7}})
    with pytest.raises(ValueError,match='epsilon'):bad.teacher_batch(e,0,1)
    old=dataclasses.replace(panel,metadata={**panel.metadata,'precision':{**precision_metadata('fp32'),'rms_epsilon':2.0**-7}})
    with pytest.raises(ValueError,match='epsilon'):old.teacher_batch(e,0,1)


def test_old_bfloat_cache_rejected_even_when_head_weights_match(tmp_path):
    e=engine();baseline=FunctionalPreservation(copy.deepcopy(e.view.gpt),kl_dtype=torch.float64)
    inputs=torch.randint(32,(1,7));sha='b'*64
    write_panel(tmp_path,1,'train',baseline,inputs,[9],state_sha256=sha,execution={})
    bind_engine(e,sha);panel=TeacherCacheReader(tmp_path,require_complete=False).read(1,'train')
    with pytest.raises(ValueError,match='precision'):panel.teacher_batch(e,0,1)


def test_chisel_action_outputs_are_fp32_with_trainable_zero_heads():
    actor=chisel_model();snapshot=make_snapshot();action=actor(snapshot)
    values=flatten(action);assert values.dtype==torch.float32 and values.count_nonzero()==0
    action.contraction(snapshot.weights,snapshot.vector_weights).backward()
    assert all(head.weight.grad.norm()>0 for head in actor.output_heads)
