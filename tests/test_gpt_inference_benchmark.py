import struct
import numpy as np
import pytest
import torch
from scripts.benchmark_gpt_inference import first_past_sequences,teacher_log_probs,preservation_pipeline
from baseline.model import GPT
from rewriter.function_preservation import gpt_logits


def test_past_input_shape_and_ids(tmp_path):
    p=tmp_path/'fineweb_train_000001.bin';tokens=np.arange(65,dtype=np.uint16)
    p.write_bytes(struct.pack('<iii',20240520,1,65)+bytes(1012)+tokens.tobytes())
    inputs=first_past_sequences(p,4,16)
    assert inputs.dtype==torch.int32 and inputs.tolist()==np.arange(64).reshape(4,16).tolist()
    with pytest.raises(ValueError):first_past_sequences(p,5,16)


def test_unlimited_batch_pipeline_matches_dense_kl():
    torch.set_num_threads(1);torch.manual_seed(12)
    teacher=GPT(16,1,128).eval();student=GPT(16,1,128).eval();student.load_state_dict(teacher.state_dict())
    with torch.no_grad():student.proj.bias.add_(torch.linspace(-.01,.01,16))
    x=torch.randint(0,16,(4,3))
    with torch.no_grad():
        a=gpt_logits(teacher,x).double().log_softmax(-1);b=gpt_logits(student,x).double().log_softmax(-1)
        expected=(a.exp()*(a-b)).sum(-1).mean()
        actual=preservation_pipeline(teacher,student,x,chunk_tokens=2)
    torch.testing.assert_close(actual,expected,atol=1e-14,rtol=1e-12)
    assert not actual.requires_grad
    torch.testing.assert_close(teacher_log_probs(gpt_logits(teacher,x),2),a)
