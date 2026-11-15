import pytest
import torch

from rewriter.accelerated_kl import fused_teacher_kl
from rewriter.function_preservation import chunked_teacher_kl


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA Triton kernel')
@pytest.mark.parametrize('vocab', [17, 50304])
@pytest.mark.parametrize('magnitude', [0., 1e-6, 1e-3, .2])
def test_fused_kl_value_and_gradient(vocab, magnitude):
    torch.manual_seed(918)
    raw = torch.randn(2, 3, vocab, device='cuda') * 4
    teacher_logits = 15 * raw * (raw.square() + 225).rsqrt()
    logq = teacher_logits.double().log_softmax(-1)
    student = (teacher_logits + torch.randn_like(raw) * magnitude).requires_grad_()
    actual = fused_teacher_kl(student, logq)
    expected = chunked_teacher_kl(student, logq, 2)
    torch.testing.assert_close(actual, expected, atol=5e-14, rtol=1e-8)
    # A nonunit cotangent verifies the precise FP64 -> FP32 cast boundary.
    ga, = torch.autograd.grad(actual * 71.3, student, retain_graph=True)
    ge, = torch.autograd.grad(expected * 71.3, student)
    torch.testing.assert_close(ga, ge, atol=3e-11, rtol=1e-5)
