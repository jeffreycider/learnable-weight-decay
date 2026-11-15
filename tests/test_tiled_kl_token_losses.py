"""Per-token tails must use the complete row KL and its weighted cotangent."""
import pytest
import torch
from rewriter.tiled_teacher_kl import tiled_teacher_kl


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA'))])
@pytest.mark.parametrize('vocab', [17, 2049])
def test_token_kl_tail_gradient_matches_fp64_oracle(device, vocab):
    torch.manual_seed(91)
    teacher=torch.randn(2,3,vocab,device=device)*3
    student=(teacher+torch.randn_like(teacher)*.03).requires_grad_()
    rows=tiled_teacher_kl(student,teacher,reduction='none')
    logq=teacher.double().log_softmax(-1)
    oracle=(logq.exp()*(logq-student.double().log_softmax(-1))).sum(-1)
    assert rows.shape==(2,3) and rows.dtype==torch.float64
    torch.testing.assert_close(rows,oracle,atol=5e-14,rtol=1e-8)
    weights=torch.tensor([[0.,1.,-2.],[3.,.1,4.]],device=device,dtype=torch.float64)
    # A fixed threshold tail, unequal and negative oracle cotangents, and zero
    # rows exercise row scaling rather than accidentally broadcasting one scalar.
    actual=(rows*weights+70*(rows-.0002).clamp_min(0).square()).sum()
    expected=(oracle*weights+70*(oracle-.0002).clamp_min(0).square()).sum()
    ga,=torch.autograd.grad(actual,student,retain_graph=True)
    ge,=torch.autograd.grad(expected,student)
    torch.testing.assert_close(ga,ge,atol=3e-11,rtol=1e-5)
    mean=tiled_teacher_kl(student,teacher)
    torch.testing.assert_close(rows.mean(),mean,atol=0,rtol=0)


def test_token_kl_rejects_unknown_reduction():
    x=torch.zeros(1,3)
    with pytest.raises(ValueError,match='reduction'):
        tiled_teacher_kl(x,x,reduction='max')

@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA'))])
def test_exact_fp64_teacher_logprob_input_and_row_weights(device):
    torch.manual_seed(121)
    t=torch.randn(2,3,2049,device=device).double().log_softmax(-1)
    s=torch.randn(2,3,2049,device=device,requires_grad=True)
    rows=tiled_teacher_kl(s,t,reduction='none',teacher_is_log_probs=True)
    ref=(t.exp()*(t-s.double().log_softmax(-1))).sum(-1)
    torch.testing.assert_close(rows,ref,atol=5e-14,rtol=1e-12)
    w=torch.tensor([[0.,1.,-2.],[3.,.1,4.]],device=device,dtype=torch.float64)
    ga,=torch.autograd.grad((rows*w).sum(),s,retain_graph=True)
    ge,=torch.autograd.grad((ref*w).sum(),s)
    torch.testing.assert_close(ga,ge,atol=3e-11,rtol=1e-5)
