import pytest
import torch
from rewriter.ranking_loss import topk_margin_loss
from rewriter.tiled_ranking_loss import tiled_topk_margin_loss

@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA'))])
@pytest.mark.parametrize('k',[1,3,10])
def test_tiled_rank_values_and_weighted_gradients(device,k):
    torch.manual_seed(712)
    t=torch.randn(2,3,37,device=device,requires_grad=True)
    s=torch.randn_like(t,requires_grad=True)
    actual=tiled_topk_margin_loss(t,s,k=k);expected=topk_margin_loss(t,s,k=k)
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)
    weight=torch.tensor([[0.,1.,-2.],[3.,.1,4.]],device=device)
    ga,=torch.autograd.grad((actual*weight).sum(),s,retain_graph=True)
    ge,=torch.autograd.grad((expected*weight).sum(),s)
    torch.testing.assert_close(ga,ge,atol=2e-6,rtol=2e-6)
    assert t.grad is None

@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA'))])
def test_outsider_ties_share_the_gradient(device):
    t=torch.tensor([[5.,4.,3.,2.,1.]],device=device)
    s=torch.tensor([[0.,0.,6.,5.5,0.]],device=device,requires_grad=True)
    actual=tiled_topk_margin_loss(t,s,k=2);expected=topk_margin_loss(t,s,k=2)
    ga,=torch.autograd.grad(actual.sum(),s,retain_graph=True)
    ge,=torch.autograd.grad(expected.sum(),s)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    torch.testing.assert_close(ga,ge,atol=0,rtol=0)
    assert ga[0,2]==ga[0,3] and ga[0,2]>0

@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA'))])
def test_top10_microgap_crossings_keep_small_nonzero_cotangents(device):
    t=(.001-torch.arange(37,device=device)*1e-6).repeat(3,1)
    s=t.clone();s[:,5]=t[:,4]+1e-7;s[:,10]=t[:,9]+1e-7;s.requires_grad_()
    actual=tiled_topk_margin_loss(t,s,k=10);expected=topk_margin_loss(t,s,k=10)
    assert torch.all(expected>1e-14) and torch.all(actual>0)
    torch.testing.assert_close(actual,expected,atol=1e-20,rtol=3e-6)
    weight=torch.tensor([-3.,.01,100.],device=device)
    ga,=torch.autograd.grad((actual*weight).sum(),s,retain_graph=True)
    ge,=torch.autograd.grad((expected*weight).sum(),s)
    assert torch.equal(ga!=0,ge!=0)
    torch.testing.assert_close(ga,ge,atol=2e-14,rtol=3e-6)

@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA'))])
@pytest.mark.parametrize('teacher_log_probs',[False,True])
def test_fp64_arithmetic_microgaps_and_weighted_cotangents(device,teacher_log_probs):
    t=(1.-torch.arange(37,device=device)*1e-6).repeat(3,1)
    s=t.clone();s[:,5]=t[:,4]+1e-7;s[:,10]=t[:,9]+1e-7;s.requires_grad_()
    if teacher_log_probs:t=t.double().log_softmax(-1)
    actual=tiled_topk_margin_loss(t,s,k=10,arithmetic='fp64')
    expected=topk_margin_loss(t.double(),s.double(),k=10)
    assert actual.dtype==torch.float64 and torch.all(actual>0)
    torch.testing.assert_close(actual,expected,atol=1e-25,rtol=1e-12)
    w=torch.tensor([-3.,.01,100.],device=device,dtype=torch.float64)
    ga,=torch.autograd.grad((actual*w).sum(),s,retain_graph=True)
    ge,=torch.autograd.grad((expected*w).sum(),s)
    assert torch.equal(ga!=0,ge!=0)
    torch.testing.assert_close(ga,ge,atol=2e-14,rtol=3e-6)


@pytest.mark.parametrize('device',['cpu']+(['cuda'] if torch.cuda.is_available() else []))
def test_reusing_exact_teacher_selection_preserves_ties_and_weighted_gradient(device):
    # More tied maxima than k: topk[0] is not a valid replacement for argmax.
    teacher=torch.zeros(3,37,device=device,dtype=torch.float64)
    teacher[:,20:]=1.;student=torch.linspace(-.01,.01,37,device=device).repeat(3,1).requires_grad_()
    weights=torch.tensor([.01,-3.,100.],device=device,dtype=torch.float64)
    plain=tiled_topk_margin_loss(teacher,student,arithmetic='fp64')
    shared,(values,indices)=tiled_topk_margin_loss(teacher,student,arithmetic='fp64',return_teacher_topk=True)
    torch.testing.assert_close(shared,plain,atol=0,rtol=0)
    first=torch.autograd.grad((plain*weights).sum(),student)[0];second=torch.autograd.grad((shared*weights).sum(),student)[0]
    torch.testing.assert_close(first,second,atol=0,rtol=0)
    expected=teacher.topk(10,dim=-1)
    assert torch.equal(indices,expected.indices) and torch.equal(values,expected.values)
    assert not values.requires_grad
