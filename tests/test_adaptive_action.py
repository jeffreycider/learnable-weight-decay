import pytest
import torch

from rewriter.adaptive_action import AdaptiveAction, AdaptiveFactors


def factors(n=7, m=5, l=3):
    torch.manual_seed(63)
    return AdaptiveFactors(*(torch.randn(*shape, dtype=torch.float64, requires_grad=True)
                             for shape in ((n,l), (l,l), (m,l))))


@pytest.mark.parametrize('shape', [(7,5,3), (2,3,5)])
def test_exact_dense_norm_contraction_change_and_all_factor_gradients(shape):
    f = factors(*shape)
    weight = torch.randn(shape[0], shape[1], dtype=torch.float64)
    dense = f.dense()
    torch.testing.assert_close(f.norm_sq(), dense.square().sum())
    torch.testing.assert_close(f.contract(weight), (dense*weight).sum())
    torch.testing.assert_close(f.norm_change(weight), (dense+weight).square().sum()-weight.square().sum())
    expected = torch.autograd.grad((dense*weight).sum(), (f.U,f.C,f.V), retain_graph=True)
    cot = f.cotangents(weight)
    for actual, ref in zip((cot.U,cot.C,cot.V), expected):
        torch.testing.assert_close(actual, ref)
    actual = torch.autograd.grad(f.norm_change(weight), (f.U,f.C,f.V), retain_graph=True)
    expected = torch.autograd.grad((dense+weight).square().sum(), (f.U,f.C,f.V))
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b)


def test_batched_cotangents_sum_not_implicit_mean_and_reject_fixed_sensor_sketches():
    f = factors()
    h = torch.randn(4,7,5,dtype=torch.float64)
    torch.testing.assert_close(f.contract(h), (h*f.dense()).sum((-1,-2)))
    batched = f.cotangents(h)
    summed = f.cotangents(h.sum(0))
    for a,b in zip((batched.U,batched.C,batched.V), (summed.U,summed.C,summed.V)):
        torch.testing.assert_close(a.sum(0),b)
    with pytest.raises(ValueError, match='sketches'):
        f.contract((torch.randn(7,2),torch.randn(5,2)))


def test_radius_derivative_and_qr_chain_match_explicit_dense_finite_difference():
    f=factors()
    u,c,v=f.U,f.C,f.V
    h=torch.randn(7,5,dtype=torch.float64)
    def compressed(u,c,v):
        action=AdaptiveFactors(torch.linalg.qr(u).Q,c,torch.linalg.qr(v).Q)
        kappa=torch.minimum(c.new_tensor(1.),c.new_tensor(.13)/action.norm_sq().sqrt())
        return action.scaled(kappa).contract(h)
    def dense(u,c,v):
        delta=torch.linalg.qr(u).Q@c@torch.linalg.qr(v).Q.T
        kappa=torch.minimum(c.new_tensor(1.),c.new_tensor(.13)/delta.norm())
        return (kappa*delta*h).sum()
    assert torch.autograd.gradcheck(compressed,(u,c,v),eps=1e-6,atol=1e-5,rtol=1e-4)
    for a,b in zip(torch.autograd.grad(compressed(u,c,v),(u,c,v)),torch.autograd.grad(dense(u,c,v),(u,c,v))):
        torch.testing.assert_close(a,b)


def test_global_action_norm_contraction_and_scale_vectors_and_matrices():
    f=factors();vector=torch.randn(4,dtype=torch.float64,requires_grad=True)
    action=AdaptiveAction({'embedding_canonical_orientation':f},{'bias':vector})
    h=torch.randn(7,5,dtype=torch.float64);hv=torch.randn(4,dtype=torch.float64)
    torch.testing.assert_close(action.norm_sq(),f.dense().square().sum()+vector.square().sum())
    torch.testing.assert_close(action.contraction({'embedding_canonical_orientation':h},{'bias':hv}),
                               (f.dense()*h).sum()+(vector*hv).sum())
    kappa=torch.tensor(.2,dtype=torch.float64,requires_grad=True)
    scaled=action.scaled(kappa)
    assert scaled.family=='adaptive_ucv_v1'
    torch.testing.assert_close(scaled.dense()['bias'],kappa*vector)
    torch.testing.assert_close(scaled.norm_sq(),kappa.square()*action.norm_sq())
    assert torch.autograd.grad(scaled.norm_sq(),kappa)[0]>0


def test_frame_span_can_move_beyond_old_fixed_anchors():
    # With fixed A=B=e0, the old family has zero (1,1) entry. A learned
    # rank-one U=V=e1 realizes it exactly. This is a changed action space.
    u=torch.tensor([[0.],[1.],[0.]],requires_grad=True)
    f=AdaptiveFactors(u,torch.ones(1,1),u)
    assert f.dense()[1,1]==1
    old_a=torch.tensor([[1.],[0.],[0.]])
    assert torch.count_nonzero(old_a.T@f.dense())==0
    assert torch.count_nonzero(f.dense()@old_a)==0
