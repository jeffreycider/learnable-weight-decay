import pytest
import torch
from rewriter.adaptive_action import AdaptiveAction, AdaptiveFactors
from rewriter.adaptive_edit_target import edit_target_terms
from scripts.fixed_frame_target import FixedFrameTarget


@pytest.mark.parametrize('near_target',[False,True])
def test_cached_terms_match_dense_and_uncached_core_gradients(near_target):
    torch.manual_seed(127)
    u=torch.randn(11,3,dtype=torch.float64)
    v=torch.randn(8,2,dtype=torch.float64)
    c=torch.randn(3,2,dtype=torch.float64,requires_grad=True)
    b=torch.randn(7,dtype=torch.float64,requires_grad=True)
    if near_target:
        t=AdaptiveFactors(u,c.detach().clone()+1e-10,v)
        tb=b.detach().clone()+1e-10
    else:
        t=AdaptiveFactors(torch.randn(11,2,dtype=torch.float64),torch.randn(2,4,dtype=torch.float64),torch.randn(8,4,dtype=torch.float64))
        tb=torch.zeros_like(b)
    action=AdaptiveAction({'W':AdaptiveFactors(u,c,v)},{'b':b})
    target=AdaptiveAction({'W':t},{'b':tb})
    cached=FixedFrameTarget(action,target)
    actual=sum(cached(action).values())
    uncached=sum(edit_target_terms(action,target).values())
    dense=(u@c@v.T-t.dense()).square().sum()+(b-tb).square().sum()
    torch.testing.assert_close(actual,uncached,rtol=1e-12,atol=1e-25)
    torch.testing.assert_close(actual,dense,rtol=1e-5 if near_target else 1e-12,atol=1e-25)
    cg=torch.autograd.grad(actual,(c,b),retain_graph=True)
    ug=torch.autograd.grad(uncached,(c,b),retain_graph=True)
    dg=torch.autograd.grad(dense,(c,b))
    for x,y,z in zip(cg,ug,dg):
        torch.testing.assert_close(x,y,rtol=1e-10,atol=1e-18)
        torch.testing.assert_close(x,z,rtol=1e-4 if near_target else 1e-12,atol=1e-12)


def test_cache_rejects_changed_or_trainable_frames_and_copies_target():
    u=torch.eye(4,dtype=torch.float64)[:,:2];v=u.clone()
    c=torch.ones(2,2,dtype=torch.float64,requires_grad=True)
    t=torch.zeros(2,2,dtype=torch.float64)
    a=AdaptiveAction({'w':AdaptiveFactors(u,c,v)}, {})
    target=AdaptiveAction({'w':AdaptiveFactors(u,t,v)}, {})
    cache=FixedFrameTarget(a,target)
    t.add_(3)
    torch.testing.assert_close(cache(a)['w'],torch.tensor(4.,dtype=torch.float64))
    changed=AdaptiveAction({'w':AdaptiveFactors(u+1,c,v)}, {})
    with pytest.raises(ValueError,match='changed'):cache(changed)
    live=AdaptiveAction({'w':AdaptiveFactors(u.clone().requires_grad_(),c,v)}, {})
    with pytest.raises(ValueError,match='frozen'):cache(live)
    with pytest.raises(ValueError,match='frozen'):FixedFrameTarget(live,target)
