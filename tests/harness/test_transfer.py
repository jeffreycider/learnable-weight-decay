import torch
from chisel.transfer import apply_transition, bound_transfer, delayed_loss

def test_transfer_conserves_effective_weights_and_keeps_radius():
    r=torch.randn(4,6,dtype=torch.float64);c=torch.randn_like(r);d=torch.randn_like(r)*100
    snapshot={'w':dict(raw=r,consolidated=c)}
    delta=bound_transfer(snapshot,{'w':d},.01)['w']
    torch.testing.assert_close(r-delta+c+delta,r+c)
    assert (r-delta).norm()>=.01

def test_delayed_credit_matches_explicit_next_forward_and_finite_difference():
    torch.manual_seed(3)
    r=torch.randn(3,4,dtype=torch.float64);c=torch.randn_like(r)*.2;u=torch.randn_like(r)
    x=torch.randn(5,4,dtype=torch.float64);target=torch.randn(5,3,dtype=torch.float64)
    phi=torch.randn(3,2,dtype=torch.float64,requires_grad=True);right=torch.randn(4,2,dtype=torch.float64)
    snapshot={'w':dict(raw=r,consolidated=c)}
    def weights(p):
        delta=bound_transfer(snapshot,{'w':p@right.T},.001)
        return apply_transition(snapshot,delta,{'w':u},{'w':.02})
    effective,consolidated=weights(phi)
    loss=(x@effective['w'].T-target).square().mean()
    expected=torch.autograd.grad(loss,phi,retain_graph=True)[0]
    g=torch.autograd.grad(loss,effective['w'],retain_graph=True)[0]
    actual=torch.autograd.grad(delayed_loss(effective,consolidated,{'w':g},0),phi)[0]
    torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-10)
    direction=torch.randn_like(phi);epsilon=1e-5
    plus=(x@weights(phi+epsilon*direction)[0]['w'].T-target).square().mean()
    minus=(x@weights(phi-epsilon*direction)[0]['w'].T-target).square().mean()
    torch.testing.assert_close((plus-minus)/(2*epsilon),(expected*direction).sum(),rtol=1e-5,atol=1e-8)
