import torch
from chisel.muonh import MuonH, direction, sphere

def reference(g, m, p, lr, mu):
    # Literal PR267 unbatched arithmetic, including scalar/BF16 promotions.
    m.lerp_(g,1-mu)
    x=g.lerp(m,mu).bfloat16();t=x.size(-2)>x.size(-1)
    if t:x=x.mT
    x=x/(x.norm(dim=(-2,-1),keepdim=True)+1e-7)
    for _ in range(12):
        a=x@x.mT;b=-1.5*a+.5*a@a;x=2*x+b@x
    if t:x=x.mT
    x*=max(1,g.size(-2)/g.size(-1))**.5
    radius=p.norm();new=p-lr*x*radius/x.norm().clamp_min(1e-10)
    return new/new.norm().clamp_min(1e-10)*radius

def test_step_matches_record_and_keeps_raw_gradient():
    torch.manual_seed(4)
    for shape in [(8,12),(12,8),(8,8)]:
        ps=[torch.nn.Parameter(torch.randn(*shape)) for _ in range(3)]
        opt=MuonH(ps,lr=.018,mu=.95,ns_steps=12)
        ms=[torch.zeros_like(p) for p in ps]
        for _ in range(3):
            for p in ps:p.grad=torch.randn_like(p)
            gradients=[p.grad.clone() for p in ps]
            expected=[reference(p.grad,m,p.detach(),.018,.95) for p,m in zip(ps,ms)]
            opt.step()
            for p,g,w in zip(ps,gradients,expected):
                torch.testing.assert_close(p,w,rtol=1e-6,atol=2e-7)
                torch.testing.assert_close(p.grad,g,rtol=0,atol=0)

def test_optimizer_roundtrip():
    p=torch.nn.Parameter(torch.randn(4,8));p.grad=torch.randn_like(p)
    opt=MuonH([p],lr=.018,mu=.95,ns_steps=12);opt.step()
    clone=MuonH([p],lr=.018,mu=.95,ns_steps=12);clone.load_state_dict(opt.state_dict())
    torch.testing.assert_close(clone.state[p]['momentum'],opt.state[p]['momentum'])
