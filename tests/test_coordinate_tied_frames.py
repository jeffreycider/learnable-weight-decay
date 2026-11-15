"""Tied private-frame geometry and derivatives; no GPU/transfer qualification."""
import copy
import torch
import pytest
from torch import nn
from rewriter.coordinate_chisel import CoordinateChisel, CheckpointProbes, TiedOrthonormalFrame, TIED_FRAME_ARCHITECTURE
from tests.test_controller import make_snapshot
from tests.test_coordinate_private_invariants import permutations, permuted_snapshot

@pytest.fixture(autouse=True)
def threads():
    before=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def frame(n,l):
    return TiedOrthonormalFrame(n,l,generator=torch.Generator().manual_seed(91)).double()


@pytest.mark.parametrize('n,l',[(7,3),(3,3),(2,5),(1,4)])
def test_signed_qr_geometry_variance_and_adam_remains_valid(n,l):
    f=frame(n,l);opt=torch.optim.Adam(f.parameters(),lr=.01)
    assert list(dict(f.named_parameters()))==['raw']
    torch.testing.assert_close(f.raw.norm(dim=0 if n>=l else 1),
        torch.ones(min(n,l),dtype=torch.double),atol=1e-7,rtol=1e-7)
    for _ in range(3):
        q=f.orthonormal_matrix()
        gram=q.T@q if n>=l else q@q.T
        torch.testing.assert_close(gram,torch.eye(min(n,l),dtype=torch.double),atol=2e-14,rtol=2e-14)
        torch.testing.assert_close(f.encoder,f.decoder,atol=0,rtol=0)
        op=f.decode(torch.eye(l,dtype=torch.double))
        torch.testing.assert_close(op.square().sum(),torch.tensor(float(n),dtype=torch.double))
        if n<l:torch.testing.assert_close(op@op.T,torch.eye(n,dtype=torch.double))
        (op*torch.arange(n*l,dtype=torch.double).reshape(n,l)).sum().backward()
        opt.step();opt.zero_grad()
    # Positive rescaling of the raw chart changes neither tied map.
    original=f.orthonormal_matrix().detach()
    with torch.no_grad():f.raw.mul_(3.7)
    torch.testing.assert_close(f.orthonormal_matrix(),original,atol=2e-14,rtol=2e-14)


@pytest.mark.parametrize('n,l',[(6,3),(2,4)])
def test_qr_read_write_vjp_matches_finite_differences(n,l):
    class ReadWrite(nn.Module):
        def __init__(self):super().__init__();self.frame=frame(n,l)
        def forward(self,h,c):return self.frame.encode(h),self.frame.decode(c)
    module=ReadWrite();gen=torch.Generator().manual_seed(12)
    h=torch.randn(n,2,generator=gen,dtype=torch.double);c=torch.randn(l,2,generator=gen,dtype=torch.double)
    def fun(raw):return torch.func.functional_call(module,{'frame.raw':raw},(h,c))
    assert torch.autograd.gradcheck(fun,(module.frame.raw,),eps=1e-6,atol=2e-6,rtol=2e-5)


def actor_fixture(latents=2,mode='tied_orthonormal'):
    torch.manual_seed(41);s=make_snapshot()
    a=CoordinateChisel(rank=2,trajectory_dim=12,model_dim=128,num_layers=1,latent_tokens=latents,frame_mode=mode)
    p=a.make_private(s,state_key=('gpt',0,6),seed=11)
    return s,a,p


def test_default_unchanged_tied_identity_and_strict_private_mode():
    s,old,p=actor_fixture(mode='independent')
    torch.manual_seed(41)
    default=CoordinateChisel(rank=2,trajectory_dim=12,model_dim=128,num_layers=1,latent_tokens=2)
    assert default.configuration()==old.configuration()
    for k,v in old.state_dict().items():torch.testing.assert_close(default.state_dict()[k],v,atol=0,rtol=0)
    q,actor,private=actor_fixture()
    assert actor(q,private).norm_sq()==0
    loss=actor(q,private).contraction(q.weights,q.vector_weights);loss.backward()
    assert actor.matrix_output.weight.grad.norm()>0
    assert actor.configuration()['name']==TIED_FRAME_ARCHITECTURE
    assert private.get_extra_state()['frame_mode']=='tied_orthonormal'
    before={k:v.clone() for k,v in p.named_parameters()}
    with pytest.raises(ValueError,match='identity'):p.load_state_dict(private.state_dict())
    for k,v in p.named_parameters():torch.testing.assert_close(v,before[k],atol=0,rtol=0)
    with pytest.raises(ValueError,match='frame mode'):actor(q,p)
    same=actor.make_private(q,state_key=('gpt',0,6),seed=123)
    same.load_state_dict(copy.deepcopy(private.state_dict()))
    assert all(torch.equal(v,dict(same.named_parameters())[k]) for k,v in private.named_parameters())


def test_tall_coordinate_permutation_preserves_tokens_action_and_private_vjp():
    s,a,p=actor_fixture();b=copy.deepcopy(a);other=copy.deepcopy(p)
    with torch.no_grad():
        for head in a.output_heads:head.weight.normal_(std=.02)
        for block in a.blocks:
            block.attn.proj.weight.normal_(std=.01);block.mlp.proj.weight.normal_(std=.01)
    b.load_state_dict(a.state_dict());perm=permutations(s);changed=permuted_snapshot(s,perm)
    mapped={}
    with torch.no_grad():
        for i,name in enumerate(p.matrix_names):
            for side,index in [('matrix_rows',0),('matrix_columns',1)]:
                param=getattr(other,side)[i].raw;order=perm[0][name][index]
                param.copy_(param[order]);mapped[f'{side}.{i}.raw']=order
        for i,name in enumerate(p.vector_names):
            param=other.vector_frames[i].raw;order=perm[1][name]
            param.copy_(param[order]);mapped[f'vector_frames.{i}.raw']=order
    torch.testing.assert_close(a._inputs(s,p)[0],b._inputs(changed,other)[0],atol=2e-6,rtol=2e-5)
    x=a(s,p);y=b(changed,other);assert x.norm_sq()>1e-4
    for name,f in x.matrices.items():
        r,c=perm[0][name]
        torch.testing.assert_close(y.matrices[name].dense(),f.dense()[r][:,c],atol=3e-6,rtol=3e-5)
        torch.testing.assert_close(f.anchors.A.T@f.P,torch.zeros_like(f.anchors.A.T@f.P),atol=2e-6,rtol=0)
    for name,v in x.vectors.items():torch.testing.assert_close(y.vectors[name],v[perm[1][name]],atol=2e-6,rtol=2e-5)
    (x.contraction(s.weights,s.vector_weights)+.1*x.norm_sq()).backward()
    (y.contraction(changed.weights,changed.vector_weights)+.1*y.norm_sq()).backward()
    for name,v in p.named_parameters():
        g=dict(other.named_parameters())[name].grad
        expected=v.grad[mapped[name]] if name in mapped else v.grad
        assert (g is None)==(expected is None)
        if g is not None:torch.testing.assert_close(g,expected,atol=3e-5,rtol=3e-4)


def test_wide_qr_chart_is_not_misrepresented_as_row_permutation_equivariant():
    # Transposed QR is order dependent in its columns. This is a chart
    # convention, not a promised invariance of the row-orthonormal mode.
    f=frame(3,5);q=f.orthonormal_matrix().detach();order=torch.tensor([2,0,1])
    with torch.no_grad():f.raw.copy_(f.raw[order])
    assert not torch.allclose(f.orthonormal_matrix(),q[order],atol=1e-6,rtol=1e-6)


def test_tied_q_vjp_through_canonical_action_and_saturated_global_radius():
    from rewriter.algebra import Anchors,Factors
    torch.manual_seed(831)
    left,right=frame(7,3),frame(5,3)
    anchors=Anchors.random(7,5,2,generator=torch.Generator().manual_seed(3),dtype=torch.double)
    lp=torch.randn(3,2,dtype=torch.double);lq=torch.randn(3,2,dtype=torch.double)
    cot=torch.randn(7,5,dtype=torch.double)
    f=Factors.canonical(left.decode(lp),right.decode(lq),anchors)
    radius=torch.tensor(.01,dtype=torch.double)
    guard=torch.minimum(torch.ones_like(radius),radius/f.norm_sq().sqrt())
    assert guard<1
    cached=guard*f.contract(cot@anchors.B,cot.T@anchors.A)
    dense=f.dense();dense_guard=torch.minimum(torch.ones_like(radius),radius/dense.norm())
    explicit=(cot*(dense_guard*dense)).sum()
    params=(left.raw,right.raw)
    g=torch.autograd.grad(cached,params,retain_graph=True)
    h=torch.autograd.grad(explicit,params)
    torch.testing.assert_close(cached,explicit,atol=2e-15,rtol=2e-13)
    for a,b in zip(g,h):
        assert a.norm()>0
        torch.testing.assert_close(a,b,atol=2e-15,rtol=2e-12)


@pytest.mark.parametrize('mode',['independent','tied_orthonormal'])
def test_fresh_private_does_not_seed_cuda_or_consume_global_cpu_rng(monkeypatch,mode):
    s,a,_=actor_fixture(mode=mode)
    before=torch.random.get_rng_state().clone()
    def forbidden(*args,**kwargs):raise AssertionError('private initialization touched CUDA RNG')
    monkeypatch.setattr(torch.cuda,'manual_seed_all',forbidden)
    private=a.make_private(s,state_key=('gpt',0,7),seed=93)
    assert torch.equal(before,torch.random.get_rng_state())
    # Same CPU distribution and exact initialization as the former CPU seed.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(93+104729)
        expected=nn.Linear(private.trajectory_input.in_features,a.model_dim)
    torch.testing.assert_close(expected.weight,private.trajectory_input.weight,atol=0,rtol=0)
    torch.testing.assert_close(expected.bias,private.trajectory_input.bias,atol=0,rtol=0)
