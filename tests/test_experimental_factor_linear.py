import pytest
import torch
from rewriter.experimental_factor_linear import factor_linear,factor_embedding


@pytest.mark.parametrize('stored_dtype,factor_dtype',[(torch.float64,torch.float64),(torch.float32,torch.float32),(torch.float32,torch.float64)])
@pytest.mark.parametrize('guard',[False,True])
@pytest.mark.parametrize('chunk_tokens',[1,3,128])
@pytest.mark.parametrize('dimensions',[(17,13),(7,23)])
def test_dense_rounded_forward_and_all_input_factor_gradients(stored_dtype,factor_dtype,guard,chunk_tokens,dimensions):
    torch.set_num_threads(1);torch.manual_seed(601)
    out_dim,in_dim=dimensions
    x=torch.randn(2,5,in_dim,dtype=stored_dtype,requires_grad=True)
    base=torch.randn(out_dim,in_dim,dtype=stored_dtype)
    u=torch.randn(out_dim,3,dtype=factor_dtype,requires_grad=True)
    c=torch.randn(3,4,dtype=factor_dtype,requires_grad=True)
    v=torch.randn(in_dim,4,dtype=factor_dtype,requires_grad=True)
    raw_scale=torch.tensor(.07,dtype=factor_dtype,requires_grad=True)
    bias=torch.randn(out_dim,dtype=factor_dtype,requires_grad=True)
    inputs=[x,u,c,v,raw_scale,bias]
    def run(direct):
        # A differentiable global guard surrogate exercises indirect factor credit.
        scale=raw_scale/(1+u.square().sum()+c.square().sum()+v.square().sum()).sqrt() if guard else raw_scale
        weight=(base.to(factor_dtype)+scale*(u@c@v.T)).to(stored_dtype)
        if direct:y=factor_linear(x,weight.detach(),u,c,v,scale,bias,chunk_tokens=chunk_tokens)
        else:y=torch.nn.functional.linear(x,weight,bias.to(stored_dtype))
        cotangent=torch.linspace(-1,1,y.numel(),dtype=stored_dtype).reshape_as(y)
        grads=torch.autograd.grad(y,inputs,cotangent)
        return y.detach(),grads
    expected,eg=run(False);actual,ag=run(True)
    assert torch.equal(actual,expected)
    for a,e in zip(ag,eg):
        torch.testing.assert_close(a,e,rtol=2e-5 if stored_dtype==torch.float32 else 2e-12,atol=2e-6 if stored_dtype==torch.float32 else 2e-12)
    # The input gradient is precisely the same dense multiplication.
    assert torch.equal(ag[0],eg[0])


def test_no_dense_weight_credit_and_no_bias():
    x=torch.randn(2,5,requires_grad=True);u=torch.randn(7,2,requires_grad=True)
    c=torch.randn(2,2,requires_grad=True);v=torch.randn(5,2,requires_grad=True);s=torch.tensor(.1,requires_grad=True)
    w=(s*(u@c@v.T)).detach()
    y=factor_linear(x,w,u,c,v,s);y.square().sum().backward()
    assert w.grad is None and all(z.grad is not None for z in (x,u,c,v,s))
    with pytest.raises(ValueError,match='detached'):factor_linear(x,w.requires_grad_(),u,c,v,s)


def test_shared_head_and_learned_qr_frame_credit():
    torch.manual_seed(610);torch.set_num_threads(1)
    head=torch.nn.Linear(5,4).double()
    raw_u=torch.randn(19,2,dtype=torch.float64,requires_grad=True)
    raw_v=torch.randn(13,2,dtype=torch.float64,requires_grad=True)
    contexts=torch.randn(2,5,dtype=torch.float64)
    xs=torch.randn(2,7,13,dtype=torch.float32)
    base=torch.randn(19,13,dtype=torch.float32)
    parameters=[*head.parameters(),raw_u,raw_v]
    def run(direct):
        u=torch.linalg.qr(raw_u,mode='reduced').Q;v=torch.linalg.qr(raw_v,mode='reduced').Q
        cores=head(contexts).reshape(2,2,2)
        scale=.1/(1+cores.square().sum()).sqrt()
        ys=[]
        for x,c in zip(xs,cores):
            w=(base.double()+scale*(u@c@v.T)).float()
            ys.append(factor_linear(x,w.detach(),u,c,v,scale) if direct else torch.nn.functional.linear(x,w))
        y=torch.stack(ys)
        return y,torch.autograd.grad(y.square().mean(),parameters)
    expected,eg=run(False);actual,ag=run(True)
    assert torch.equal(actual,expected)
    for a,e in zip(ag,eg):torch.testing.assert_close(a,e,rtol=2e-5,atol=2e-7)


@pytest.mark.parametrize('chunk_tokens',[1,4,128])
@pytest.mark.parametrize('shape',[(11,5),(5,11)])
@pytest.mark.parametrize('stored_dtype,factor_dtype',[(torch.float64,torch.float64),(torch.float32,torch.float64),(torch.float32,torch.float32)])
def test_embedding_computational_orientation_repeated_ids_and_shared_credit(chunk_tokens,shape,stored_dtype,factor_dtype):
    torch.manual_seed(612);torch.set_num_threads(1)
    vocab,dim=shape
    ids=(torch.tensor([[0,1,1],[2,0,4],[1,3,0]])%vocab).T
    base=torch.randn(vocab,dim,dtype=stored_dtype)
    u=torch.randn(dim,2,dtype=factor_dtype,requires_grad=True)
    c=torch.randn(2,3,dtype=factor_dtype,requires_grad=True)
    v=torch.randn(vocab,3,dtype=factor_dtype,requires_grad=True)
    raw_scale=torch.tensor(.04,dtype=factor_dtype,requires_grad=True)
    parameters=[u,c,v,raw_scale]
    def run(direct):
        scale=raw_scale/(1+u.square().sum()+c.square().sum()+v.square().sum()).sqrt()
        weight=(base.to(factor_dtype)+scale*(u@c@v.T).T).to(stored_dtype)
        if direct:
            embedded=factor_embedding(ids,weight.detach(),u,c,v,scale,chunk_tokens=chunk_tokens)
            # The SAME parameters also feed an output projection: swapped factors
            # and transposed C map computational embedding to linear orientation.
            out=factor_linear(embedded,weight.detach(),v,c.T,u,scale,chunk_tokens=chunk_tokens)
            repeated=factor_embedding(ids[:1],weight.detach(),u,c,v,scale,chunk_tokens=chunk_tokens)
        else:
            embedded=torch.nn.functional.embedding(ids,weight)
            out=torch.nn.functional.linear(embedded,weight)
            repeated=torch.nn.functional.embedding(ids[:1],weight)
        loss=out.square().mean()+repeated.square().mean()
        return (embedded.detach(),out.detach()),torch.autograd.grad(loss,parameters)
    expected,eg=run(False);actual,ag=run(True)
    for a,e in zip(actual,expected):assert torch.equal(a,e)
    for a,e in zip(ag,eg):torch.testing.assert_close(a,e,rtol=3e-5 if stored_dtype==torch.float32 else 3e-12,atol=3e-6 if stored_dtype==torch.float32 else 3e-12)


@pytest.mark.parametrize('chunk_tokens',[0,-1,True,1.5])
def test_invalid_chunk_size_rejected(chunk_tokens):
    x=torch.ones(2,3);u=torch.ones(4,2);c=torch.ones(2,2);v=torch.ones(3,2);s=torch.tensor(1.)
    with pytest.raises(ValueError,match='positive integer'):factor_linear(x,torch.ones(4,3),u,c,v,s,chunk_tokens=chunk_tokens)
    with pytest.raises(ValueError,match='positive integer'):factor_embedding(torch.tensor([0]),torch.ones(3,4),u,c,v,s,chunk_tokens=chunk_tokens)


@pytest.mark.parametrize('embedding',[False,True])
def test_backward_reuses_core_credit_for_scale_without_extra_gemm(embedding):
    # Count executed CPU GEMMs, not inferred timings: two chunks require 7
    # contractions each for Linear (plus dx), or 5 each for Embedding.
    # Computing scale credit through token activations would add two more.
    torch.set_num_threads(1);torch.manual_seed(629)
    u=torch.randn(7,2,dtype=torch.float64,requires_grad=True)
    c=torch.randn(2,3,dtype=torch.float64,requires_grad=True)
    v=torch.randn(11,3,dtype=torch.float64,requires_grad=True)
    scale=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
    w=torch.randn(7,11,dtype=torch.float32)
    if embedding:
        y=factor_embedding(torch.tensor([1,2,1,3,1]),w.T,u,c,v,scale,chunk_tokens=3)
        parameters=[u,c,v,scale];expected=10
    else:
        x=torch.randn(5,11,requires_grad=True)
        y=factor_linear(x,w,u,c,v,scale,chunk_tokens=3)
        parameters=[x,u,c,v,scale];expected=15
    dy=torch.randn_like(y)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        gradients=torch.autograd.grad(y,parameters,dy)
    mm_count=sum(event.count for event in profile.key_averages() if event.key=='aten::mm')
    assert mm_count==expected
    # Zero scale must retain finite, nonzero scale credit without dividing by s.
    assert torch.isfinite(gradients[-1]) and gradients[-1].abs()>0
