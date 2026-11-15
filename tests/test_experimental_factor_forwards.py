import pytest
import torch
from baseline.model import GPT,Linear
from rewriter.function_preservation import _LogitsView
from rewriter.grouped_norm import GroupedNormObjective
from rewriter.experimental_factor_forwards import factor_forwards


@pytest.mark.parametrize('zero',[False,True])
def test_tiny_gpt_logits_loss_shared_actor_and_private_gradients(zero):
    torch.set_num_threads(1);torch.manual_seed(620)
    model=GPT(16,1,128).float().eval().requires_grad_(False)
    original={n:p.detach().clone() for n,p in model.named_parameters()}
    identities={n:id(p) for n,p in model.named_parameters()}
    buffers={n:b.clone() for n,b in model.named_buffers()}
    grouped=GroupedNormObjective(original,zero_reference_policy='freeze')
    modules=dict(model.named_modules())
    names=[n for n,p in model.named_parameters() if n.endswith('weight') and p.ndim==2]
    # One shared head produces all U/C/V/bias perturbations; private vectors
    # select per-matrix behavior. Nonlinear factor formation exercises actor VJP.
    actor=torch.nn.Linear(5,8).double()
    if zero:
        with torch.no_grad():actor.weight.zero_();actor.bias.zero_()
    private=torch.nn.Parameter(torch.randn(len(names),5,dtype=torch.float64))
    frames={}
    for name in names:
        stored=original[name];out_dim,in_dim=(stored.shape[::-1] if name=='embed.weight' else stored.shape)
        frames[name]=(torch.randn(out_dim,2,dtype=torch.float64),torch.randn(in_dim,2,dtype=torch.float64))
    inputs=torch.tensor([[1,2,1,3]])
    target=torch.tensor([3,1,2,0])
    view=_LogitsView(model,'fp32')
    def run(direct):
        generated=actor(private);scale=.03/(1+generated.square().sum()).sqrt()
        replacements={n:p for n,p in original.items()};factors={}
        for i,name in enumerate(names):
            ru,rv=frames[name];z=generated[i]
            u=torch.linalg.qr(ru+z[4],mode='reduced').Q
            v=torch.linalg.qr(rv+z[5],mode='reduced').Q
            c=z[:4].reshape(2,2)
            delta=u@c@v.T
            if name=='embed.weight':delta=delta.T
            replacements[name]=(original[name].double()+scale*delta).float()
            factors[name]=(u,c,v,scale)
            bias_name=name[:-6]+'bias'
            if bias_name in original:replacements[bias_name]=(original[bias_name].double()+scale*z[6]*torch.ones_like(original[bias_name],dtype=torch.float64)).float()
        # Exact realized dense norm term is retained as a separate graph.
        norm=100*grouped.evaluate(replacements).mean_relative_change
        def evaluate():return torch.func.functional_call(view,{'gpt.'+n:w for n,w in replacements.items()},(inputs,),strict=False)
        if direct:
            with factor_forwards(model,factors,chunk_tokens=3):logits=evaluate()
        else:logits=evaluate()
        loss=torch.nn.functional.cross_entropy(logits.reshape(-1,16),target)+norm
        gradients=torch.autograd.grad(loss,[*actor.parameters(),private])
        assert all(torch.isfinite(g).all() for g in gradients)
        return logits.detach(),loss.detach(),gradients
    expected,expected_loss,eg=run(False);actual,actual_loss,ag=run(True)
    assert torch.equal(actual,expected) and torch.equal(actual_loss,expected_loss)
    for a,e in zip(ag,eg):torch.testing.assert_close(a,e,rtol=2e-4,atol=2e-7)
    assert any(g.abs().max()>0 for g in ag)
    assert {n:id(p) for n,p in model.named_parameters()}==identities
    for n,p in model.named_parameters():assert torch.equal(p,original[n]) and p.grad is None
    for n,b in model.named_buffers():assert torch.equal(b,buffers[n])
    assert all('forward' not in m.__dict__ and not hasattr(m,'_factor_forward_active') for m in modules.values())


def test_context_repeated_module_use_nested_exception_and_owned_forward_restore():
    torch.set_num_threads(1);torch.manual_seed(621)
    layer=Linear(5,7).float().requires_grad_(False)
    old_forward=layer.forward;layer.forward=old_forward
    u=torch.randn(7,2,dtype=torch.float64,requires_grad=True)
    c=torch.randn(2,2,dtype=torch.float64,requires_grad=True)
    v=torch.randn(5,2,dtype=torch.float64,requires_grad=True)
    scale=torch.tensor(.02,dtype=torch.float64,requires_grad=True)
    x=torch.randn(3,5,requires_grad=True);base=layer.weight.detach().clone()
    def run(direct):
        w=(base.double()+scale*(u@c@v.T)).float()
        def call():
            first=torch.func.functional_call(layer,{'weight':w},(x,))
            second=torch.func.functional_call(layer,{'weight':w},(x*.3,))
            return (first+second).square().sum()
        if direct:
            with factor_forwards(layer,{'weight':(u,c,v,scale)},chunk_tokens=1):loss=call()
        else:loss=call()
        return torch.autograd.grad(loss,[x,u,c,v,scale])
    for a,e in zip(run(True),run(False)):torch.testing.assert_close(a,e,rtol=3e-5,atol=2e-6)
    with pytest.raises(RuntimeError,match='deliberate'):
        with factor_forwards(layer,{'weight':(u,c,v,scale)}):
            with pytest.raises(ValueError,match='nested'):
                with factor_forwards(layer,{'weight':(u,c,v,scale)}):pass
            raise RuntimeError('deliberate')
    assert layer.__dict__['forward'] is old_forward and not hasattr(layer,'_factor_forward_active')
    assert torch.equal(layer.weight,base)


def test_failed_entry_does_not_patch_any_module_or_remove_hooks():
    torch.set_num_threads(1)
    model=torch.nn.Sequential(Linear(3,4),torch.nn.Embedding(5,3,padding_idx=0))
    calls=[];handle=model[0].register_forward_hook(lambda *args:calls.append(1))
    factors={'0.weight':(torch.ones(4,2),torch.ones(2,2),torch.ones(3,2),torch.tensor(1.)),
             '1.weight':(torch.ones(3,2),torch.ones(2,2),torch.ones(5,2),torch.tensor(1.))}
    with pytest.raises(ValueError,match='embedding options'):
        with factor_forwards(model,factors):pass
    assert all('forward' not in m.__dict__ and not hasattr(m,'_factor_forward_active') for m in model.modules())
    model[0](torch.zeros(1,3));assert calls==[1]
    handle.remove()
