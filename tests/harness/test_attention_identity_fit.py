import pytest
import torch
from baseline.model import GPT
from chisel.network import Chisel, PrivateProbes, BANKS
from chisel.attention_identity_fit import propose
from chisel.transfer import apply_edits, bound_edits
from chisel.preservation import group_norm_change


def example(device='cpu', dtype=torch.float64, causal=True):
    torch.manual_seed(72)
    model = GPT(32, 2, 16, head_dim=8, causal=causal, learned_qk_scale=True).to(device=device, dtype=dtype)
    base = {n: p.detach().clone() for n,p in model.named_parameters()}
    snapshot = {n: {bank: (p.clone() if bank == 'raw' else torch.zeros_like(p)) for bank in BANKS}
                for n,p in base.items() if n.startswith('blocks.') and p.ndim == 2}
    actor = Chisel(2,16,1,head_dim=4,causal=False).to(device=device,dtype=dtype)
    probes = PrivateProbes({n:p['raw'].shape for n,p in snapshot.items()},2).to(device=device,dtype=dtype)
    return model,base,snapshot,actor,probes


@pytest.mark.parametrize('causal',[True,False])
@pytest.mark.parametrize('parameterization',['bounded_symmetric','exponential_consolidated'])
def test_finite_edits_preserve_logits_with_nonzero_bias(causal, parameterization):
    model,base,snapshot,actor,probes = example(causal=causal)
    with torch.no_grad(): actor.proj.weight.normal_(std=.2)
    edits = propose(actor,probes,snapshot,base,8,parameterization)
    bounded = bound_edits(snapshot,edits,1e-8)
    for n in edits:
        torch.testing.assert_close(bounded[n]['raw'],edits[n]['raw'],rtol=0,atol=0)
        torch.testing.assert_close(bounded[n]['consolidated'],edits[n]['consolidated'],rtol=0,atol=0)
        if parameterization == 'exponential_consolidated':
            assert edits[n]['raw'].count_nonzero() == 0
    effective = apply_edits(snapshot,edits)[2]
    inputs = torch.randint(0,32,(3,7))
    before = torch.func.functional_call(model,base,(inputs,))
    after = torch.func.functional_call(model,base|effective,(inputs,))
    torch.testing.assert_close(after,before,rtol=1e-6,atol=2e-7) # GPT casts logits to FP32.
    for n in effective:
        if not (n.endswith('.attn.v.weight') or n.endswith('.attn.proj.weight')):
            torch.testing.assert_close(effective[n],base[n],atol=0,rtol=0)


@pytest.mark.parametrize('parameterization',['bounded_symmetric','exponential_consolidated'])
def test_zero_heads_learn_norm_reduction_without_analytic_initialization(parameterization):
    model,base,snapshot,actor,probes = example()
    parameters = list(actor.parameters())+list(probes.parameters())
    optimizer = torch.optim.Adam(parameters,lr=.01)
    for step in range(30):
        optimizer.zero_grad()
        effective = apply_edits(snapshot,propose(actor,probes,snapshot,base,8,parameterization))[2]
        loss = group_norm_change(snapshot,effective)
        loss.backward()
        if step==0:
            assert loss.abs()<1e-15
            assert actor.proj.weight.grad.norm()>0
        optimizer.step()
    assert loss < -.001
    assert probes.write_left[2].grad.norm()>0
    # A remote MLP's observations influence this attention rewrite through global attention.
    bank = 'consolidated' if parameterization == 'exponential_consolidated' else 'raw'
    before = propose(actor,probes,snapshot,base,8,parameterization)['blocks.0.attn.v.weight'][bank]
    snapshot['blocks.1.mlp.fc.weight']['gradient'].normal_()
    after = propose(actor,probes,snapshot,base,8,parameterization)['blocks.0.attn.v.weight'][bank]
    assert (after-before).norm()>1e-10


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('snapshot_metadata',[False,True])
@pytest.mark.parametrize('parameterization',['bounded_symmetric','exponential_consolidated'])
def test_cuda_compilation_preservation_gradient_and_reuse(snapshot_metadata,parameterization):
    from torch._dynamo.utils import counters
    torch.set_float32_matmul_precision('highest')
    model,base,snapshot,actor,probes = example('cuda',torch.float32)
    if snapshot_metadata:
        for i,block in enumerate(model.blocks):
            name=f'blocks.{i}.attn.v.weight'
            snapshot[name]['attention_head_dim']=block.attn.head_dim
            snapshot[name]['attention_value_bias']=base[name.replace('weight','bias')]
    with torch.no_grad():actor.proj.weight.normal_(std=.03)
    compiled = torch.compile(propose,fullgraph=True,dynamic=False)
    params=list(actor.parameters())+list(probes.parameters())
    def compute(fn):
        args=(actor,probes,snapshot) if snapshot_metadata else (actor,probes,snapshot,base,8)
        effective=apply_edits(snapshot,fn(*args,parameterization=parameterization))[2]
        loss=group_norm_change(snapshot,effective)
        gradients=torch.autograd.grad(loss,params,allow_unused=True)
        return effective,gradients
    eager,eg=compute(propose); actual,ag=compute(compiled)
    for n in eager:torch.testing.assert_close(actual[n],eager[n],rtol=1e-5,atol=1e-6)
    for a,b in zip(ag,eg):
        if b is None:assert a is None
        else:torch.testing.assert_close(a,b,rtol=1e-3,atol=2e-6)
    count=counters['stats']['unique_graphs']
    for _ in range(3):
        with torch.no_grad():actor.proj.bias.add_(.0001)
        compute(compiled)
    assert counters['stats']['unique_graphs']==count
    inputs=torch.randint(0,32,(3,7),device='cuda')
    before=torch.func.functional_call(model,base,(inputs,))
    after=torch.func.functional_call(model,base|actual,(inputs,))
    torch.testing.assert_close(after,before,rtol=3e-5,atol=2e-6)
