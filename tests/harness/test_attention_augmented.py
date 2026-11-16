import copy
import pytest
import torch
from chisel.network import Chisel, propose
from chisel.attention_identity_fit import propose as attention_only
from chisel.attention_stage import expanded_weights
from chisel.transfer import apply_edits
from chisel.preservation import group_norm_change
from test_attention_identity_fit import example


def fixture(device='cpu', dtype=torch.float64):
    model, base, snapshot, old, probes = example(device, dtype)
    with torch.no_grad():
        old.proj.weight.normal_(std=.03)
    actor = Chisel(2,16,1,head_dim=4,causal=False,attention_branch=True,
                   full_rank_gains=True,normalize_gains=True).to(device=device,dtype=dtype)
    actor.load_state_dict(expanded_weights(actor,old.state_dict()))
    for i,block in enumerate(model.blocks):
        name=f'blocks.{i}.attn.v.weight'
        snapshot[name]['attention_head_dim']=block.attn.head_dim
        snapshot[name]['attention_value_bias']=base[name.replace('weight','bias')]
    return model,base,snapshot,old,actor,probes


def test_expansion_preserves_fitted_action_and_has_no_frozen_parameters():
    model,base,snapshot,old,actor,probes=fixture()
    a=attention_only(old,probes,snapshot);b=propose(actor,probes,snapshot)
    for name in a:
        for bank in a[name]:torch.testing.assert_close(b[name][bank],a[name][bank],rtol=0,atol=0)
    effective=apply_edits(snapshot,b)[2]
    loss=group_norm_change(snapshot,effective)
    loss.backward()
    assert all(p.requires_grad for p in actor.parameters())
    for head in [actor.proj,actor.gain_proj,actor.attention_proj]:
        assert head.weight.grad.norm()>0
    assert actor.proj.bias.grad.view(2,-1).norm(dim=1).min()>0


def test_independent_edits_include_mlp_and_consolidated_banks():
    _,_,snapshot,_,actor,probes=fixture()
    baseline=propose(actor,probes,snapshot)
    with torch.no_grad():actor.proj.bias.view(2,2,2)[1,0,0]=.5
    changed=propose(actor,probes,snapshot)
    for name,item in changed.items():
        torch.testing.assert_close(item['raw'],baseline[name]['raw'],rtol=0,atol=0)
        assert item['consolidated'].norm()>0
    with torch.no_grad():actor.proj.bias.view(2,2,2)[0,0,0]=.25
    raw=propose(actor,probes,snapshot)
    name='blocks.1.mlp.fc.weight'
    assert (raw[name]['raw']-changed[name]['raw']).norm()>0
    torch.testing.assert_close(raw[name]['consolidated'],changed[name]['consolidated'],rtol=0,atol=0)


def test_attention_preservation_at_nonzero_consolidated_weights():
    model,base,snapshot,_,actor,probes=fixture()
    for item in snapshot.values():
        item['consolidated']=item['raw']*.4
        item['raw']=item['raw']*.6
    effective=apply_edits(snapshot,propose(actor,probes,snapshot))[2]
    inputs=torch.randint(0,32,(3,7))
    before=torch.func.functional_call(model,base,(inputs,))
    after=torch.func.functional_call(model,base|effective,(inputs,))
    torch.testing.assert_close(after,before,rtol=1e-6,atol=2e-7)


def test_unrestricted_branch_has_actual_logit_loss_gradients():
    model,base,snapshot,_,actor,probes=fixture()
    with torch.no_grad():actor.proj.weight.normal_(std=.02)
    params=list(actor.parameters())+list(probes.parameters())
    effective=apply_edits(snapshot,propose(actor,probes,snapshot))[2]
    inputs=torch.randint(0,32,(3,7))
    reference=torch.func.functional_call(model,base,(inputs,)).detach()
    actual=torch.func.functional_call(model,base|effective,(inputs,))
    loss=(actual-reference).double().square().mean()
    gradient=torch.autograd.grad(loss,params)
    assert loss>0
    assert all(torch.isfinite(g).all() for g in gradient)
    index=next(i for i,p in enumerate(params) if p is actor.proj.weight)
    assert gradient[index].norm()>0


def test_reject_wrong_source_architecture():
    _,_,_,old,actor,_=fixture()
    bad=copy.deepcopy(old.state_dict());bad['unexpected']=torch.zeros(1)
    with pytest.raises(ValueError,match='unexpected'):expanded_weights(actor,bad)
    with pytest.raises(ValueError,match='attention-only'):expanded_weights(actor,actor.state_dict())


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_compiled_combined_gradients_and_graph_reuse():
    from torch._dynamo.utils import counters
    torch.set_float32_matmul_precision('highest')
    _,_,snapshot,_,actor,probes=fixture('cuda',torch.float32)
    with torch.no_grad():actor.proj.weight.normal_(std=.001)
    compiled=torch.compile(propose,fullgraph=True,dynamic=False)
    def compute(fn):
        edits=fn(actor,probes,snapshot)
        effective=apply_edits(snapshot,edits)[2]
        loss=group_norm_change(snapshot,effective)
        grads=torch.autograd.grad(loss,list(actor.parameters())+list(probes.parameters()))
        return edits,grads
    expected,eg=compute(propose);actual,ag=compute(compiled)
    for n in expected:
        for bank in expected[n]:torch.testing.assert_close(actual[n][bank],expected[n][bank],rtol=1e-4,atol=2e-6)
    for a,b in zip(ag,eg):torch.testing.assert_close(a,b,rtol=1e-3,atol=2e-5)
    count=counters['stats']['unique_graphs']
    for _ in range(3):
        with torch.no_grad():actor.proj.bias.add_(.00001)
        compute(compiled)
    assert counters['stats']['unique_graphs']==count


def test_ordinary_mixed_objective_and_exact_resume(tmp_path):
    from test_attention_identity_cohort import recipe
    from chisel.harness import run
    cfg=recipe(tmp_path)
    cfg['setup']=[h for h in cfg['setup'] if h['name']!='chisel.attention_identity_cohort.setup']
    cfg['model']['attention_branch']=True
    cfg.update(proposal_function='chisel.network.propose',train_steps=4)
    cfg['setup'].append(dict(name='chisel.attention_stage.audit_initial'))
    expected=run(cfg)
    import json
    initial=json.loads((tmp_path/'fit/audits/train_audit_00000000.json').read_text())
    assert len(initial['checkpoints'])==4
    assert all(row['input_origin']=='recorded' and row['maximum_logit_change']==0
               for row in initial['checkpoints'])
    assert expected.extras['training_forwards']>0
    assert expected.model.attention_proj.weight.norm()>0
    assert expected.model.proj.weight.norm()>0
    cfg['output_dir']=str(tmp_path/'prefix');cfg['stop_step']=2
    run(cfg)
    cfg['resume']=str(tmp_path/'prefix/latest_rank0.pt')
    cfg['output_dir']=str(tmp_path/'resumed');cfg['stop_step']=4
    actual=run(cfg)
    for name,value in expected.model.state_dict().items():
        torch.testing.assert_close(actual.model.state_dict()[name],value,rtol=0,atol=0)
    for identity,p in actual.extras['private_probes'].latest.items():
        a=torch.load(p,weights_only=False)['probes']
        b=torch.load(expected.extras['private_probes'].latest[identity],weights_only=False)['probes']
        for name,value in b.items():torch.testing.assert_close(a[name],value,rtol=0,atol=0)
