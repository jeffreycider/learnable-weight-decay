import pytest
import torch
from chisel.compiled_preservation import configure, raw_logit_loss
from chisel.harness import RunState
from chisel.preservation_fit import logits, preservation_gradient


def config():
    return dict(compile_preservation_loss=True, measure_kl=False, measure_rankings=False,
        microbatch_size=3, runtime=dict(autocast=None), preservation=dict(
        constraint_metric='logit_mse', center_logits=False, top_k=4,
        margin_fraction=.5, logit_mse_weight=2., ranking_weight=0.))


def test_fused_objective_matches_chunked_cotangents_and_metrics():
    torch.manual_seed(14)
    model = torch.nn.Linear(4, 5).double()
    base = {n:p.detach().clone() for n,p in model.named_parameters()}
    inputs = torch.randn(7,4,dtype=torch.float64)
    results = []
    for fused in [False, True]:
        effective = {'weight': (base['weight'] + .01).requires_grad_()}
        state = RunState(extras=dict(device=torch.device('cpu'), logits_forward=logits))
        if fused:
            state.extras['raw_logit_loss'] = raw_logit_loss
        results.append(preservation_gradient(model, base, base|effective, effective,
            dict(inputs=inputs), config(), state, 3.))
    torch.testing.assert_close(results[0][0]['weight'], results[1][0]['weight'], rtol=1e-12, atol=1e-12)
    assert results[0][1].keys() == results[1][1].keys()
    for key in results[0][1]:
        assert results[0][1][key] == pytest.approx(results[1][1][key], rel=1e-12, abs=1e-15)


@pytest.mark.parametrize('override', [dict(measure_kl=True), dict(measure_rankings=True),
    dict(preservation=config()['preservation'] | dict(center_logits=True))])
def test_fused_loss_rejects_incompatible_objective(override):
    with pytest.raises(ValueError, match='raw MSE'):
        configure(config() | override, RunState())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_compiled_loss_transformer_parity_and_reuse():
    from chisel.models import CifarGPT
    from torch._dynamo.utils import counters
    torch.manual_seed(17)
    torch.set_float32_matmul_precision('highest')
    model = CifarGPT(32,2,4,'mean',head_dim=8,causal=False,
        learned_qk_scale=True,expansion=2).cuda().float()
    torch.nn.init.normal_(model.proj.weight, std=.1)
    base = {n:p.detach().clone() for n,p in model.named_parameters()}
    selected = {n:p for n,p in base.items() if n.startswith('blocks.') and p.ndim==2}
    cfg = config() | dict(microbatch_size=4)
    compiled = RunState(extras=dict(device=torch.device('cuda'),
        logits_forward=torch.compile(logits,fullgraph=True,dynamic=False)))
    configure(cfg, compiled)
    reference = RunState(extras=dict(device=torch.device('cuda'),
        logits_forward=compiled.extras['logits_forward']))
    def measure(state, delta):
        effective = {n:(p+delta).requires_grad_() for n,p in selected.items()}
        return preservation_gradient(model,base,base|effective,effective,
            dict(inputs=torch.ones(4,3,32,32,device='cuda')),cfg,state,3.)
    for delta in [0., .001]:
        a, ma = measure(reference, delta)
        b, mb = measure(compiled, delta)
        if delta:
            assert ma['logit_mse'] > 0 and any(g.abs().max() > 0 for g in a.values())
        for name in a:
            torch.testing.assert_close(a[name],b[name],rtol=3e-4,atol=3e-7)
        assert mb['logit_mse'] == pytest.approx(ma['logit_mse'],rel=3e-4,abs=1e-12)
        assert mb['maximum_logit_change'] == pytest.approx(ma['maximum_logit_change'],rel=3e-4,abs=3e-6)
    count = counters['stats']['unique_graphs']
    for delta in [.002,.003,.004]:
        measure(compiled, delta)
    assert counters['stats']['unique_graphs'] == count
