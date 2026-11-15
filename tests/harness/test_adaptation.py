from copy import deepcopy
import pytest
import torch
from chisel.adaptation import fit, split_batch
from chisel.network import Chisel, PrivateProbes
from chisel.preservation import Dual
from chisel.preservation_fit import logits
from chisel.harness import RunState


def test_support_query_partition_is_disjoint_replayable_and_rng_isolated():
    batch = dict(inputs=torch.arange(20).reshape(10, 2), targets=torch.arange(10))
    rng = torch.get_rng_state().clone()
    support, query = split_batch(batch, 4, 3, 42)
    assert not set(support['targets'].tolist()) & set(query['targets'].tolist())
    assert torch.equal(torch.get_rng_state(), rng)
    again = split_batch(batch, 4, 3, 42)
    assert torch.equal(support['inputs'], again[0]['inputs'])
    assert torch.equal(query['inputs'], again[1]['inputs'])
    with pytest.raises(ValueError):
        split_batch(batch, 6, 5, 42)


def test_adaptation_changes_private_probes_only_and_never_reads_query_feedback():
    torch.manual_seed(4)
    actor = Chisel(2, 8, 1, head_dim=4, causal=False)
    torch.nn.init.normal_(actor.proj.weight, std=.02)
    model = torch.nn.Linear(4, 4, bias=False)
    base = dict(weight=model.weight.detach().clone())
    snapshot = dict(weight={key: base['weight'].clone() if key == 'raw' else torch.zeros_like(base['weight'])
                           for key in ('raw', 'consolidated', 'gradient', 'momentum', 'proposal')})
    probes = PrivateProbes(dict(weight=(4, 4)), 2)
    initial = deepcopy(probes.state_dict())
    actor_before = deepcopy(actor.state_dict())
    for parameter in actor.parameters():
        parameter.grad = torch.ones_like(parameter)
    state = RunState(model=actor, extras=dict(logits_forward=logits, device=torch.device('cpu'),
        training_forwards=17, dual=Dual(1., 1., 1e-10, 1e4)))
    config = dict(validation_adaptation=dict(steps=3, support_examples=4, query_examples=3,
        seed=9, private_lr=.001), radius_floor=1e-6, consolidated_penalty=.01, gradient_clip=1.,
        microbatch_size=2, runtime=dict(autocast=None),
        preservation=dict(top_k=4, margin_fraction=.5, logit_mse_weight=10., ranking_weight=3.))
    batch = dict(inputs=torch.randn(8, 4), targets=torch.arange(8))
    optimizer = torch.optim.Adam(probes.parameters(), lr=.001)
    query, _, metrics = fit(config, state, 'a'*64, model, base, snapshot, {}, {}, batch, probes, optimizer)
    assert len(query['inputs']) == 3 and metrics['adaptation_model_forwards'] == 12
    assert any(not torch.equal(initial[name], value) for name, value in probes.state_dict().items())
    assert state.extras['dual'].value == 1. and state.extras['training_forwards'] == 17
    for name, value in actor.state_dict().items():
        torch.testing.assert_close(value, actor_before[name], rtol=0, atol=0)
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in actor.parameters())
    # Alter only query inputs, then repeat from the original private probes.
    changed = deepcopy(batch)
    changed['inputs'][query['targets']] *= 100
    other = PrivateProbes(dict(weight=(4, 4)), 2)
    other.load_state_dict(initial)
    fit(config, state, 'a'*64, model, base, snapshot, {}, {}, changed, other,
        torch.optim.Adam(other.parameters(), lr=.001))
    for first, second in zip(probes.parameters(), other.parameters()):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
