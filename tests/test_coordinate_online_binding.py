import copy

import pytest
import torch

from rewriter.coordinate_chisel import CoordinateChisel
from rewriter.coordinate_online_binding import CoordinateOnlineBinding
from rewriter.transformer_online import TransformerOnlineLifecycle
from tests.test_transformer_action import fixture


def make(radius=0.001, run_id='test-live'):
    base, snapshot = fixture()
    torch.manual_seed(251)
    shared = CoordinateChisel(rank=2, trajectory_dim=12, model_dim=128,
                              num_layers=1, latent_tokens=2,
                              frame_mode='tied_orthonormal')
    private = shared.make_private(snapshot, state_key=('gpt', 0, 499), seed=617)
    with torch.no_grad():
        for head in shared.output_heads:
            head.weight.normal_(std=.05)
    rng = torch.get_rng_state().clone()
    binding = CoordinateOnlineBinding.from_fitted(shared, private, snapshot, run_id=run_id)
    assert torch.equal(rng, torch.get_rng_state())
    optimizer = torch.optim.Adam([
        dict(params=binding.shared.parameters(), lr=1e-5),
        dict(params=binding.private.parameters(), lr=1e-4),
    ], betas=(.9, .99))
    lifecycle = TransformerOnlineLifecycle(binding, optimizer, global_token_count=128,
                                          final_action_step=502, radius_fraction=radius,
                                          gradient_clip=1e6,
                                          policy_provenance=binding.get_extra_state())
    return lifecycle, {n:p.clone().requires_grad_() for n,p in base.items()}, snapshot, shared, private


@pytest.mark.parametrize('radius', [0.0001, 100.])
def test_delayed_credit_matches_full_autograd_for_shared_and_private(radius):
    online, live, snapshot, source_shared, source_private = make(radius)
    originals = [p.detach().clone() for p in (*source_shared.parameters(), *source_private.parameters())]
    online.commit(snapshot, live, action_step=499)
    proposal = online.action.materialize()
    targets = {n:torch.sin(p.detach()) for n,p in live.items()}
    parameters = tuple(online.actor.parameters())
    expected = torch.autograd.grad(sum(.5*(p-targets[n]).square().sum() for n,p in proposal.items()),
                                   parameters, retain_graph=True, allow_unused=True)
    task = {n:(p.detach()-targets[n])*128 for n,p in live.items()}
    online.consume(task, gradient_step=500)
    for p,g in zip(parameters, expected):
        if g is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(p.grad, g, rtol=3e-5, atol=2e-6)
    assert any(p.grad is not None and p.grad.norm()>0 for p in online.actor.shared.parameters())
    assert any(p.grad is not None and p.grad.norm()>0 for p in online.actor.private.parameters())
    for p,old in zip((*source_shared.parameters(), *source_private.parameters()), originals):
        torch.testing.assert_close(p, old, rtol=0, atol=0)


def test_pending_resume_preserves_both_adams_and_private_identity():
    online, live, snapshot, _, _ = make()
    online.commit(snapshot, live, action_step=499)
    online.consume({n:p.detach()*128 for n,p in live.items()}, gradient_step=500)
    # Both shared/private Adam histories are now nonempty at the saved boundary.
    online.commit(snapshot, live, action_step=500)
    saved = online.state_dict()
    restarted, restored, _, _, _ = make()
    with torch.no_grad():
        for n,p in restored.items():p.copy_(live[n])
    restarted.load_state_dict(saved, restored)
    gradients = {n:p.detach()*128 for n,p in live.items()}
    online.consume(gradients, gradient_step=501)
    restarted.consume(gradients, gradient_step=501)
    for p,q in zip(online.actor.parameters(), restarted.actor.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            if p in online.optimizer.state:
                torch.testing.assert_close(online.optimizer.state[p][key], restarted.optimizer.state[q][key], rtol=0, atol=0)
    private = online.actor.private
    online.commit(snapshot, live, action_step=501)
    assert online.actor.private is private
    assert private.state_key == ('online', 'test-live')


def test_cross_run_load_rejected_before_mutation_and_pending_private_changes_rejected():
    online, live, snapshot, _, _ = make()
    other, _, _, _, _ = make(run_id='another-live-run')
    before = [p.clone() for p in other.actor.parameters()]
    with pytest.raises(ValueError, match='identity'):
        other.actor.load_state_dict(copy.deepcopy(online.actor.state_dict()))
    for p,q in zip(other.actor.parameters(), before):torch.testing.assert_close(p,q,rtol=0,atol=0)
    online.commit(snapshot, live, action_step=499)
    with torch.no_grad():next(online.actor.private.parameters()).add_(1e-4)
    with pytest.raises(RuntimeError, match='mutated'):
        online.consume({n:p.detach()*128 for n,p in live.items()}, gradient_step=500)


def test_adaptive_private_maps_cannot_be_silently_relabelled_as_anchored():
    from rewriter.adaptive_coordinate_chisel import AdaptiveCheckpointProbes
    _, _, snapshot, shared, _ = make()
    adaptive = AdaptiveCheckpointProbes(snapshot, model_dim=128, latent_tokens=2,
                                       seed=617, state_key=('gpt',0,499))
    with pytest.raises(TypeError, match='unsupported action family'):
        CoordinateOnlineBinding.from_fitted(shared, adaptive, snapshot, run_id='test-live')
