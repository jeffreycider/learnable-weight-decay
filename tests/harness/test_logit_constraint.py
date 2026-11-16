import torch
from chisel.harness import RunState, run
from chisel.preservation_fit import logits, preservation_gradient
from chisel.preservation import MultiplicativeDual
from tests.harness.test_checkpoint_duals import configuration


def test_dual_mse_cotangent_matches_full_autograd_without_kl_or_rankings():
    torch.manual_seed(14)
    model = torch.nn.Linear(4, 5).double()
    original = {n:p.detach().clone() for n,p in model.named_parameters()}
    effective = {'weight': (original['weight'] + .01).requires_grad_()}
    batch = {'inputs': torch.randn(7,4,dtype=torch.float64)}
    config = dict(microbatch_size=3, runtime=dict(autocast=None), measure_kl=False,
                  measure_rankings=False, preservation=dict(constraint_metric='logit_mse',
                  center_logits=False, top_k=4, margin_fraction=.5, logit_mse_weight=2., ranking_weight=0.))
    state = RunState(extras=dict(device=torch.device('cpu'), logits_forward=logits))
    gradients, metrics = preservation_gradient(model, original, original|effective,
        effective, batch, config, state, 3.)
    teacher = logits(model, original, batch['inputs'])
    student = logits(model, original|effective, batch['inputs'])
    expected, = torch.autograd.grad(5 * (student-teacher).square().mean(), effective['weight'])
    torch.testing.assert_close(gradients['weight'], expected, atol=1e-12, rtol=1e-10)
    assert 'kl' not in metrics and 'rank_changed_positions' not in metrics


def test_full_offline_loop_updates_dual_from_mse_and_accounts_for_loss(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    config['preservation']['dual_scope'] = 'shared'
    config['measure_kl'] = True
    measurements = []
    original = MultiplicativeDual.update
    def observe(self, value):
        measurements.append(value)
        original(self, value)
    monkeypatch.setattr(MultiplicativeDual, 'update', observe)
    state = run(config)
    metrics = state.extras['metrics']
    assert len(measurements) == config['train_steps']
    assert measurements[-1] == metrics['logit_mse']
    assert measurements[-1] != metrics['kl']
    expected = -metrics['norm_reduction_pct']/100 + metrics['preservation_penalty']
    assert abs(expected-state.extras['train_loss']) < 1e-12
    assert 'kl_penalty' not in metrics and 'rank_changed_positions' not in metrics
