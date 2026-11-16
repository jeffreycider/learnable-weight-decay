from pathlib import Path
import pytest
import torch
import yaml
from chisel.harness import RunState, run
from chisel.logit_constraints import LogitConstraints, update
from chisel.preservation_fit import logits, preservation_gradient


def test_multiplier_vjp_matches_full_objective_with_uneven_microbatches():
    torch.manual_seed(81)
    model = torch.nn.Linear(4, 5).double()
    original = {n: p.detach().clone() for n, p in model.named_parameters()}
    effective = {'weight': (original['weight'] + .01 * torch.randn_like(original['weight'])).requires_grad_()}
    inputs = torch.randn(7, 4, dtype=torch.double)
    teacher = logits(model, original, inputs).detach()
    constraints = LogitConstraints(teacher, 20.)
    constraints.multipliers.normal_()
    config = dict(microbatch_size=3, runtime=dict(autocast=None), measure_kl=False,
                  measure_rankings=False, preservation=dict(top_k=5, margin_fraction=.5,
                  constraint_metric='logit_mse', center_logits=False,
                  logit_mse_weight=10., ranking_weight=0.))
    state = RunState(extras=dict(logits_forward=logits, device=torch.device('cpu'),
                                logit_constraints=constraints))
    gradient, metrics = preservation_gradient(model, original, original | effective,
        effective, dict(inputs=inputs), config, state, 0.)
    difference = logits(model, original | effective, inputs) - teacher
    loss = 10 * difference.square().mean() + (constraints.multipliers * difference).mean()
    expected = torch.autograd.grad(loss, effective['weight'])[0]
    torch.testing.assert_close(gradient['weight'], expected, atol=1e-12, rtol=1e-10)
    assert abs(loss.item() - (10 * metrics['logit_mse'] + metrics['constraint_linear'])) < 1e-12
    assert state.extras['training_forwards'] == 3
    torch.testing.assert_close(constraints.teacher, teacher, rtol=0, atol=0)


def test_augmented_lagrangian_retains_nontrivial_shrinkage_at_equality():
    # A one-output linear model has a known nontrivial minimum-norm rewrite.
    weight = torch.nn.Parameter(torch.tensor([[3., 1.]], dtype=torch.double))
    inputs = torch.ones(1, 2, dtype=torch.double)
    constraints = LogitConstraints(torch.tensor([[4.]], dtype=torch.double), 4.)
    for _ in range(20):
        optimizer = torch.optim.LBFGS([weight], max_iter=15, tolerance_grad=1e-12,
                                     tolerance_change=1e-16, line_search_fn='strong_wolfe')
        def closure():
            optimizer.zero_grad()
            error = inputs @ weight.T - constraints.teacher
            loss = .5 * weight.square().sum() + (constraints.multipliers * error).mean() + 2 * error.square().mean()
            loss.backward()
            return loss
        optimizer.step(closure)
        constraints.ascend(inputs @ weight.T)
    torch.testing.assert_close(weight, torch.tensor([[2., 2.]], dtype=torch.double), atol=1e-7, rtol=0)
    assert weight.norm() < torch.tensor([3., 1.]).norm() * .9
    assert constraints.updates == 20
    assert not constraints.multipliers.requires_grad


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA required'))])
@pytest.mark.parametrize('missing_restore', [False, True])
def test_actual_chisel_multiplier_resume_is_exact(tmp_path, device, missing_restore):
    from tests.harness.test_checkpoint_duals import configuration
    config = configuration(tmp_path)
    config['runtime']['device'] = device
    path = Path(config['data']['manifest'])
    manifest = yaml.safe_load(path.read_text())
    path.write_text(yaml.safe_dump({'states': manifest['states'][:1]}))
    config['data']['batch_counts'] = {'text': 1}
    config.update(diagnostic_lbfgs=True, diagnostic_kl_weight=0., diagnostic_loss_scale=100.,
                  gradient_clip=float('inf'), measure_kl=False, measure_rankings=False,
                  fit_logit_constraints=True)
    config['preservation'].update(constraint_metric='logit_mse', center_logits=False,
                                  logit_mse_weight=10., ranking_weight=0.)
    insert = next(i for i,h in enumerate(config['setup']) if h['name']=='chisel.offline.setup')+1
    config['setup'][insert:insert] = [
        dict(name='chisel.overfit_diagnostics.setup', hyperparams=dict(kind='chisel')),
        dict(name='chisel.logit_constraints.setup'),
        dict(name='chisel.diagnostic_lbfgs.setup', hyperparams=dict(max_iter=2, history_size=4))]
    config['pre_forward'] = []
    config['post_optimizer'] = [dict(name='chisel.logit_constraints.update', hyperparams=dict(every=2)),
        dict(name='chisel.parallel_diagnostics.checkpoint', hyperparams=dict(every=2))]
    if missing_restore:
        config['setup'] = [h for h in config['setup'] if h['name'] != 'chisel.recording.restore']
    expected = run(config)
    config.update(output_dir=str(tmp_path/'prefix'), stop_step=2)
    run(config)
    config.update(output_dir=str(tmp_path/'resumed'), stop_step=4,
                  resume=str(tmp_path/'prefix/latest_rank0.pt'))
    if missing_restore:
        from tests.harness.test_resume_compatibility import declaration
        config['setup'].append(dict(name='chisel.recording.restore'))
        config['resume_compatibility'] = declaration(tmp_path, ['setup.restore_hook'], {})
    actual = run(config)
    for a,b in [(actual.model, expected.model),
                (actual.extras['fit']['probes'], expected.extras['fit']['probes']),
                (actual.extras['logit_constraints'], expected.extras['logit_constraints'])]:
        torch.testing.assert_close(a.state_dict(), b.state_dict(), rtol=0, atol=0)
    assert actual.extras['logit_constraints'].multipliers.abs().max() > 0
    assert actual.extras['logit_constraints'].updates == 2
    assert not actual.optimizer.optimizer.state  # Cleared after changing the objective.


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_compiled_constraint_gradients_and_no_recompile_when_multipliers_change():
    from chisel.compiled_preservation import constrained_logit_loss
    from torch._dynamo.utils import counters
    torch.manual_seed(8)
    model = torch.nn.Linear(4, 5).cuda()
    parameters = dict(model.named_parameters())
    inputs, teacher, multipliers = (torch.randn(*s, device='cuda') for s in [(7, 4), (7, 5), (7, 5)])
    compiled = torch.compile(constrained_logit_loss, fullgraph=True, dynamic=False)
    counts = []
    for _ in range(3):
        multipliers.add_(.01)
        a = constrained_logit_loss(model, parameters, inputs, teacher, multipliers)
        b = compiled(model, parameters, inputs, teacher, multipliers)
        torch.testing.assert_close(a, b)
        ga = torch.autograd.grad(a[0] + a[2], tuple(parameters.values()))
        gb = torch.autograd.grad(b[0] + b[2], tuple(parameters.values()))
        torch.testing.assert_close(ga, gb)
        counts.append(counters['stats']['unique_graphs'])
    assert counts[0] == counts[1] == counts[2]


def test_invalid_multiplier_settings_are_rejected():
    with pytest.raises(ValueError):
        LogitConstraints(torch.zeros(1), 0.)
    with pytest.raises(ValueError):
        update(every=0)
