import torch
import pytest
import yaml
from pathlib import Path
from chisel.diagnostic_lbfgs import ClosureLBFGS


def test_optimizer_closure_minimizes_actual_objective_and_restores_history():
    p = torch.nn.Parameter(torch.tensor([5., -3.]))
    opt = ClosureLBFGS([p], lr=1., max_iter=8, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad(set_to_none=True)
        loss = (p - 2).square().sum()
        loss.backward()
        return loss.detach()
    opt.closure = closure
    opt.step()
    torch.testing.assert_close(p, torch.full_like(p, 2.))
    saved = opt.state_dict()
    other = ClosureLBFGS([p], lr=1., max_iter=8)
    other.load_state_dict(saved)
    assert other.state_dict()['state'][0]['func_evals'] == saved['state'][0]['func_evals']


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA required'))])
def test_single_checkpoint_resume_restores_live_private_parameters(tmp_path, device):
    from chisel.harness import run
    from tests.harness.test_checkpoint_duals import configuration
    config = configuration(tmp_path)
    config['runtime']['device'] = device
    path = Path(config['data']['manifest'])
    manifest = yaml.safe_load(path.read_text())
    path.write_text(yaml.safe_dump({'states': manifest['states'][:1]}))
    config['data']['batch_counts'] = {'text': 1}
    config.update(diagnostic_lbfgs=True, diagnostic_kl_weight=0., diagnostic_loss_scale=100.,
                  gradient_clip=float('inf'), measure_kl=False, measure_rankings=False)
    config['preservation'].update(constraint_metric='logit_mse', center_logits=False,
                                  logit_mse_weight=10., ranking_weight=0.)
    insert = next(i for i,h in enumerate(config['setup']) if h['name']=='chisel.offline.setup')+1
    config['setup'][insert:insert] = [
        dict(name='chisel.overfit_diagnostics.setup',hyperparams=dict(kind='chisel',initial='zero')),
        dict(name='chisel.diagnostic_lbfgs.setup',hyperparams=dict(max_iter=2,history_size=4))]
    config['pre_forward'] = []
    config['post_optimizer'] = [dict(name='chisel.parallel_diagnostics.checkpoint',hyperparams=dict(every=2))]
    expected = run(config)
    config.update(output_dir=str(tmp_path/'prefix_diagnostic'), stop_step=2)
    prefix = run(config)
    config.update(output_dir=str(tmp_path/'resumed_diagnostic'), stop_step=4,
                  resume=str(tmp_path/'prefix_diagnostic/latest_rank0.pt'))
    actual = run(config)
    assert any(not torch.equal(a,b) for a,b in zip(prefix.extras['fit']['probes'].parameters(),
                                                   expected.extras['fit']['probes'].parameters()))
    torch.testing.assert_close(actual.model.state_dict(),expected.model.state_dict(),rtol=0,atol=0)
    torch.testing.assert_close(actual.extras['fit']['probes'].state_dict(),
                               expected.extras['fit']['probes'].state_dict(),rtol=0,atol=0)
    a,b = actual.optimizer.state_dict(),expected.optimizer.state_dict()
    assert a['param_groups'] == b['param_groups']
    torch.testing.assert_close(a['state'],b['state'],rtol=0,atol=0)
