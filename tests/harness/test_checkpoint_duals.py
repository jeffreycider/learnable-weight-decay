from copy import deepcopy
import json
import torch
import yaml
from chisel.harness import run, resolve
from chisel.preservation import CheckpointDuals, Dual, MultiplicativeDual
from tests.harness.test_integration import source_config


def test_dual_updates_do_not_leak_between_checkpoints_or_saved_states():
    template = MultiplicativeDual(1., .1, 1e-5, 1e4)
    bank = CheckpointDuals(template)
    bank.update('difficult', bank.get('difficult'), 1.)
    bank.update('preserved', bank.get('preserved'), 0.)
    assert bank.get('difficult').value > 1 > bank.get('preserved').value
    assert bank.get('unseen').value == template.value == 1
    saved = bank.state_dict()
    restored = CheckpointDuals(template)
    restored.load_state_dict(saved)
    restored.update('difficult', restored.get('difficult'), 1.)
    assert bank.state_dict() == saved != restored.state_dict()
    template.tolerance = 1e-7
    assert restored.get('difficult').tolerance == 1e-7


def configuration(tmp_path):
    source = source_config(tmp_path)
    run(source)
    entries = [json.loads(p.read_text()) for p in sorted((tmp_path/'source/trajectory').rglob('state.ready.json'))[1:4]]
    manifest = tmp_path/'states.yaml'
    manifest.write_text(yaml.safe_dump(dict(states=[dict(path=e['path'], sha256=e['sha256'], domain='text') for e in entries])))
    config = resolve('recipes/offline.yaml')
    config.update(train_steps=4, output_dir=str(tmp_path/'continuous'), microbatch_size=1,
                  validation_data=None, compile_gpt_loss=False, compile_proposal=False)
    config['runtime'].update(device='cpu', autocast=None)
    config['model'].update(rank=2, width=16, depth=1, head_dim=4, expansion=2)
    config['private_probes']['rank'] = 2
    config['data'].update(manifest=str(manifest), batch_counts=dict(text=2))
    config['preservation']['dual_scope'] = 'checkpoint'
    config['post_optimizer'] = [dict(name='chisel.recording.progress', hyperparams=dict(every=1)),
                               dict(name='chisel.recording.checkpoint', hyperparams=dict(every=2))]
    return config


def test_checkpoint_duals_resume_with_identical_weights_and_accounted_loss(tmp_path):
    config = configuration(tmp_path)
    continuous = run(config)
    config.update(output_dir=str(tmp_path/'prefix'), stop_step=2)
    run(config)
    config.update(output_dir=str(tmp_path/'resumed'), resume=str(tmp_path/'prefix/latest_rank0.pt'), stop_step=4)
    resumed = run(config)
    assert resumed.extras['checkpoint_duals'].state_dict() == continuous.extras['checkpoint_duals'].state_dict()
    assert len(resumed.extras['checkpoint_duals'].states) == 3
    for left, right in zip(resumed.model.parameters(), continuous.model.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    metrics = resumed.extras['metrics']
    expected = (-metrics['norm_reduction_pct']/100 + config['consolidated_penalty']*metrics['consolidated_penalty']
                + metrics['kl_penalty'] + config['preservation']['logit_mse_weight']*metrics['logit_mse']
                + config['preservation']['ranking_weight']*metrics['ranking_loss'])
    assert abs(expected - resumed.extras['train_loss']) < 1e-12
    assert continuous.extras['dual'].value == config['preservation']['dual']['initial']
