import json
import torch
import yaml
from chisel.harness import run, resolve
from chisel.preservation_fit import sampled_inputs
from chisel.preservation_data import training_batch_step
from test_integration import source_config


def test_full_audit_replays_last_fitted_tokens_and_persists_identity(tmp_path, monkeypatch):
    from chisel import offline, preservation_fit
    run(source_config(tmp_path))
    entry = json.loads((tmp_path/'source/trajectory/000003/state.ready.json').read_text())
    manifest = tmp_path/'manifest.yaml'
    manifest.write_text(yaml.safe_dump(dict(states=[dict(path=entry['path'],sha256=entry['sha256'],domain='text')])))
    cfg = resolve('recipes/offline.yaml')
    cfg.update(train_steps=2,output_dir=str(tmp_path/'offline'),microbatch_size=1,
               validation_data=None,compile_gpt_loss=False,compile_proposal=False,
               preservation_examples=1,training_audit_inputs='last_fitted')
    cfg['runtime'].update(device='cpu',autocast=None)
    cfg['compile']['enabled']=False
    cfg['model'].update(rank=2,width=16,depth=1,head_dim=4,expansion=2)
    cfg['private_probes']['rank']=2
    cfg['data'].update(manifest=str(manifest),batch_counts=dict(text=1))
    cfg['training_audit_data']=dict(cfg['data'])
    cfg['preservation_data']=dict(policy='uniform_past',seed=0)
    while training_batch_step(cfg,entry['sha256'],1,3)==3:
        cfg['preservation_data']['seed']+=1
    fitted, audited = [], []
    gradient, audit = offline.preservation_gradient, preservation_fit.audit
    def train(*args):
        fitted.append(sampled_inputs(args[4],args[5]).clone())
        return gradient(*args)
    def evaluate(*args):
        audited.append(sampled_inputs(args[3],args[4]).clone())
        return audit(*args)
    monkeypatch.setattr(offline,'preservation_gradient',train)
    monkeypatch.setattr(preservation_fit,'audit',evaluate)
    state=run(cfg)
    assert len(audited)==1
    torch.testing.assert_close(audited[-1],fitted[-1],rtol=0,atol=0)
    observation=state.extras['private_probes'].measurements[entry['sha256']]
    assert observation['batch_step']!=3
    saved=torch.load(tmp_path/'offline/latest_rank0.pt',weights_only=False)
    assert saved['extras']['private_probes']['measurements'][entry['sha256']]==observation
    report=json.loads((tmp_path/'offline/audits/train_audit_00000002.json').read_text())
    assert report['checkpoints'][0]['input_origin']=='last_fitted'
    assert report['checkpoints'][0]['fitting_observation']==observation
