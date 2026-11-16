import copy
import json
from pathlib import Path
import pytest
import torch
import yaml
from chisel.harness import run, resolve
from chisel.attention_identity_cohort import AttentionReplay
from chisel.attention_identity_fit import propose
from chisel.transfer import apply_edits
from test_integration import source_config


class ImageStream:
    def __init__(self,device,rank=0,world_size=1):self.device,self.step=device,0;self.batch_size=2
    def __next__(self):
        rng=torch.Generator().manual_seed(50+self.step);self.step+=1
        return torch.randn(2,3,32,32,generator=rng).to(self.device),torch.randint(10,(2,),generator=rng).to(self.device)
    def state_dict(self):return dict(step=self.step)
    def load_state_dict(self,state):self.step=state['step']


def recipe(tmp_path):
    states=[]
    for domain in ['nanogpt','cifar']:
        cfg=source_config(tmp_path/domain)
        if domain=='cifar':
            cfg['model']=dict(_target_='chisel.models.CifarGPT',width=8,depth=1,patch=16,readout='mean',head_dim=4,learned_qk_scale=True)
            cfg['data']=dict(_target_=__name__+'.ImageStream')
            cfg.pop('initialize_model',None)
            cfg['optimizers']['scalars']['parameters']=r'(.*\.(bias|gains|log_scale)|class_token)'
        run(cfg)
        for p in sorted((tmp_path/domain/'source/trajectory').rglob('state.ready.json'))[2:4]:
            e=json.loads(p.read_text());states.append(dict(path=e['path'],sha256=e['sha256'],domain=domain))
    manifest=tmp_path/'mixed.yaml';manifest.write_text(yaml.safe_dump(dict(states=states)))
    cfg=resolve('recipes/offline.yaml')
    cfg.update(train_steps=24,output_dir=str(tmp_path/'fit'),microbatch_size=1,validation_data=None,
               compile_gpt_loss=False,compile_proposal=False,training_audit_inputs='recorded',private_steps=1)
    cfg['runtime'].update(device='cpu',autocast=None);cfg['compile']['enabled']=False
    cfg['model'].update(rank=2,width=16,depth=1,head_dim=4,expansion=2)
    cfg['private_probes'].update(rank=2,cache_bytes=1<<20)
    cfg['private_probes']['optimizer']['lr']=.01
    cfg['optimizers']['actor']['optimizer']['lr']=.003
    cfg['data'].update(manifest=str(manifest),batch_counts=dict(cifar=2,nanogpt=2))
    cfg['replay']=dict(_target_='chisel.attention_identity_cohort.AttentionReplay')
    cfg['proposal_function']='chisel.attention_identity_fit.propose'
    cfg['setup'].insert(4,dict(name='chisel.attention_identity_cohort.setup'))
    cfg['training_audit_data']=copy.deepcopy(cfg['data'])
    for hook in cfg['post_optimizer']:
        if hook['name']=='chisel.offline.evaluate':hook['hyperparams']['every']=12
    return cfg


def test_mixed_domains_train_and_audit_exact_same_decoder(tmp_path):
    cfg=recipe(tmp_path);state=run(cfg)
    assert isinstance(state.extras['replay'],AttentionReplay)
    assert state.model.proj.weight.norm()>0
    assert state.extras['metrics']['train_audit_minimum_norm_reduction_pct']>0
    assert state.extras['metrics']['train_audit_maximum_logit_change']<1e-5
    rows=json.loads((tmp_path/'fit/audits/train_audit_00000024.json').read_text())['checkpoints']
    assert len(rows)==4 and {r['domain'] for r in rows}=={'cifar','nanogpt'}
    assert all(r['input_origin']=='recorded' and r['fitting_observation']=={} for r in rows)
    for identity,p in state.extras['private_probes'].latest.items():
        saved=torch.load(p,weights_only=False)
        assert saved['optimizer']['state']
        assert any(v['exp_avg'].norm()>0 for v in saved['optimizer']['state'].values())
    # No model evaluations supervise this algebraically constrained norm-only task.
    assert state.extras.get('training_forwards',0)==0


def test_norm_only_task_rejects_unrestricted_decoder(tmp_path):
    cfg=recipe(tmp_path);cfg['proposal_function']='chisel.network.propose'
    with pytest.raises(ValueError,match='exact attention identity'):run(cfg)


def test_snapshot_metadata_matches_explicit_reference_proposal():
    from test_attention_identity_fit import example
    model,base,snapshot,actor,probes=example()
    with torch.no_grad():actor.proj.weight.normal_(std=.03)
    for i,block in enumerate(model.blocks):
        name=f'blocks.{i}.attn.v.weight'
        snapshot[name]['attention_head_dim']=block.attn.head_dim
        snapshot[name]['attention_value_bias']=base[name.replace('weight','bias')]
    a=propose(actor,probes,snapshot,base,8);b=propose(actor,probes,snapshot)
    for name in a:
        for bank in a[name]:torch.testing.assert_close(a[name][bank],b[name][bank],rtol=0,atol=0)


def test_distributed_matches_serial_shared_and_private_updates(tmp_path):
    import torch.multiprocessing as mp
    from test_offline_distributed_integration import worker
    cfg=recipe(tmp_path);cfg['train_steps']=2
    cfg['distributed_audit']=True
    expected=run(cfg)
    cfg['runtime']['world_size']=2;cfg['output_dir']=str(tmp_path/'distributed')
    mp.spawn(worker,args=(str(tmp_path/'rendezvous'),cfg),nprocs=2,join=True)
    for rank in range(2):
        saved=torch.load(tmp_path/f'distributed/latest_rank{rank}.pt',weights_only=False)
        for name,value in expected.model.state_dict().items():
            torch.testing.assert_close(saved['model'][name],value,rtol=1e-4,atol=2e-6)
        for identity,path in saved['extras']['private_probes']['latest'].items():
            actual=torch.load(path,weights_only=False)['probes']
            reference=torch.load(expected.extras['private_probes'].latest[identity],weights_only=False)['probes']
            for name,value in reference.items():torch.testing.assert_close(actual[name],value,rtol=1e-4,atol=2e-6)
    audit=json.loads((tmp_path/'distributed/audits/train_audit_00000002.json').read_text())
    assert len(audit['checkpoints'])==4


def test_resume_restores_shared_weights_and_checkpoint_private_adam(tmp_path):
    cfg=recipe(tmp_path);cfg['train_steps']=4
    expected=run(cfg)
    cfg['output_dir']=str(tmp_path/'prefix');cfg['stop_step']=2
    run(cfg)
    cfg['resume']=str(tmp_path/'prefix/latest_rank0.pt')
    cfg['output_dir']=str(tmp_path/'resumed');cfg['stop_step']=4
    actual=run(cfg)
    for name,value in expected.model.state_dict().items():
        torch.testing.assert_close(actual.model.state_dict()[name],value,rtol=0,atol=0)
    for identity,p in actual.extras['private_probes'].latest.items():
        a=torch.load(p,weights_only=False)
        b=torch.load(expected.extras['private_probes'].latest[identity],weights_only=False)
        for name,value in b['probes'].items():torch.testing.assert_close(a['probes'][name],value,rtol=0,atol=0)
        for index,states in b['optimizer']['state'].items():
            for name,value in states.items():torch.testing.assert_close(a['optimizer']['state'][index][name],value,rtol=0,atol=0)
