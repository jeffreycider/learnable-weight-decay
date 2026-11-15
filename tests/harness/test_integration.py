import json
from pathlib import Path
import torch
import yaml
from chisel.harness import run, resolve

class ToyStream:
    def __init__(self, device, rank=0, world_size=1):self.device,self.step=device,0;self.batch_size,self.seq_len=8,4
    def __next__(self):
        rng=torch.Generator().manual_seed(100+self.step);self.step+=1
        return torch.randint(16,(2,4),generator=rng).to(self.device),torch.randint(16,(2,4),generator=rng).to(self.device)
    def state_dict(self):return dict(step=self.step)
    def load_state_dict(self,saved):self.step=saved['step']

def source_config(tmp_path):
    cfg=resolve('recipes/nanogpt_capture.yaml')
    cfg.update(train_steps=5,output_dir=str(tmp_path/'source'),microbatch_size=1)
    cfg['runtime'].update(device='cpu',world_size=1,autocast=None)
    cfg['compile']['enabled']=False
    cfg['model'].update(vocab_size=16,num_layers=1,model_dim=8,head_dim=4,expansion=2)
    cfg['data']=dict(_target_=__name__+'.ToyStream')
    for spec in cfg['optimizers'].values():
        spec['tensor_lr']=False
        if 'fused' in spec['optimizer']:spec['optimizer']['fused']=False
        if 'compile' in spec['optimizer']:spec['optimizer']['compile']=False
    cfg['post_optimizer']=[]
    for hook in cfg['setup']:
        if hook['name']=='chisel.activation_records.setup':hook['hyperparams']['samples']=4
    return cfg

def test_capture_offline_and_online_share_the_loop(tmp_path):
    source=source_config(tmp_path);run(source)
    receipts=sorted((tmp_path/'source/trajectory').rglob('state.ready.json'))[1:-1]
    assert len(receipts)==3
    entries=[json.loads(p.read_text()) for p in receipts]
    manifest=tmp_path/'manifest.yaml'
    manifest.write_text(yaml.safe_dump(dict(states=[dict(path=e['path'],sha256=e['sha256'],domain='nanogpt') for e in entries])))
    cfg=resolve('recipes/offline.yaml');cfg.update(train_steps=3,output_dir=str(tmp_path/'offline'),microbatch_size=1)
    cfg['runtime'].update(device='cpu',autocast=None);cfg['compile']['enabled']=False
    cfg['model'].update(rank=2,width=16,depth=1,head_dim=4,expansion=2)
    cfg['private_probes']['rank']=2
    cfg['data'].update(manifest=str(manifest),batch_counts=dict(nanogpt=2))
    cfg['validation_data']=None;cfg['compile_gpt_loss']=False
    learned=run(cfg)
    assert learned.model.proj.weight.norm()>0
    assert len(learned.extras['private_probes'].latest)==3
    online=resolve('recipes/nanogpt_online.yaml')
    for key in ['train_steps','runtime','compile','model','data','optimizers','microbatch_size']:
        online[key]=source[key]
    online['output_dir']=str(tmp_path/'online')
    online['chisel']['model']=cfg['model'];online['chisel']['start_step']=1
    online['post_optimizer']=[dict(name='chisel.online.after_optimizer')]
    result=run(online)
    assert result.extras['consolidation'].pending_graph is None
    assert result.extras['consolidation'].actor.proj.weight.norm()>0
    assert result.extras['training_forwards']==10

def test_online_resume_with_pending_credit_matches_continuation(tmp_path):
    source=source_config(tmp_path)
    config=resolve('recipes/nanogpt_online.yaml')
    for key in ['train_steps','runtime','compile','model','data','optimizers','microbatch_size']:
        config[key]=source[key]
    config['chisel']['model'].update(rank=2,width=16,depth=1,head_dim=4,expansion=2)
    config['chisel']['start_step']=1
    config['post_optimizer']=[dict(name='chisel.online.after_optimizer'),dict(name='chisel.recording.checkpoint',hyperparams=dict(every=2))]
    config['output_dir']=str(tmp_path/'continuous')
    continuous=run(config)
    assert len(list((tmp_path/'continuous').glob('checkpoint_*_rank0.pt')))==3
    config['output_dir']=str(tmp_path/'prefix');config['stop_step']=2
    prefix=run(config)
    assert prefix.extras['consolidation'].pending_graph is not None
    config['resume']=str(tmp_path/'prefix/latest_rank0.pt');config['stop_step']=5;config['output_dir']=str(tmp_path/'resumed')
    resumed=run(config)
    for a,b in zip(continuous.model.parameters(),resumed.model.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    for a,b in zip(continuous.extras['consolidation'].actor.parameters(),resumed.extras['consolidation'].actor.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
