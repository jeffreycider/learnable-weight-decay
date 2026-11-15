import json
import pytest
import os
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from chisel.harness import run,resolve
from test_integration import source_config


def worker(rank, rendezvous, config):
    os.environ['WORLD_SIZE']='2'
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    try:run(config)
    finally:dist.destroy_process_group()


@pytest.mark.parametrize("dual_scope", ["shared", "checkpoint"])
def test_distributed_offline_matches_single_process(tmp_path, dual_scope):
    source=source_config(tmp_path);run(source)
    entries=[json.loads(path.read_text()) for path in sorted((tmp_path/'source/trajectory').rglob('state.ready.json'))[1:4]]
    manifest=tmp_path/'manifest.yaml'
    manifest.write_text(yaml.safe_dump(dict(states=[dict(path=e['path'],sha256=e['sha256'],domain='text') for e in entries])))
    cfg=resolve('recipes/offline.yaml');cfg.update(train_steps=2,output_dir=str(tmp_path/'single'),microbatch_size=2,validation_data=None,compile_gpt_loss=False)
    cfg['runtime'].update(device='cpu',autocast=None);cfg['compile']['enabled']=False;cfg['compile_proposal']=False
    cfg['model'].update(rank=2,width=16,depth=1,head_dim=4,expansion=2)
    cfg['private_probes']['rank']=2;cfg['data'].update(manifest=str(manifest),batch_counts=dict(text=3))
    cfg['training_audit_data']=dict(cfg['data'])
    cfg['preservation']['dual_scope']=dual_scope
    single=run(cfg)
    cfg['runtime']['world_size']=2;cfg['output_dir']=str(tmp_path/'distributed')
    mp.spawn(worker,args=(str(tmp_path/'join'),cfg),nprocs=2,join=True)
    for rank in range(2):
        checkpoint=torch.load(tmp_path/f'distributed/latest_rank{rank}.pt',weights_only=False)
        for name,value in single.model.state_dict().items():
            torch.testing.assert_close(checkpoint['model'][name],value,rtol=1e-4,atol=2e-6)
        assert set(checkpoint['extras']['private_probes']['latest'])=={entry['sha256'] for entry in entries}
        if dual_scope == 'checkpoint':
            expected = single.extras['checkpoint_duals'].state_dict()
            assert checkpoint['extras']['checkpoint_duals'] == expected
    logged=json.loads((tmp_path/'distributed/metrics.jsonl').read_text().splitlines()[-1])
    for name,value in single.extras['metrics'].items():
        if name.startswith('train_audit_'):
            assert abs(logged[name]-value)<1e-6,(name,logged[name],value)
    assert (tmp_path/'distributed/generation_000002.yaml').exists()
