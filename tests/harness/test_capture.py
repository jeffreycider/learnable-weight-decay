import torch
import json
from chisel.capture import TrajectoryWriter
from chisel.recording import cpu_copy

def test_capture_owns_storage_and_writes_receipt(tmp_path):
    source=torch.ones(4)
    writer=TrajectoryWriter(tmp_path)
    writer.submit(0,cpu_copy({'weight':source}))
    source.zero_();writer.close()
    assert torch.load(tmp_path/'000000/state.pt',weights_only=True)['weight'].sum()==4
    receipt=json.loads((tmp_path/'000000/state.ready.json').read_text())
    assert len(receipt['sha256'])==64
    assert receipt['bytes']==(tmp_path/'000000/state.pt').stat().st_size

def test_capture_is_selected_only_through_yaml():
    from chisel.harness import resolve
    for domain in ['cifar','nanogpt']:
        base=resolve(f'recipes/{domain}.yaml');capture=resolve(f'recipes/{domain}_capture.yaml')
        for key in ['model','optimizers','data','train_steps']:
            assert base[key]==capture[key]

def test_snapshot_phase_and_replay_cursor(tmp_path):
    from chisel.harness import run
    from test_integration import source_config
    cfg=source_config(tmp_path);cfg['stop_step']=2
    run(cfg)
    saved=torch.load(tmp_path/'source/trajectory/000001/state.pt',weights_only=False)
    assert saved['phase']=='after_backward_before_optimizer'
    assert saved['current']['data']['step']==1
    assert saved['next_data']['step']==2
    assert saved['gradient_units']=='sum'
    assert saved['raw_radii']
    assert saved['weights'].keys() >= saved['gradients'].keys()
    assert (tmp_path/'source/hyperparameters.yaml').read_bytes()==(tmp_path/'source/recipe.yaml').read_bytes()
