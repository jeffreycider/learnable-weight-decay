import torch
import json
from chisel.capture import TrajectoryWriter
from chisel.recording import cpu_copy

def test_capture_owns_storage_and_writes_receipt(tmp_path):
    source=torch.ones(4)
    writer=TrajectoryWriter(tmp_path)
    writer.submit(0,cpu_copy({'weight':source}))
    source.zero_();writer.close()
    assert torch.load(tmp_path/'step000000.pt',weights_only=True)['weight'].sum()==4
    receipt=json.loads((tmp_path/'step000000.ready.json').read_text())
    assert len(receipt['sha256'])==64
    assert receipt['bytes']==(tmp_path/'step000000.pt').stat().st_size

def test_capture_is_selected_only_through_yaml():
    from chisel.harness import resolve
    for domain in ['cifar','nanogpt']:
        base=resolve(f'recipes/{domain}.yaml');capture=resolve(f'recipes/{domain}_capture.yaml')
        for key in ['model','optimizers','data','train_steps']:
            assert base[key]==capture[key]
