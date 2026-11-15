import torch
from chisel.recording import cpu_copy, save_atomic

def test_snapshot_is_independent_and_atomic(tmp_path):
    x=torch.ones(2,requires_grad=True);saved=cpu_copy({'x':[x]})
    with torch.no_grad():x.add_(3)
    assert saved['x'][0].tolist()==[1,1]
    assert not saved['x'][0].requires_grad
    path=tmp_path/'state.pt';save_atomic(saved,path)
    assert torch.load(path,weights_only=True)['x'][0].tolist()==[1,1]
    assert not path.with_suffix('.pt.tmp').exists()
