import pytest
import torch
from rewriter.owned_cpu_snapshot import owned_cpu_snapshot


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_owned_snapshot_exact_layout_metadata_and_independence(device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    original = dict(parameters={'x':torch.arange(12.,device=device).reshape(3,4).T},
                    optimizer={'state':{0:{'step':torch.tensor(4.),'exp_avg':torch.ones(4,3,device=device)}}},
                    identity=('gpt',1,6), none=None, list=[{'mutable':[1]}])
    copied = owned_cpu_snapshot(original)
    assert copied['parameters']['x'].device.type == 'cpu'
    assert copied['parameters']['x'].stride() == original['parameters']['x'].stride()
    assert torch.equal(copied['parameters']['x'], original['parameters']['x'].cpu())
    assert copied['identity'] == original['identity'] and copied['none'] is None
    original['parameters']['x'].zero_(); original['list'][0]['mutable'].append(2)
    assert copied['parameters']['x'].sum() == 66
    assert copied['list'] == [{'mutable':[1]}]
    assert copied['optimizer']['state'][0]['step'] == 4


def test_snapshot_rejects_unsupported_device():
    with pytest.raises(ValueError, match='CPU and CUDA'):
        owned_cpu_snapshot({'invalid':torch.empty(2,device='meta')})
