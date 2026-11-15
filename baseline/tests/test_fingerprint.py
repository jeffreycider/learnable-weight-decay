import copy
import torch
from baseline.fingerprint import model_fingerprint


def test_full_fingerprint_includes_bfloat16_and_buffers_without_forward():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.,2.],dtype=torch.bfloat16))
            self.register_buffer('scale',torch.tensor(1.))
        def forward(self,*args):
            raise AssertionError('fingerprinting must not execute GPT')
    model = Model()
    result = model_fingerprint(model)
    assert result == model_fingerprint(copy.deepcopy(model))
    assert result['bytes'] == 8
    assert set(result['tensor_sha256']) == {'parameter:weight','buffer:scale'}
    model.scale.add_(1)
    changed = model_fingerprint(model)
    assert changed['sha256'] != result['sha256']
    assert changed['tensor_sha256']['parameter:weight'] == result['tensor_sha256']['parameter:weight']
