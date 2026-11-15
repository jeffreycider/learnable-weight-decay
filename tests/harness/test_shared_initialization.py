import pytest
import torch
from chisel.harness import RunState
from chisel.offline import initialize_shared_weights
from chisel.storage import sha256


def test_shared_warm_start_is_explicit_and_does_not_load_private_state(tmp_path):
    model=torch.nn.Linear(3,2)
    original={n:p.detach().clone() for n,p in model.named_parameters()}
    path=tmp_path/'pretrained.pt'
    torch.save(dict(model=original,extras=dict(private_probes={'old':'private'})),path)
    state=RunState(model=torch.nn.Linear(3,2),extras=dict(private_probes=object()))
    private=state.extras['private_probes']
    initialize_shared_weights(str(path),sha256(path))({},state)
    for name,value in state.model.state_dict().items():torch.testing.assert_close(value,original[name])
    assert state.extras['private_probes'] is private
    with pytest.raises(ValueError,match='either'):
        initialize_shared_weights(str(path),sha256(path))(dict(resume=str(path)),state)
    with pytest.raises(ValueError,match='SHA256'):
        initialize_shared_weights(str(path),'wrong')({},state)
