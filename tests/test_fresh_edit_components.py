import pytest
import torch
from scripts.evaluate_chisel_fresh import select_edit_component


def test_component_edits_partition_only_editable_tensors_without_mutating_inputs():
    original={'matrix':torch.randn(3,2),'bias':torch.randn(3),'gain':torch.randn(3),'scalar':torch.tensor(1.)}
    proposal={name:value+1 for name,value in original.items()}
    old={name:value.clone() for name,value in original.items()}
    matrices=select_edit_component(proposal,original,{'matrix','bias'},'matrices')
    biases=select_edit_component(proposal,original,{'matrix','bias'},'biases')
    for name in original:
        assert torch.equal(original[name],old[name])
        assert matrices[name] is (proposal[name] if name=='matrix' else original[name])
        assert biases[name] is (proposal[name] if name=='bias' else original[name])
        delta=matrices[name]-original[name]+biases[name]-original[name]
        assert torch.allclose(delta,proposal[name]-original[name] if name in {'matrix','bias'} else torch.zeros_like(delta))


def test_component_selection_rejects_unknown_scope_and_name_mismatch():
    original={'matrix':torch.zeros(2,2)}
    with pytest.raises(ValueError):select_edit_component(original,original,original,'all')
    with pytest.raises(ValueError):select_edit_component({},original,original,'matrices')
    with pytest.raises(ValueError):select_edit_component(original,original,{'missing'},'matrices')
