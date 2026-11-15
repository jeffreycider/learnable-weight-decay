"""The optional compiler must not replace the canonical checkpoint/Adam owner."""
from unittest.mock import Mock, patch
import torch
from scripts.train_frozen_preservation import chisel_forward_callable


def test_compile_constructed_once_and_canonical_state_keys_preserved():
    actor=torch.nn.Linear(3,2)
    original_keys=set(actor.state_dict())
    wrapper=Mock(side_effect=actor)
    with patch('torch.compile',return_value=wrapper) as compiler:
        forward=chisel_forward_callable(actor,compile_chisel=True)
        optimizer=torch.optim.Adam(actor.parameters(),lr=.001)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            forward(torch.ones(1,3)).square().sum().backward()
            optimizer.step()
        compiler.assert_called_once_with(actor,fullgraph=True,dynamic=False)
    assert wrapper.call_count==3
    assert set(actor.state_dict())==original_keys=={'weight','bias'}
    assert all(not key.startswith('_orig_mod.') for key in actor.state_dict())
    assert optimizer.param_groups[0]['params'][0] is actor.weight


def test_eager_default_does_not_invoke_compiler():
    actor=torch.nn.Linear(3,2)
    with patch('torch.compile') as compiler:
        assert chisel_forward_callable(actor) is actor
        compiler.assert_not_called()
