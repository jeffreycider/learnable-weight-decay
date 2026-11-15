import pytest
import torch
from scripts.qualify_coordinate_chisel import validate_context_access,parameter_motion


def test_heldout_support_requires_frozen_cleared_global_actor_and_disjoint_role():
    actor=torch.nn.Linear(2,1);split={'train':[1,2],'audit':[3]}
    with pytest.raises(ValueError,match='fitting model'):validate_context_access(split,3,role='train',purpose='fitting')
    with pytest.raises(ValueError,match='frozen'):validate_context_access(split,3,role='train',purpose='private_support',shared_actor=actor)
    actor.requires_grad_(False)
    validate_context_access(split,3,role='train',purpose='private_support',shared_actor=actor)
    validate_context_access(split,3,role='audit',purpose='private_query',shared_actor=actor)
    with pytest.raises(ValueError,match='role mismatch'):validate_context_access(split,3,role='audit',purpose='private_support',shared_actor=actor)
    with pytest.raises(ValueError,match='wholly held-out'):validate_context_access(split,1,role='train',purpose='private_support',shared_actor=actor)
    actor.weight.grad=torch.ones_like(actor.weight)
    with pytest.raises(ValueError,match='gradient-cleared'):validate_context_access(split,3,role='train',purpose='private_support',shared_actor=actor)
