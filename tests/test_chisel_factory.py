import argparse
import pytest
import torch
from rewriter.chisel_factory import add_chisel_arguments,chisel_architecture,validate_checkpoint_architecture,scale_output_heads


def specification():
    parser=argparse.ArgumentParser();add_chisel_arguments(parser)
    return chisel_architecture(parser.parse_args([]),rank=8,trajectory_dim=400)


def test_architecture_metadata_is_explicit_and_legacy_weights_fail_closed():
    architecture=specification()
    assert architecture['model_dim']==256 and architecture['num_layers']==4 and architecture['patch_size']==32
    assert architecture['residual_block']=='baseline.model.Block'
    with pytest.raises(ValueError,match='legacy'):
        validate_checkpoint_architecture({'metadata':{'options':{}}},architecture)
    validate_checkpoint_architecture({'metadata':{'architecture':architecture}},architecture)
    wrong=dict(architecture,model_dim=128)
    with pytest.raises(ValueError,match='mismatch'):
        validate_checkpoint_architecture({'metadata':{'architecture':wrong}},architecture)


def test_output_head_scaling_uses_public_linear_head_interface():
    class Actor(torch.nn.Module):
        def __init__(self):
            super().__init__();self.a=torch.nn.Linear(2,3);self.b=torch.nn.Linear(2,1,bias=False)
        @property
        def output_heads(self):return self.a,self.b
    actor=Actor();before={n:p.detach().clone() for n,p in actor.named_parameters()}
    scale_output_heads(actor,.2)
    for n,p in actor.named_parameters():torch.testing.assert_close(p,before[n]*.2)


def test_zero_head_restart_keeps_learned_features_and_trainable_actions():
    from tests.test_transformer_chisel import active_model, flatten
    from tests.test_controller import make_snapshot
    actor=active_model();snapshot=make_snapshot()
    before=actor.matrix_input.weight.detach().clone()
    assert flatten(actor(snapshot)).norm()>0
    scale_output_heads(actor,0.)
    torch.testing.assert_close(actor.matrix_input.weight,before,atol=0,rtol=0)
    action=actor(snapshot)
    assert flatten(action).count_nonzero()==0
    action.contraction(snapshot.weights,snapshot.vector_weights).backward()
    assert all(head.weight.grad.norm()>0 for head in actor.output_heads)
