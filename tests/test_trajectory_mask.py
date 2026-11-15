from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from rewriter.transformer_chisel import TransformerChisel
from rewriter.chisel_factory import chisel_architecture,build_chisel,validate_checkpoint_architecture
from tests.test_prefix_cache import toy_record


def fixture():
    snap,_=toy_record()
    # Four observed boundaries padded to GPT's25, eight anchor directions.
    tr=torch.randn(5,100)
    return replace(snap,trajectories=tr,trajectory_mask=torch.tensor([True]*4+[False]*21))


def test_masked_values_do_not_change_actor_and_valid_values_do():
    torch.set_num_threads(1);torch.manual_seed(8)
    snap=fixture();actor=TransformerChisel(rank=2,trajectory_dim=100,model_dim=256,num_layers=1,patch_size=32,trajectory_mask_features=True)
    x=actor._inputs(snap)[0]
    corrupt=snap.trajectories.clone();corrupt[:,16:]=float('nan')
    y=actor._inputs(replace(snap,trajectories=corrupt))[0]
    torch.testing.assert_close(x,y,rtol=0,atol=0)
    different=snap.trajectories.clone();different[:,:4]+=1
    assert not torch.equal(x,actor._inputs(replace(snap,trajectories=different))[0])
    action=actor(snap)
    assert all(torch.count_nonzero(v)==0 for v in action.vectors.values())
    for f in action.matrices.values():assert torch.count_nonzero(f.P)==torch.count_nonzero(f.Q)==0


def test_old_architecture_unchanged_and_rejects_padded_input():
    actor=TransformerChisel(rank=2,trajectory_dim=100,model_dim=256,num_layers=1,patch_size=32)
    assert actor.trajectory_input.in_features==102
    with pytest.raises(ValueError,match='explicit new'):actor._inputs(fixture())
    snap=replace(fixture(),trajectory_mask=None)
    assert torch.isfinite(actor._inputs(snap)[0]).all()


def test_architecture_and_resume_pin_mask_features():
    args=SimpleNamespace(chisel_model_dim=256,chisel_layers=1,chisel_patch_size=32)
    old=chisel_architecture(args,rank=8,trajectory_dim=400)
    assert 'trajectory_mask_features' not in old
    args.chisel_trajectory_mask_features=True
    new=chisel_architecture(args,rank=8,trajectory_dim=400)
    assert build_chisel(new).trajectory_input.in_features==427
    with pytest.raises(ValueError):validate_checkpoint_architecture({'metadata':{'architecture':old}},new)


@pytest.mark.parametrize('mask',[torch.zeros(25,dtype=torch.bool),torch.ones(25),torch.ones(2,25,dtype=torch.bool)])
def test_invalid_masks_rejected(mask):
    with pytest.raises(ValueError):replace(fixture(),trajectory_mask=mask)


def test_snapshot_copy_and_prefix_serialization_preserve_optional_mask():
    from rewriter.transformer_action import _copy_observation,_tensors
    from rewriter.prefix_cache import pack_record,unpack_record
    snap,train=toy_record()
    assert _copy_observation(snap).trajectory_mask is None
    assert _tensors(snap)
    snap=replace(snap,trajectory_mask=torch.tensor([True,False,True]))
    copied=_copy_observation(snap)
    assert torch.equal(copied.trajectory_mask,snap.trajectory_mask)
    assert copied.trajectory_mask.data_ptr()!=snap.trajectory_mask.data_ptr()
    restored=unpack_record(pack_record(0,snap,train)).snapshot
    assert torch.equal(restored.trajectory_mask,snap.trajectory_mask)
