import copy
import dataclasses
import math

import pytest
import torch

from rewriter.adaptive_coordinate_chisel import AdaptiveCoordinateChisel, ARCHITECTURE
from rewriter.coordinate_chisel import CheckpointProbes, TiedOrthonormalFrame
from tests.test_controller import make_snapshot


def setup(l=3):
    torch.set_num_threads(1);torch.manual_seed(141)
    snapshot=make_snapshot()
    actor=AdaptiveCoordinateChisel(rank=2,trajectory_dim=12,model_dim=128,num_layers=2,latent_tokens=l)
    private=actor.make_private(snapshot,state_key=('gpt',0,6),seed=73)
    return snapshot,actor,private


@pytest.mark.parametrize('l',[3,8])
def test_zero_head_then_frame_rotation_and_shared_global_learning(l):
    snapshot,actor,private=setup(l)
    shared=torch.optim.SGD(actor.parameters(),lr=.001)
    weights={name:torch.randn(obs.rows.shape[0],obs.columns.shape[0]) for name,obs in snapshot.matrices.items()}
    vector_weights={name:torch.randn_like(obs.weight) for name,obs in snapshot.vectors.items()}
    action=actor(snapshot,private)
    assert action.norm_sq()==0
    loss=action.contraction(weights,vector_weights);loss.backward()
    assert actor.matrix_output.weight.grad.norm()>0
    assert actor.vector_output.weight.grad.norm()>0
    assert all(p.grad is None or p.grad.count_nonzero()==0 for p in private.parameters())
    shared.step();shared.zero_grad();private.zero_grad()
    action=actor(snapshot,private)
    action.contraction(weights,vector_weights).backward()
    assert private.matrix_rows[0].raw.grad.norm()>0
    assert private.matrix_columns[0].raw.grad.norm()>0
    assert private.vector_frames[0].raw.grad.norm()>0
    # Gradient rotation changes a physical frame, not just its raw scale.
    frame=private.matrix_rows[0];before=frame.orthonormal_matrix().detach().clone()
    with torch.no_grad():frame.raw.sub_(.001*frame.raw.grad)
    assert (frame.orthonormal_matrix()-before).norm()>0
    shared.step();shared.zero_grad();private.zero_grad()
    actor(snapshot,private).contraction(weights,vector_weights).backward()
    assert private.trajectory_input.weight.grad.norm()>0


def test_exact_row_column_readout_scaling_and_nominal_energy_for_tall_and_wide_frames():
    snapshot,actor,private=setup(8)
    with torch.no_grad():actor.matrix_output.weight.normal_(std=.05)
    captured=[]
    hook=actor.matrix_output.register_forward_hook(lambda module,args,out:captured.append(out))
    action=actor(snapshot,private);hook.remove()
    latent=captured[0].reshape(-1,8,8)
    for i,(name,f) in enumerate(action.matrices.items()):
        n,m=f.U.shape[0],f.V.shape[0]
        scale=math.sqrt(n*m/(min(n,8)*min(m,8)))
        torch.testing.assert_close(f.C,scale*(latent[2*i]+latent[2*i+1].T)/math.sqrt(2))
        torch.testing.assert_close(f.norm_sq(),f.dense().square().sum())
        # Sum squared responses to the standard core basis = expected norm²
        # for an independent unit-variance core; no stochastic approximation.
        energy=sum((scale*f.U[:,j,None]*f.V[None,:,k]).square().sum() for j in range(8) for k in range(8))
        torch.testing.assert_close(energy,energy.new_tensor(float(n*m)))
        assert torch.linalg.matrix_rank(f.dense())<=min(n,m,8)
    config=actor.configuration()
    assert config['name']==ARCHITECTURE and not config['fixed_action_anchors']
    assert config['external_group_scale']=='group_RMS_only_no_anchored_dimension_multiplier'


def test_original_anchor_objects_are_sensor_only_and_not_write_factors():
    snapshot,actor,private=setup()
    with torch.no_grad():actor.matrix_output.weight.normal_(std=.05)
    first=actor(snapshot,private)
    changed={}
    for name,obs in snapshot.matrices.items():
        # Hold actual cached fields fixed, replace the old anchor metadata only.
        anchors=dataclasses.replace(obs.anchors,A=-obs.anchors.A,B=-obs.anchors.B)
        changed[name]=dataclasses.replace(obs,anchors=anchors)
    second=actor(dataclasses.replace(snapshot,matrices=changed),private)
    for name in first.matrices:
        torch.testing.assert_close(first.matrices[name].dense(),second.matrices[name].dense(),atol=0,rtol=0)
    with pytest.raises(ValueError,match='sketches'):
        first.contraction(snapshot.weights,snapshot.vector_weights)


def test_strict_private_shared_identity_and_rng_preservation():
    snapshot,actor,private=setup()
    before=torch.get_rng_state().clone()
    other=actor.make_private(snapshot,state_key=('gpt',0,7),seed=99)
    assert torch.equal(before,torch.get_rng_state())
    with pytest.raises(ValueError,match='identity'):
        other.load_state_dict(copy.deepcopy(private.state_dict()))
    legacy=CheckpointProbes(snapshot,model_dim=128,latent_tokens=3,seed=73,state_key=('gpt',0,6),frame_mode='tied_orthonormal')
    with pytest.raises(ValueError,match='identity'):
        private.load_state_dict(legacy.state_dict())
    with pytest.raises(ValueError,match='adaptive'):
        actor(snapshot,legacy)
    payload=copy.deepcopy(actor.state_dict())
    actor.load_state_dict(payload)
    del payload['_extra_state']
    with pytest.raises(ValueError,match='architecture'):
        actor.load_state_dict(payload)


def test_private_frame_qr_vjp_and_observation_perturbations_reach_global_writes():
    snapshot,actor,private=setup()
    with torch.no_grad():
        actor.matrix_output.weight.normal_(std=.05)
        # Enable cross-token mixing; original residual blocks start identity.
        for block in actor.blocks:
            block.attn.proj.weight.normal_(std=.02)
    before=actor(snapshot,private)
    matrices=dict(snapshot.matrices)
    matrices['first']=dataclasses.replace(matrices['first'],rows=matrices['first'].rows+1)
    after=actor(dataclasses.replace(snapshot,matrices=matrices),private)
    assert (before.matrices['second'].dense()-after.matrices['second'].dense()).norm()>0
    frame=TiedOrthonormalFrame(5,3,generator=torch.Generator().manual_seed(1)).double()
    raw=frame.raw.detach().clone().requires_grad_()
    def matrix(raw):
        q,r=torch.linalg.qr(raw,mode='reduced')
        return q*torch.where(r.diagonal()<0,-1.,1.)
    assert torch.autograd.gradcheck(matrix,(raw,))
