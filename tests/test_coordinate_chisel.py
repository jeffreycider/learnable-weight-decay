import copy
import pytest
import torch
from rewriter.coordinate_chisel import CoordinateChisel,CheckpointProbes,ARCHITECTURE
from tests.test_controller import make_snapshot


def fixture():
    torch.set_num_threads(1);torch.manual_seed(141)
    snapshot=make_snapshot();actor=CoordinateChisel(rank=2,trajectory_dim=12,model_dim=128,num_layers=2,latent_tokens=3)
    private=actor.make_private(snapshot,state_key=('gpt',0,6),seed=73)
    return snapshot,actor,private


def test_explicit_private_ownership_and_identity_initialization():
    snapshot,actor,private=fixture();before=torch.get_rng_state().clone()
    other=actor.make_private(snapshot,state_key=('gpt',0,7),seed=74)
    assert torch.equal(before,torch.get_rng_state())
    assert set(map(id,actor.parameters())).isdisjoint(map(id,private.parameters()))
    assert set(map(id,other.parameters())).isdisjoint(map(id,private.parameters()))
    action=actor(snapshot,private)
    assert action.norm_sq()==0
    for frame in [*private.matrix_rows,*private.matrix_columns,*private.vector_frames]:
        torch.testing.assert_close(frame.encoder,frame.decoder,atol=0,rtol=0)
        assert frame.encoder.data_ptr()!=frame.decoder.data_ptr() and frame.decoder.norm()>0
    assert actor.configuration()['name']==ARCHITECTURE


def test_one_vjp_reaches_shared_heads_then_private_encoders_and_decoders():
    snapshot,actor,private=fixture();shared=torch.optim.Adam(actor.parameters(),lr=1e-3);local=torch.optim.Adam(private.parameters(),lr=1e-3)
    action=actor(snapshot,private);action.contraction(snapshot.weights,snapshot.vector_weights).backward()
    assert actor.matrix_output.weight.grad.norm()>0
    assert private.matrix_rows[0].encoder.grad is not None and private.matrix_rows[0].encoder.grad.count_nonzero()==0
    shared.step();local.step();shared.zero_grad();local.zero_grad()
    action=actor(snapshot,private);action.contraction(snapshot.weights,snapshot.vector_weights).backward()
    assert private.matrix_rows[0].encoder.grad.norm()>0
    assert private.matrix_rows[0].decoder.grad.norm()>0
    # Zero shared output heads and zero residual projections unblock in stages:
    # output heads first, residual projections second, global trajectory mixing third.
    assert private.trajectory_input.weight.grad.count_nonzero()==0
    shared.step();local.step();shared.zero_grad();local.zero_grad()
    action=actor(snapshot,private);action.contraction(snapshot.weights,snapshot.vector_weights).backward()
    assert private.trajectory_input.weight.grad.norm()>0
    assert actor.matrix_input.weight.grad.norm()>0
    for f in action.matrices.values():torch.testing.assert_close(f.anchors.A.T@f.P,torch.zeros_like(f.anchors.A.T@f.P),atol=2e-6,rtol=0)


def test_all_reads_precede_writes_and_checkpoint_identity_is_strict():
    snapshot,actor,private=fixture();x,reads,matrix_count=actor._inputs(snapshot,private)
    counts=actor.token_counts(snapshot)
    assert reads==counts['observation_tokens'] and x.shape==(1,counts['total_tokens'],128)
    assert matrix_count==2*len(snapshot.matrices)*3
    same=actor.make_private(snapshot,state_key=('gpt',0,6),seed=5);same.load_state_dict(copy.deepcopy(private.state_dict()))
    other=actor.make_private(snapshot,state_key=('gpt',0,7),seed=5)
    with pytest.raises(ValueError,match='identity'):other.load_state_dict(private.state_dict())


def test_metadata_bypasses_private_pooling_and_stays_constant():
    snapshot,actor,private=fixture()
    for frame,features,dynamic in [(private.matrix_rows[0],snapshot.matrices['first'].rows,list(range(11))),
                                   (private.vector_frames[0],snapshot.vectors['gain'].coordinates,[0,2,4,6])]:
        constant=[i for i in range(features.shape[1]) if i not in dynamic]
        features=features.clone();features[:,constant]=torch.arange(len(constant)).float()+1
        with torch.no_grad():frame.encoder.normal_()
        result=frame.encode_with_constants(features,dynamic)
        torch.testing.assert_close(result[:,constant],features[:1,constant].expand(frame.encoder.shape[1],-1),atol=0,rtol=0)
        torch.testing.assert_close(result[:,dynamic],frame.encode(features[:,dynamic]))


def test_decoder_preserves_average_coordinate_variance_units():
    snapshot,actor,private=fixture()
    # For independent unit-variance latent entries, the total output variance is n,
    # independent of n/L. This is an exact Frobenius calculation, not sampling.
    for frame in [*private.matrix_rows,*private.matrix_columns,*private.vector_frames]:
        operator=frame.decode(torch.eye(frame.decoder.shape[1]))
        torch.testing.assert_close(operator.square().sum(),torch.tensor(float(frame.decoder.shape[0])),rtol=2e-6,atol=2e-6)
