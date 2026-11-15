import copy
import pytest
import torch
from rewriter.coordinate_initialization import weight_qr_frame,initialize_cached_weight_qr,initialize_warm_private
from tests.test_coordinate_chisel import fixture


def test_weight_basis_and_rank_deficient_completion_are_deterministic():
    torch.manual_seed(13);w=torch.randn(12,2);w=torch.cat([w,w[:,:1]],1);fallback=torch.randn(12,4)
    q,stats=weight_qr_frame(w,fallback)
    assert stats['weight_rank']==2 and stats['complement_columns']==2
    torch.testing.assert_close(q.T@q,torch.eye(4),atol=2e-7,rtol=0)
    torch.testing.assert_close(q@(q.T@w),w,atol=5e-7,rtol=1e-6)
    assert torch.equal(q,weight_qr_frame(w,fallback)[0])
    perm=torch.randperm(12);qp,_=weight_qr_frame(w[perm],fallback[perm])
    torch.testing.assert_close(qp,q[perm],atol=2e-7,rtol=0)


def test_zero_and_short_coordinate_axes_have_finite_unit_columns():
    for n,latent in [(1,8),(3,8),(8,8)]:
        q,stats=weight_qr_frame(torch.zeros(n,2),torch.zeros(n,latent))
        assert torch.isfinite(q).all() and stats['weight_rank']==0
        torch.testing.assert_close(q.norm(dim=0),torch.ones(latent),atol=0,rtol=0)
        assert stats['repeated_columns']==max(0,latent-n)


def test_initializer_changes_only_fresh_frames_not_sensors_trajectory_actor_or_rng():
    snapshot,actor,private=fixture();saved=copy.deepcopy(snapshot);shared=copy.deepcopy(actor.state_dict())
    trajectory=copy.deepcopy(private.trajectory_input.state_dict());rng=torch.get_rng_state().clone()
    pointers=[p.data_ptr() for p in private.parameters()]
    report=initialize_cached_weight_qr(private,snapshot)
    assert report['gpt_forwards']==report['backwards']==report['optimizer_steps']==0
    assert pointers==[p.data_ptr() for p in private.parameters()] and torch.equal(rng,torch.get_rng_state())
    for n,p in actor.state_dict().items():assert torch.equal(p,shared[n])
    for n,p in private.trajectory_input.state_dict().items():assert torch.equal(p,trajectory[n])
    for n,obs in snapshot.matrices.items():
        for a,b in zip(obs.weight_probes,saved.matrices[n].weight_probes):assert torch.equal(a,b)
        assert torch.equal(obs.anchors.A,saved.matrices[n].anchors.A) and torch.equal(obs.rows,saved.matrices[n].rows)
    for frame in [*private.matrix_rows,*private.matrix_columns,*private.vector_frames]:
        assert torch.equal(frame.encoder,frame.decoder)
        torch.testing.assert_close(frame.encoder.norm(dim=0),torch.ones(private.latent_tokens),atol=2e-7,rtol=0)
    assert actor(snapshot,private).norm_sq()==0
    private.matrix_rows[0].encoder.grad=torch.zeros_like(private.matrix_rows[0].encoder)
    with pytest.raises(ValueError,match='gradient-cleared'):initialize_cached_weight_qr(private,snapshot)


def test_warm_start_preserves_new_identity_and_copies_no_optimizer_state():
    snapshot,actor,source=fixture();target=actor.make_private(snapshot,state_key=('gpt',0,501),seed=99)
    pointers=[p.data_ptr() for p in target.parameters()];key=target.state_key
    report=initialize_warm_private(target,source.state_dict(),source_state_sha256='a'*64,anchors_equal=True)
    assert target.state_key==key and target.get_extra_state()['state_key']==key
    assert pointers==[p.data_ptr() for p in target.parameters()]
    for a,b in zip(target.parameters(),source.parameters()):assert torch.equal(a,b)
    assert report['optimizer_steps']==0 and report['source_state_key']!=key
    before=copy.deepcopy(target.state_dict());bad=copy.deepcopy(source.state_dict());bad['_extra_state']['model_dim']=999
    with pytest.raises(ValueError,match='layout'):initialize_warm_private(target,bad,source_state_sha256='a'*64,anchors_equal=True)
    for n,p in target.named_parameters():assert torch.equal(p,before[n])
    with pytest.raises(ValueError,match='anchors'):initialize_warm_private(target,source.state_dict(),source_state_sha256='a'*64,anchors_equal=False)


def test_cached_weight_initializer_rejects_tied_computed_views():
    from rewriter.coordinate_chisel import CoordinateChisel
    from tests.test_controller import make_snapshot
    snapshot=make_snapshot();actor=CoordinateChisel(rank=2,trajectory_dim=12,model_dim=128,num_layers=1,latent_tokens=2,frame_mode='tied_orthonormal')
    private=actor.make_private(snapshot,state_key=('gpt',0,6),seed=1)
    before={n:p.detach().clone() for n,p in private.named_parameters()}
    with pytest.raises(ValueError,match='independent'):initialize_cached_weight_qr(private,snapshot)
    assert all(torch.equal(p,before[n]) for n,p in private.named_parameters())
