"""CPU mechanics only: these checks do not establish held-out transfer."""
import copy
import pytest
import torch
from rewriter.coordinate_transfer_control import frozen_block_pair, fresh_paired_private
from rewriter.distributed_probe_store import _digest
from tests.test_coordinate_tied_frames import actor_fixture

@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def fitted_fixture():
    snapshot, initial, _ = actor_fixture()
    trained=copy.deepcopy(initial)
    with torch.no_grad():
        for head in trained.output_heads:head.weight.normal_(std=.01)
        for block in trained.blocks:
            block.attn.proj.weight.normal_(std=.005)
            block.mlp.proj.weight.normal_(std=.005)
    return snapshot,trained,initial


def test_initial_stack_is_identity_but_paired_private_control_is_not_dead():
    s,trained,initial=fitted_fixture()
    original=_digest(trained.state_dict());rng=torch.get_rng_state().clone()
    learned,control,meta=frozen_block_pair(trained,initial)
    assert torch.equal(rng,torch.get_rng_state())
    p,q=fresh_paired_private(learned,control,s,state_key=('gpt',0,5,'heldout'),seed=73)
    assert _digest(p.state_dict())==_digest(q.state_dict())
    tokens=torch.randn(1,7,128)
    value=tokens
    for block in control.blocks:value=block(value)
    assert torch.equal(tokens,value)
    results=[]
    for actor,private in ((learned,p),(control,q)):
        opt=torch.optim.Adam(private.parameters(),lr=1e-4,betas=(.9,.99),eps=1e-12)
        assert not opt.state
        before=_digest(private.state_dict());shared=_digest(actor.state_dict())
        action=actor(s,private)
        loss=action.contraction(s.weights,s.vector_weights)+.01*action.norm_sq()
        loss.backward()
        assert sum(float(v.grad.square().sum()) for v in private.parameters() if v.grad is not None)>0
        assert all(v.grad is None for v in actor.parameters())
        opt.step();results.append(float(loss.detach()))
        assert before!=_digest(private.state_dict())
        assert shared==_digest(actor.state_dict())
    assert results[0]!=results[1]
    assert original==_digest(trained.state_dict())
    assert meta['independent_identity_stack_training'] is False


def test_zero_head_naive_control_has_zero_private_gradient_and_is_rejected():
    s,initial,private=actor_fixture();initial.requires_grad_(False)
    action=initial(s,private)
    (action.contraction(s.weights,s.vector_weights)+action.norm_sq()).backward()
    assert all(p.grad is None or not torch.count_nonzero(p.grad) for p in private.parameters())
    with pytest.raises(ValueError,match='dead zero'):frozen_block_pair(initial,initial)


def test_control_changes_only_blocks_and_rejects_invalid_initialization():
    s,trained,initial=fitted_fixture();learned,control,_=frozen_block_pair(trained,initial)
    for name,value in learned.state_dict().items():
        target=initial.state_dict()[name] if name.startswith('blocks.') else value
        assert torch.equal(control.state_dict()[name],target)
    invalid=copy.deepcopy(initial)
    with torch.no_grad():invalid.blocks[0].attn.proj.bias[0]=1
    with pytest.raises(ValueError,match='zero-output'):frozen_block_pair(trained,invalid)
    learned.requires_grad_(True)
    with pytest.raises(ValueError,match='frozen'):fresh_paired_private(learned,control,s,state_key=('gpt',0,5),seed=1)
