import copy
from types import SimpleNamespace
import pytest
import torch
from rewriter.adaptive_online import AdaptiveOnlineBinding,AdaptiveOnlineLifecycle
from rewriter.grouped_coordinate_chisel import GroupedCoordinateChisel
from rewriter.adaptive_preservation import build_adaptive_proposal
from rewriter.grouped_norm import GroupedNormObjective
from scripts.fit_cpu_edit_teacher import exact_state_equal
from tests.test_adaptive_preservation_actor import actual_actor_case


def fixture(radius=.01,run_id='adaptive-live'):
    torch.set_num_threads(1)
    engine,_,_,snapshot,_,_=actual_actor_case()
    torch.manual_seed(23)
    actor=GroupedCoordinateChisel(rank=2,trajectory_dim=12,model_dim=128,num_layers=1,latent_tokens=2,
        fresh_weight_core=True,factor_precision='fp64',residual_precision='fp32')
    private=actor.make_private(snapshot,state_key=('fixture',6),seed=74)
    with torch.no_grad():
        for bank in actor.output_heads:
            for head in bank.values():head.weight.normal_(std=.002)
    rng=torch.get_rng_state().clone();binding=AdaptiveOnlineBinding.from_fitted(actor,private,snapshot,run_id=run_id)
    assert torch.equal(rng,torch.get_rng_state())
    shared=torch.optim.Adam(binding.shared.parameters(),lr=1e-7,betas=(.9,.99),eps=1e-8)
    probes=torch.optim.Adam(binding.private.parameters(),lr=1e-6,betas=(.9,.99),eps=1e-12)
    life=AdaptiveOnlineLifecycle(binding,shared,probes,global_token_count=128,final_action_step=3,radius_fraction=radius,
        shared_clip=1e20,private_clip=1e20,parameter_view='unwrapped_stored_fp32')
    live={n:p.clone().requires_grad_() for n,p in engine.parameters.items()}
    return life,live,snapshot,actor,private


def explicit(binding,snapshot,base,radius):
    engine=SimpleNamespace(parameters=base,transpose_storage={n:n=='embed.weight' for n in base})
    grouped=GroupedNormObjective(base,zero_reference_policy='freeze')
    return build_adaptive_proposal(lambda s:binding(s,base,embedding_names=('embed.weight',)),snapshot,engine,grouped,
        radius_fraction=radius,detach_credit=False)[0]


@pytest.mark.parametrize('radius',[1e-8,100.])
def test_exact_offline_commit_and_delayed_shared_private_vjp(radius):
    life,live,snapshot,source_actor,source_private=fixture(radius)
    original_actor=copy.deepcopy(source_actor.state_dict());original_private=copy.deepcopy(source_private.state_dict())
    reference=copy.deepcopy(life.binding);base={n:p.detach().clone() for n,p in live.items()}
    rewritten=explicit(reference,snapshot,base,radius)
    expected=torch.autograd.grad(sum(v.sin().square().sum() for v in rewritten.values()),tuple(reference.parameters()),allow_unused=True)
    row=life.commit(snapshot,live,action_step=0,base_step_completed=0)
    assert row['radius_saturated']==(radius<1)
    for name,value in live.items():torch.testing.assert_close(value,rewritten[name],atol=0,rtol=0)
    # One ordinary task gradient at the committed stored representation.
    gradients=torch.autograd.grad(sum(v.sin().square().sum() for v in live.values()),tuple(live.values()))
    task={n:g*128 for n,g in zip(live,gradients)};oldtask={n:g.clone() for n,g in task.items()}
    # Sensor refresh cannot affect the retained graph or the old private frames.
    snapshot.trajectories.add_(17.)
    result=life.consume(task,gradient_step=1)
    for p,wanted in zip(life.binding.parameters(),expected):
        if wanted is None:assert p.grad is None
        else:torch.testing.assert_close(p.grad,wanted,atol=3e-6,rtol=3e-5)
    assert any(p.grad is not None and p.grad.abs().max()>0 for p in life.binding.shared.parameters())
    assert any(p.grad is not None and p.grad.abs().max()>0 for p in life.binding.private.parameters())
    assert all(torch.equal(task[n],oldtask[n]) for n in task)
    assert exact_state_equal(source_actor.state_dict(),original_actor) and exact_state_equal(source_private.state_dict(),original_private)
    assert result['gpt_forwards']==result['gpt_backwards']==0


def test_pending_roundtrip_preserves_both_optimizers_and_rng():
    life,live,snapshot,_,_=fixture();life.commit(snapshot,live,action_step=0,base_step_completed=0)
    life.consume({n:v.detach()*128 for n,v in live.items()},gradient_step=1)
    with torch.no_grad():
        for v in live.values():v.add_(1e-5)  # ordinary base optimizer happened after consume
    life.commit(snapshot,live,action_step=1,base_step_completed=1);saved=life.state_dict()
    other,restored,_,_,_=fixture()
    with torch.no_grad():
        for n,v in restored.items():v.copy_(live[n])
    rng=torch.get_rng_state().clone();other.load_state_dict(saved,restored);assert torch.equal(rng,torch.get_rng_state())
    task={n:v.detach()*128 for n,v in live.items()};life.consume(task,gradient_step=2);other.consume(task,gradient_step=2)
    assert exact_state_equal(life.binding.state_dict(),other.binding.state_dict())
    assert exact_state_equal(life.shared_optimizer.state_dict(),other.shared_optimizer.state_dict())
    assert exact_state_equal(life.private_optimizer.state_dict(),other.private_optimizer.state_dict())


@pytest.mark.parametrize('mutation',['shared','private','live','snapshot'])
def test_pending_versions_reject_mutation_before_credit(mutation):
    life,live,snapshot,_,_=fixture();life.commit(snapshot,live,action_step=0,base_step_completed=0)
    value={'shared':next(life.binding.shared.parameters()),'private':next(life.binding.private.parameters()),
        'live':next(iter(live.values())),'snapshot':life.pending['snapshot'].trajectories}[mutation]
    with torch.no_grad():value.add_(1e-5)
    with pytest.raises(RuntimeError,match='mutated'):life.consume({n:v.detach()*128 for n,v in live.items()},gradient_step=1)


def test_explicit_ownership_wrapper_terminal_and_identity_guards():
    life,live,snapshot,_,_=fixture();before={n:v.clone() for n,v in live.items()}
    with pytest.raises(ValueError,match='base optimizer'):life.commit(snapshot,live,action_step=0,base_step_completed=1)
    terminal=life.commit(snapshot,live,action_step=3,base_step_completed=3)
    assert terminal['terminal_identity'] and not life.lag.pending
    assert all(torch.equal(v,before[n]) for n,v in live.items())
    other,_,_,_,_=fixture(run_id='other')
    with pytest.raises(ValueError,match='identity'):other.binding.load_state_dict(life.binding.state_dict())
    with pytest.raises(ValueError,match='lookahead'):
        AdaptiveOnlineLifecycle(life.binding,life.shared_optimizer,life.private_optimizer,global_token_count=128,final_action_step=3,
            radius_fraction=.01,shared_clip=1.,private_clip=1.,parameter_view='lookahead')
    wrong=torch.optim.Adam(life.binding.shared.parameters(),lr=1e-7)
    with pytest.raises(ValueError,match='own'):
        AdaptiveOnlineLifecycle(life.binding,life.shared_optimizer,wrong,global_token_count=128,final_action_step=3,
            radius_fraction=.01,shared_clip=1.,private_clip=1.,parameter_view='unwrapped_stored_fp32')


def test_fresh_weight_sensing_keeps_private_derivative_and_freezes_base():
    from rewriter.adaptive_sensing import query_weight_cores
    life,live,snapshot,_,_=fixture()
    binding=life.binding
    cores=query_weight_cores(live,binding.private)
    credit=sum(c.square().sum() for c in cores.values())
    private_grad=torch.autograd.grad(credit,tuple(binding.private.parameters()),allow_unused=True,retain_graph=True)
    base_grad=torch.autograd.grad(credit,tuple(live.values()),allow_unused=True)
    assert all(g is None for g in base_grad)
    assert any(g is not None and g.abs().max()>0 for g in private_grad)
    assert all(not v.requires_grad for v in life._prepare(snapshot,live)['base'].values())


def test_pending_restore_rejects_mismatched_live_values_and_dtype_before_mutation():
    life,live,snapshot,_,_=fixture();life.commit(snapshot,live,action_step=0,base_step_completed=0);saved=life.state_dict()
    other,restored,_,_,_=fixture();before=copy.deepcopy(other.binding.state_dict())
    with pytest.raises(ValueError,match='live weights'):other.load_state_dict(saved,restored)
    assert exact_state_equal(before,other.binding.state_dict())
    with pytest.raises(ValueError,match='FP32'):other.load_state_dict(saved,{n:v.double() for n,v in live.items()})
    wrong=copy.deepcopy(saved);wrong['meta_updates']=-1
    with pytest.raises(ValueError,match='clock'):other.load_state_dict(wrong,live)


def test_pending_guard_rejects_owned_postbase_mutation_and_wrong_gradient_clock():
    life,live,snapshot,_,_=fixture();life.commit(snapshot,live,action_step=0,base_step_completed=0)
    with pytest.raises(ValueError,match='next ordinary'):life.consume({n:v.detach()*128 for n,v in live.items()},gradient_step=2)
    with torch.no_grad():next(iter(life.pending['base'].values())).add_(.01)
    with pytest.raises(RuntimeError,match='mutated'):life.state_dict()


def test_missing_readonly_gradient_requires_explicit_zero_and_keeps_pending():
    life,live,snapshot,_,_=fixture();life.commit(snapshot,live,action_step=0,base_step_completed=0)
    name=next(n for n,v in live.items() if v.ndim==1 and n not in life.pending['editable'])
    gradients={n:v.detach()*128 for n,v in live.items()};gradients[name]=None
    with pytest.raises(ValueError,match='explicit zeros'):life.consume(gradients,gradient_step=1)
    assert life.lag.pending and life.meta_updates==0
    gradients[name]=torch.zeros_like(live[name]);life.consume(gradients,gradient_step=1)
    assert life.meta_updates==1
