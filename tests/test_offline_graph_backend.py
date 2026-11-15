from types import SimpleNamespace
from unittest.mock import Mock
import pytest
import torch
from rewriter.offline_graph_backend import OfflineGraphAccumulator


def fake_adapter():
    actor=torch.nn.Linear(2,1)
    engine=SimpleNamespace(device=torch.device('cuda:0'),gpt_precision='fp32',kl_dtype=torch.float64)
    constants=SimpleNamespace(engine=engine)
    return OfflineGraphAccumulator(actor,actor,engine,None,constants,radius_fraction=.01)


def test_explicit_scope_accepts_qualified_rewards_and_rejects_other_weights():
    a=fake_adapter()
    positive=OfflineGraphAccumulator(a.actor,a.forward,a.engine,None,a.constants,radius_fraction=.01,shrinkage_coefficient=1.)
    assert positive.shrinkage_coefficient==1.
    with pytest.raises(ValueError,match='coefficient'):
        OfflineGraphAccumulator(a.actor,a.forward,a.engine,None,a.constants,radius_fraction=.01,shrinkage_coefficient=.5)


def test_accumulation_requires_complete_exact_local_batch_and_no_optimizer():
    a=fake_adapter();a.begin_update([1,3],global_checkpoint_count=200)
    with pytest.raises(RuntimeError,match='incomplete'):a.finish_update()
    with pytest.raises(RuntimeError,match='active'):a.begin_update([1],global_checkpoint_count=200)
    with pytest.raises(ValueError,match='exactly once'):
        a.accumulate(SimpleNamespace(completed_steps=2),None,SimpleNamespace(multipliers={2:0.}))
    a._seen={1,3}
    for g in a.buffers:g.fill_(.25)
    before=[p.detach().clone() for p in a.parameters]
    assert a.finish_update() is a.buffers
    for p,g,b in zip(a.parameters,a.buffers,before):
        assert p.grad is g
        assert torch.equal(p,b)
    a.begin_update([1,3],global_checkpoint_count=200)
    assert all(torch.count_nonzero(g)==0 for g in a.buffers)
    a.close()


def test_failed_bind_prevents_partial_update_publication():
    a=fake_adapter();a.begin_update([1],global_checkpoint_count=200)
    a._capture=Mock(side_effect=ValueError('wrong SHA'))
    with pytest.raises(ValueError,match='wrong SHA'):
        a.accumulate(SimpleNamespace(completed_steps=1),SimpleNamespace(inputs=torch.ones(16,2)),SimpleNamespace(multipliers={1:0.}))
    with pytest.raises(RuntimeError,match='failed'):a.finish_update()
    with pytest.raises(RuntimeError,match='valid'):a.accumulate(SimpleNamespace(completed_steps=1),None,None)
    a.close();a.begin_update([1],global_checkpoint_count=200)


def test_compute_counts_capture_and_replay_without_teacher_transformer():
    a=fake_adapter();a.engine.microbatch_sequences=16
    inputs=torch.ones(64,1024)
    a._count_execution(inputs,3);a._count_execution(inputs)
    assert a.compute['student_forwards']==16
    assert a.compute['student_backwards']==16
    assert a.compute['cached_teacher_head_tokens']==4*64*1024
    assert a.compute['chisel_forwards']==4
    assert 'teacher_forwards' not in a.compute


def test_empty_rank_publishes_zero_for_the_global_sum():
    a=fake_adapter();a.begin_update([],global_checkpoint_count=3)
    for p in a.parameters:p.grad=torch.ones_like(p)
    a.finish_update()
    assert all(torch.count_nonzero(p.grad)==0 for p in a.parameters)


def test_training_guards_are_explicit_and_leave_reference_default_alone():
    from rewriter.offline_graph_backend import validate_training_options
    options=dict(gpt_precision='fp32',norm_backend='fused',objective='primal-dual',
        actor_backend='transformer',device='cuda',shrinkage_coefficient=0.,
        sequences_per_checkpoint=64,gpt_microbatch=16,checkpoint_batch=200,
        teacher_cache='cache',offline_systems_cache=True,eager_chisel=False,
        compile_gpt=False,gradient_diagnostics=False,no_component_timing=True)
    validate_training_options(SimpleNamespace(**options))
    for key,value in [('shrinkage_coefficient',.5),('sequences_per_checkpoint',32),
                      ('checkpoint_batch',100),('gpt_precision','baseline'),
                      ('gradient_diagnostics',True),('teacher_cache',None)]:
        with pytest.raises(ValueError):validate_training_options(SimpleNamespace(**dict(options,**{key:value})))


@pytest.mark.parametrize('coefficient',[0.,1.])
def test_actual_trainer_graph_branch_keeps_adam_dual_and_audits(tmp_path,monkeypatch,coefficient):
    """CPU dense stand-in tests trainer wiring; real graph is GPU-qualified separately."""
    from tests.test_offline_integration import test_integrated_offline_flags_exact_two_epoch_adam
    import scripts.train_multistate_chisel as train
    import rewriter.offline_graph_backend as module
    from rewriter.edit_scope import scoped_snapshot
    original_main=train.main
    instances=[]
    class DenseStandIn:
        def __init__(self,actor,forward,engine,cache,constants,**options):
            self.actor=actor;self.forward=forward;self.engine=engine;self.cache=cache;self.constants=constants
            self.parameters=list(actor.parameters());self.buffers=[torch.zeros_like(p) for p in self.parameters]
            self.compute={};self.graph=None;self.closes=0;self.batches=0;self.state_ids=[]
            self.previous_heads=0;instances.append(self)
        def begin_update(self,ids,*,global_checkpoint_count):
            self.count=global_checkpoint_count;self.batches+=1
            torch._foreach_zero_(self.buffers)
        def accumulate(self,record,panel,dual):
            self.state_ids.append(record.completed_steps)
            self.constants.load(self.cache,record);grouped=self.constants.grouped()
            self.actor.zero_grad(set_to_none=True)
            params,regularizer,stats,_,credit=train.build_proposal(self.forward,record.snapshot,self.engine,grouped,
                radius_fraction=.01,norm_squared=self.constants.radius_squared(grouped),shrinkage_coefficient=coefficient)
            measured=train.mean_kl_and_backward(self.engine,params,regularizer,panel.inputs,kl_budget=.1,
                state_weight=1/self.count,actor_credit=credit,teacher_provider=lambda a,b:panel.teacher_batch(self.engine,a,b),
                primal_dual=dual,state_id=record.completed_steps)
            torch._foreach_add_(self.buffers,[p.grad for p in self.parameters])
            self.graph=SimpleNamespace(grouped=grouped)
            return measured,stats
        def finish_update(self):
            for p,g in zip(self.parameters,self.buffers):p.grad=g
        def close(self):self.graph=None;self.closes+=1
    def run(args):
        args=[*[a for a in args if a!='--gradient-diagnostics'],'--objective','primal-dual','--shrinkage-coefficient',str(coefficient)]
        if '--offline-systems-cache' in args:args+=['--offline-backend','strict_graph']
        return original_main(args)
    monkeypatch.setattr(module,'validate_training_options',lambda args:None)
    monkeypatch.setattr(module,'OfflineGraphAccumulator',DenseStandIn)
    monkeypatch.setattr(train,'main',run)
    # This pre-existing real trainer test checks exact controller/Adam/dual state
    # over four optimizer updates plus audits, panel IDs and complete metrics.
    test_integrated_offline_flags_exact_two_epoch_adam(tmp_path)
    assert len(instances)==1 and instances[0].batches==4
    assert instances[0].closes>=2
    assert len(instances[0].state_ids)==8
