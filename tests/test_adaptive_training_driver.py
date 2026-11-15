"""Opt-in adaptive family: recipe guards, finite dispatch and exact joint resume."""
import copy,datetime,hashlib,json
from pathlib import Path
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from scripts.train_coordinate_chisel import build_actor,validate_recipe,run_training,measure_context
from scripts.qualify_coordinate_chisel import FixedContexts,proposal_builder_for_family
from rewriter.private_probe_slot import PrivateProbeSlot
from rewriter.coordinate_chisel import CoordinateChisel
from rewriter.tokenwise_reference import TokenwiseConfig,TokenwiseDual
from tests.test_coordinate_training_driver import recipe,ToyContexts
from tests.test_controller import make_snapshot
from tests.test_distributed_probe_store import assert_tree_equal


def adaptive_recipe():
    r=recipe();r.update(action_family='adaptive_ucv_v1',student_backend='dense',private_actor_backend='eager',private_inner_steps=2)
    r['architecture']['frame_mode']='tied_orthonormal';r['readout']['heldout']['enabled']=False
    r['private_optimizer']['tick_schedule']=dict(cold_steps=2,warm_lr=1e-4)
    r['private_inner_panels']={'gpt':dict(source_size=32,inner_size=16),'cifar10':dict(source_size=256,inner_size=128)}
    return r


def test_default_actor_and_builder_are_unchanged():
    r=recipe();validate_recipe(r)
    torch.manual_seed(7);a=build_actor(r)
    torch.manual_seed(7);b=CoordinateChisel(**r['architecture'])
    assert_tree_equal(a.state_dict(),b.state_dict())
    from scripts.train_multistate_chisel import build_proposal
    assert proposal_builder_for_family('fixed_anchored_v1') is build_proposal
    with pytest.raises(ValueError,match='action family'):proposal_builder_for_family('automatic')


@pytest.mark.parametrize('change',[
    {'actor_backend':'eager_actor_graph'}, {'private_actor_backend':'eager_actor_graph'},
    {'student_backend':'factorized_fp32'}, {'student_backend':None},
    {'offline_backend':'tokenwise_graph'}, {'shared_initialization':{'mode':'checkpoint_actor_only'}},
    {'architecture':dict(recipe()['architecture'],frame_mode='independent')},
])
def test_adaptive_rejects_unqualified_or_old_family_paths(change):
    r=adaptive_recipe();r.update(change)
    with pytest.raises(ValueError):validate_recipe(r)


def test_family_checkpoint_and_private_schema_reject_before_copy():
    r=adaptive_recipe();validate_recipe(r);a=build_actor(r)
    legacy=CoordinateChisel(**r['architecture']);before=copy.deepcopy(a.state_dict())
    with pytest.raises(ValueError,match='architecture'):a.load_state_dict(legacy.state_dict())
    assert_tree_equal(a.state_dict(),before)
    snap=make_snapshot();private=a.make_private(snap,state_key=('gpt',1),seed=8)
    old=legacy.make_private(snap,state_key=('gpt',1),seed=8);slot=PrivateProbeSlot(a,private);before=slot.export_state()
    with pytest.raises(ValueError,match='schema'):slot.stage(old.state_dict(),state_key=('gpt',1))
    assert_tree_equal(slot.export_state(),before)


def test_fixed_context_explicit_adaptive_proposal_uses_real_dense_student():
    from tests.test_adaptive_preservation import case
    engine,forward,shared,private,snapshot,grouped,inputs=case()
    teachers=[engine.cache_teacher(x[None],data_role='synthetic')[0] for x in inputs]
    class Actor:
        def __call__(self,snapshot,unused):return forward(snapshot)
        def configuration(self):return {'action_family':'adaptive_ucv_v1'}
    contexts=FixedContexts.__new__(FixedContexts);contexts.action_family='adaptive_ucv_v1'
    contexts.proposal_builder=proposal_builder_for_family(contexts.action_family);contexts.compute={}
    context=dict(domain='gpt',local_id=3,engine=engine,snapshot=snapshot,grouped=grouped,inputs=inputs,ids=[1,2],norm_squared=None,teacher=lambda a,b:teachers[a])
    config=TokenwiseConfig(mean_tolerance=1e-3,tail_tolerance=1e-3,scale=1e-3,rho=.01,ranking_weight=0.)
    dual=TokenwiseDual([3],config=config,dual_lr=.01)
    row=measure_context(contexts,context,Actor(),None,config,dual,learning=True)
    assert row['action_family']=='adaptive_ucv_v1' and row['fixed_anchor_gradient_banks_used'] is False
    assert contexts.compute['student_forwards']==contexts.compute['student_backwards']==2
    assert contexts.compute['teacher_forwards']==0
    assert shared.grad is not None and any(p.grad is not None and p.grad.norm()>0 for p in private)


def test_fresh_weight_sensing_context_uses_local_weights_and_retains_private_credit():
    from tests.test_adaptive_preservation_actor import actual_actor_case
    from rewriter.adaptive_coordinate_chisel import AdaptiveCoordinateChisel
    engine,_,_,snapshot,grouped,inputs=actual_actor_case()
    torch.manual_seed(133)
    actor=AdaptiveCoordinateChisel(rank=2,trajectory_dim=12,model_dim=128,num_layers=1,
                                  latent_tokens=2,fresh_weight_core=True)
    private=actor.make_private(snapshot,state_key=('test',0,6),seed=77)
    with torch.no_grad():
        for head in actor.output_heads:head.weight.normal_(std=.001)
        actor.blocks[0].attn.proj.weight.normal_(std=.001)
    teachers=[engine.cache_teacher(x[None],data_role='synthetic')[0] for x in inputs]
    contexts=FixedContexts.__new__(FixedContexts);contexts.action_family='adaptive_ucv_v1'
    contexts.proposal_builder=proposal_builder_for_family(contexts.action_family);contexts.compute={}
    context=dict(domain='gpt',local_id=6,engine=engine,snapshot=snapshot,grouped=grouped,
                 inputs=inputs,ids=[1,2],norm_squared=None,teacher=lambda a,b:teachers[a])
    config=TokenwiseConfig(mean_tolerance=1e-3,tail_tolerance=1e-3,scale=1e-3,rho=.01,ranking_weight=0.)
    dual=TokenwiseDual([6],config=config,dual_lr=.01)
    measure_context(contexts,context,actor,private,config,dual,learning=True)
    assert contexts.compute['fresh_weight_core_calls']==1
    assert contexts.compute['fresh_weight_core_matrix_queries']==len(snapshot.matrices)
    assert contexts.compute['fresh_weight_core_weight_elements_scanned']==sum(
        engine.parameters[name].numel() for name in snapshot.matrices)
    assert contexts.compute['student_forwards']==contexts.compute['student_backwards']==2
    assert contexts.compute['teacher_forwards']==0
    assert any(p.grad is not None and p.grad.norm()>0 for name,p in actor.named_parameters() if 'weight_core' in name)
    assert any(p.grad is not None and p.grad.norm()>0 for name,p in private.named_parameters() if name.endswith('.raw'))
    assert all(not p.requires_grad and p.grad is None for p in engine.parameters.values())


def test_fresh_weight_sensing_flag_is_explicit_and_adaptive_only():
    r=recipe();r['architecture']['fresh_weight_core']=True
    with pytest.raises(ValueError,match='adaptive boolean'):validate_recipe(r)
    r=adaptive_recipe();r['architecture']['fresh_weight_core']='true'
    with pytest.raises(ValueError,match='adaptive boolean'):validate_recipe(r)
    r['architecture']['fresh_weight_core']=True;validate_recipe(r)


def test_real_cifar_context_reads_frozen_whitening_without_writing_it():
    from tests.test_cifar_adapter import fixture
    from rewriter.cifar_adapter import observe_cifar_terminal,cifar_groups
    from rewriter.grouped_norm import GroupedNormObjective
    from rewriter.adaptive_coordinate_chisel import AdaptiveCoordinateChisel
    _,engine,inputs,labels=fixture();engine.microbatch_sequences=2
    snapshot,_=observe_cifar_terminal(engine,inputs,labels,momentum={},anchor_seed=87,rank=2,trajectory_images=4)
    actor=AdaptiveCoordinateChisel(rank=2,trajectory_dim=100,model_dim=128,num_layers=1,latent_tokens=2,fresh_weight_core=True)
    private=actor.make_private(snapshot,state_key=('cifar10',0),seed=77)
    with torch.no_grad():
        for head in actor.output_heads:head.weight.normal_(std=.001)
        actor.blocks[0].attn.proj.weight.normal_(std=.001)
    cached=engine._forward(engine.parameters,inputs).detach()
    before={n:p.clone() for n,p in {**engine._all_parameters,**engine.buffers}.items()}
    contexts=FixedContexts.__new__(FixedContexts);contexts.action_family='adaptive_ucv_v1'
    contexts.proposal_builder=proposal_builder_for_family(contexts.action_family);contexts.compute={}
    ids=list(range(len(inputs)))
    context=dict(domain='cifar10',local_id=1,engine=engine,snapshot=snapshot,
                 grouped=GroupedNormObjective(engine.parameters,groups=cifar_groups(engine.model),zero_reference_policy='freeze'),
                 inputs=inputs,target=labels,ids=ids,norm_squared=None,
                 teacher=lambda a,b:engine.teacher_batch(inputs[a:b],cached[a:b],ids[a:b]))
    config=TokenwiseConfig(mean_tolerance=1e-3,tail_tolerance=1e-3,scale=1e-3,rho=.01,ranking_weight=0.,ranking_k=100,ranking_tie_policy='exact_teacher_ties')
    dual=TokenwiseDual([1],config=config,dual_lr=.01)
    row=measure_context(contexts,context,actor,private,config,dual,learning=True)
    assert row['action_family']=='adaptive_ucv_v1'
    assert 'whiten.weight' in snapshot.matrices and 'whiten.weight' not in engine.parameters
    assert contexts.compute['fresh_weight_core_matrix_queries']==len(snapshot.matrices)
    assert contexts.compute['student_forwards']==contexts.compute['student_backwards']==3
    assert contexts.compute['teacher_forwards']==0
    assert all(torch.equal(p,before[n]) and not p.requires_grad and p.grad is None
               for n,p in {**engine._all_parameters,**engine.buffers}.items())
    assert actor.weight_core_input.weight.grad.norm()>0
    assert any(p.grad is not None and p.grad.norm()>0 for p in private.parameters())


class AdaptiveToyContexts(ToyContexts):
    action_family='adaptive_ucv_v1'
    def load(self,*args,**kwargs):
        context=super().load(*args,**kwargs)
        context['dense_weights']={name:torch.full((obs.rows.shape[0],obs.columns.shape[0]),.05) for name,obs in context['snapshot'].matrices.items()}
        context['dense_weights']['first']=context['weight']
        return context
    def measure(self,context,actor,private,config,dual,*,learning,weight=1.):
        if getattr(actor,'fresh_weight_core',False):
            from rewriter.adaptive_sensing import query_weight_cores
            action=actor(context['snapshot'],private,weight_cores=query_weight_cores(context['dense_weights'],private,embedding_names=()))
        else:action=actor(context['snapshot'],private)
        logits=context['inputs']@(context['weight']+action.matrices['first'].dense()).T
        teacher=(context['inputs']@context['weight'].T).log_softmax(-1).detach()
        kl=(teacher.exp()*(teacher-logits.log_softmax(-1))).sum(-1).mean()
        # Explicit dense toy weights, never the cached fixed-anchor sketches.
        norm=.01*action.norm_change(context['dense_weights'],context['snapshot'].vector_weights)
        loss=kl+norm
        if learning:(loss*weight).backward()
        self.compute['toy_forwards']=self.compute.get('toy_forwards',0)+1
        self.compute['toy_backwards']=self.compute.get('toy_backwards',0)+int(learning)
        return dict(domain=context['domain'],local_id=context['local_id'],context_ids=context['ids'],mean_kl=float(kl.detach()),mean_group_l2_percentage_reduction=-float(norm.detach()),strict_topk_changes=int((logits.argmax(-1)!=teacher.argmax(-1)).sum()),objective=float(loss.detach()))


def worker(rank,root,rendezvous):
    torch.set_num_threads(1);dist.init_process_group('gloo',init_method='file://'+str(rendezvous),rank=rank,world_size=2,timeout=datetime.timedelta(seconds=120))
    try:
        cfg=adaptive_recipe();cfg['architecture']['fresh_weight_core']=True
        metadata=dict(recipe=cfg,scope='toy finite logits,not GPU scientific qualification')
        full=run_training(cfg,Path(root)/'full',AdaptiveToyContexts(),torch.device('cpu'),metadata=metadata)
        def interrupt(update):
            if update==2:raise RuntimeError('after committed joint batch')
        with pytest.raises(RuntimeError,match='committed'):
            run_training(cfg,Path(root)/'resumed',AdaptiveToyContexts(),torch.device('cpu'),metadata=metadata,after_commit=interrupt)
        resumed=run_training(cfg,Path(root)/'resumed',AdaptiveToyContexts(),torch.device('cpu'),metadata=metadata,resume=True)
        assert_tree_equal(full,resumed)
        # Check all committed numerical owners, not only the returned actor.
        from rewriter.distributed_probe_store import DistributedProbeStore
        from rewriter.distributed_probe_store import _digest
        identity=hashlib.sha256(json.dumps(metadata,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        from rewriter.private_probe_store import PrivateIdentity
        def restored(path):
            with DistributedProbeStore(path,run_identity=identity) as s:
                state=s.restore();private={}
                for token,v in s._versions.items():
                    ident=PrivateIdentity(tuple(v['identity']['state_key']),v['identity']['state_sha256'],v['identity']['layout_sha256'])
                    private[token]=s.shard.read_private(ident,version=v)
                return state,private
        one,private_one=restored(Path(root)/'full/joint_store')
        two,private_two=restored(Path(root)/'resumed/joint_store')
        assert_tree_equal(private_one,private_two)
        assert_tree_equal(one['shared'],two['shared']);assert_tree_equal(one['rng'],two['rng']);assert_tree_equal(one['sampler'],two['sampler'])
        (Path(root)/f'rank{rank}-pass').write_text('pass')
    finally:dist.destroy_process_group()


def test_two_rank_adaptive_inner_joint_exact_resume(tmp_path):
    mp.spawn(worker,args=(tmp_path,tmp_path/'rdzv'),nprocs=2,join=True)
    assert all((tmp_path/f'rank{i}-pass').exists() for i in range(2))
