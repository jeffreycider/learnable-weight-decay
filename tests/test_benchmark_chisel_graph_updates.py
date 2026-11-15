import copy
import hashlib
import pytest
import torch

from rewriter.primal_dual import PrimalDual
from rewriter.qualification_metrics import optimizer_snapshot
from rewriter.teacher_cache import TeacherPanel
from scripts.benchmark_chisel_graph_updates import (selected_panel, paired_orders,
    production_batch, compare_signatures, validate_prerequisite, checked_local_stage, detached_fitting_proposal,
    load_pinned_policy, check_shrinkage_policy, benchmark_objective, realized_norm_readout, row_errors,
    benchmark_records, benchmark_tolerance)
import scripts.train_multistate_chisel as train


@pytest.mark.parametrize('fixed', [False,True])
def test_exact_production_state_distribution_and_balanced_panels(fixed):
    states=[s for s in range(1,1001) if s%5]
    opts=dict(data_seed=19,split_seed=7,fixed_state_shards=fixed)
    saved=dict(metadata=dict(options=opts,world_size=8),next_epoch=10,next_batch=2)
    batch,epoch=production_batch(saved,states,8)
    partitions=[batch[r::8] for r in range(8)]
    assert all(len(p)==25 for p in partitions)
    assert len(set(sum(partitions,[])))==200
    ids=[s for s in range(1000) if not train.heldout_sequence(s,7)][:128]
    panel=TeacherPanel(torch.arange(128).reshape(128,1),torch.arange(256.).reshape(128,1,2),tuple(ids),{'microbatch_sequences':16})
    a,b=[selected_panel(panel,batch[0],e,opts) for e in (10,11)]
    assert not set(a.sequence_ids)&set(b.sequence_ids)
    assert set(a.sequence_ids)|set(b.sequence_ids)==set(ids)
    assert a.sequence_ids==selected_panel(panel,batch[0],10,opts).sequence_ids
    with pytest.raises(ValueError,match='eight ranks'):production_batch(saved,states,4)
    assert paired_orders(2)==[('reference','graph'),('graph','reference'),('reference','graph')]


def signatures():
    p=torch.nn.Parameter(torch.tensor([.1,.2]))
    optimizer=torch.optim.Adam([p],lr=2e-6)
    initial={'p':p.detach().clone()};p.grad=torch.tensor([.3,.4]);optimizer.step()
    rows=[dict(state_id=n,sequence_ids=(2*n,),mean_kl=1e-6,objective=.03) for n in range(200)]
    dual=PrimalDual(range(200),tolerance=1e-6,rho=.01,dual_lr=.1)
    dual.step([(r['state_id'],r['mean_kl']) for r in rows],expected_ids=range(200))
    return dict(parameters={'p':p.detach().clone()},preclip={'p':p.grad.clone()},clipped={'p':p.grad.clone()},
                adam=optimizer_snapshot(optimizer,{'p':p}),rows=rows,post=copy.deepcopy(rows),dual=dual.state_dict()),initial


def test_global_gate_checks_update_moments_panels_and_dual_counter():
    a,initial=signatures();b=copy.deepcopy(a)
    assert compare_signatures(a,b,initial,.1,1e-6)['within_declared_global_gates']
    b['adam']['states']['p']['exp_avg_sq'].mul_(1.1)
    assert not compare_signatures(a,b,initial,.1,1e-6)['within_declared_global_gates']
    b=copy.deepcopy(a);b['rows'][9]['mean_kl']+=1e-8
    assert not compare_signatures(a,b,initial,.1,1e-6)['within_declared_global_gates']
    b=copy.deepcopy(a);b['post'][7]['sequence_ids']=(999,)
    with pytest.raises(ValueError,match='identities'):compare_signatures(a,b,initial,.1,1e-6)
    b=copy.deepcopy(a);b['dual']['updates']+=1
    with pytest.raises(ValueError,match='counter'):compare_signatures(a,b,initial,.1,1e-6)


def test_prerequisite_rejects_wrong_policy_small_workload_and_bad_layout():
    states=[997,998,999]
    metric=dict(relative_error=4e-6,cosine=1.,absolute_kl_difference=0.)
    summary=dict(policy_sha256='abc',sequences=64,states=states,shrinkage_coefficient=1.,
                 linear_backend='ieee',dx_backend='ieee',activation_backend='native',
                 original_layout_action={f'{n}:True':dict(relative_error=0.) for n in states},
                 qualification={f'{n}:{raw}':metric.copy() for n in states for raw in (True,False)})
    summary['qualification'].update({key:metric.copy() for key in ('accum:True','accum:False','adam_update')})
    validate_prerequisite(summary,'abc',1.,False)
    with pytest.raises(ValueError,match='prefetched rebind'):
        validate_prerequisite(summary,'abc',1.,False,prefetch_checkpoints=True)
    summary['prefetch_checkpoints']=True
    validate_prerequisite(summary,'abc',1.,False,prefetch_checkpoints=True)
    for key,value in [('linear_backend','tf32x3'),('dx_backend','cublas3_large'),('activation_backend','fused')]:
        original=summary[key]
        summary[key]=value
        with pytest.raises(ValueError,match='backend prerequisite'):
            validate_prerequisite(summary,'abc',1.,False)
        del summary[key]
        with pytest.raises(ValueError,match='backend prerequisite'):
            validate_prerequisite(summary,'abc',1.,False)
        summary[key]=original
    summary['kl_backend']='tiled'
    with pytest.raises(ValueError,match='prerequisite'):validate_prerequisite(summary,'abc',1.,False)
    validate_prerequisite(summary,'abc',1.,False,kl_backend='tiled')
    with pytest.raises(ValueError,match='unsupported'):validate_prerequisite(summary,'abc',1.,False,kl_backend='unknown')
    summary['kl_backend']='chunked'
    with pytest.raises(ValueError,match='prerequisite'):validate_prerequisite(summary,'abc',1.,False,kl_backend='tiled')
    with pytest.raises(ValueError,match='prerequisite'):validate_prerequisite(summary,'different',1.,False)
    summary['sequences']=16
    with pytest.raises(ValueError,match='prerequisite'):validate_prerequisite(summary,'abc',1.,False)
    summary['sequences']=64;summary['original_layout_action']['998:True']['relative_error']=float('nan')
    with pytest.raises(ValueError,match='proposal'):validate_prerequisite(summary,'abc',1.,False)
    summary['original_layout_action']['998:True']['relative_error']=0.
    summary['qualification']['adam_update']['relative_error']=.1
    with pytest.raises(ValueError,match='Adam'):validate_prerequisite(summary,'abc',1.,False)
    summary['qualification']['adam_update']['relative_error']=0.
    with pytest.raises(ValueError,match='prerequisite'):validate_prerequisite(summary,'abc',1.,False,True)
    summary['dynamic_zero_groups']=True
    summary['qualification']['early_zero_mask']=dict(mode='dynamic',state_id=1,
        rejected_changed_mask=False,zero_reference_groups=['bias'])
    with pytest.raises(ValueError,match='initial'):validate_prerequisite(summary,'abc',1.,False,True)
    summary['qualification']['early_zero_mask']['state_id']=997
    validate_prerequisite(summary,'abc',1.,False,True)


def test_rank_local_failures_are_exchanged_before_returning_to_collectives():
    exchanged=[]
    def exchange(error):
        exchanged.append(error)
        return [None,error]
    def failing():raise MemoryError('injected preclip clone failure')
    with pytest.raises(RuntimeError,match='preclip clone'):
        checked_local_stage(failing,exchange)
    assert len(exchanged)==1 and 'MemoryError' in exchanged[0]
    with pytest.raises(RuntimeError,match='peer failure'):
        checked_local_stage(lambda:42,lambda error:[error,'peer failure'])
    assert checked_local_stage(lambda:42,lambda error:[error,None])==42


def test_post_update_readout_keeps_grad_enabled_actor_specialization():
    parameter=torch.nn.Parameter(torch.tensor([.2,.3]))
    modes=[]
    def proposal():
        modes.append(torch.is_grad_enabled())
        edited=parameter*2
        return {'p':edited},edited.square().sum(),{},None,None
    with torch.no_grad():
        edited,norm=detached_fitting_proposal(proposal)
        assert not torch.is_grad_enabled()
    assert modes==[True] and not edited['p'].requires_grad
    assert norm==float((parameter*2).square().sum().detach())
    assert parameter.grad is None


def test_checkpoint_and_coefficient_are_explicitly_pinned(tmp_path,monkeypatch):
    path=tmp_path/'frozen.pt'
    saved=dict(metadata=dict(options=dict(shrinkage_coefficient=0.),
                             objective_policy=dict(shrinkage_coefficient=0.)),
               optimizer=dict(state={}),primal_dual=dict(updates=8))
    torch.save(saved,path)
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    loaded,stamp=load_pinned_policy(path,sha)
    assert loaded==saved and len(stamp)==4
    assert check_shrinkage_policy(loaded,0.)==0.
    with pytest.raises(ValueError,match='coefficient'):check_shrinkage_policy(loaded,1.)
    with pytest.raises(ValueError,match='SHA'):load_pinned_policy(path,'wrong')
    original=torch.load
    def changing_load(*args,**kwargs):
        result=original(*args,**kwargs)
        path.write_bytes(path.read_bytes()+b'changed')
        return result
    monkeypatch.setattr(torch,'load',changing_load)
    with pytest.raises(ValueError,match='changed'):load_pinned_policy(path,sha)


def test_benchmark_override_preserves_saved_objective_and_requires_matching_prerequisite():
    saved=dict(metadata=dict(options=dict(shrinkage_coefficient=0.),
                             objective_policy=dict(shrinkage_coefficient=0.)),
               controller={'x':torch.tensor([1.])}, optimizer=dict(state={'step':8}),
               primal_dual=dict(updates=8))
    before=copy.deepcopy(saved)
    default=benchmark_objective(saved,0.)
    assert default['effective_shrinkage_coefficient']==0.
    assert default['explicit_benchmark_override'] is False
    changed=benchmark_objective(saved,0.,1.)
    assert changed['saved_shrinkage_coefficient']==0.
    assert changed['effective_shrinkage_coefficient']==1.
    assert changed['explicit_benchmark_override'] is True
    assert saved==before
    with pytest.raises(ValueError,match='saved shrinkage'):
        benchmark_objective(saved,1.,1.)
    for invalid in (-1.,float('nan'),float('inf')):
        with pytest.raises(ValueError,match='finite and nonnegative'):
            benchmark_objective(saved,0.,invalid)
    # An old coefficient-zero qualification cannot authorize coefficient one.
    prerequisite=dict(policy_sha256='same-policy',sequences=64,states=[997,998,999],
                      shrinkage_coefficient=0.)
    with pytest.raises(ValueError,match='prerequisite'):
        validate_prerequisite(prerequisite,'same-policy',changed['effective_shrinkage_coefficient'],False)
    saved['metadata']['objective_policy']['shrinkage_coefficient']=1.
    with pytest.raises(ValueError,match='contradicts'):
        benchmark_objective(saved,0.,1.)


def test_tighter_benchmark_tau_restores_same_saved_dual_without_mutation():
    original=PrimalDual([1,2],tolerance=1e-6,rho=.01,dual_lr=.1)
    original.multipliers[1]=.3;original.updates=7
    saved=dict(metadata=dict(options=dict(kl_tolerance=1e-6)),primal_dual=original.state_dict())
    before=copy.deepcopy(saved)
    state,policy=benchmark_tolerance(saved,1e-8)
    assert policy['saved_kl_tolerance']==1e-6 and policy['effective_kl_tolerance']==1e-8
    assert policy['explicit_tolerance_override']
    for backend in ('reference','graph','hybrid'):
        dual=PrimalDual([1,2],tolerance=1e-8,rho=.01,dual_lr=.1)
        dual.load_state_dict(state)
        assert dual.updates==7 and dual.multipliers==original.multipliers
        assert dual.terms(5e-8,1)['kl_coefficient']==pytest.approx((.3+.01*4)/1e-8)
        dual.step([(1,5e-8),(2,0.)],expected_ids=[1,2])
        assert dual.updates==8 and dual.multipliers[1]==pytest.approx(.7)
    assert saved==before and state['updates']==7 and state['multipliers'][1]==.3
    assert benchmark_tolerance(saved)[0]==saved['primal_dual']
    for value in (0.,-1.,float('nan'),float('inf')):
        with pytest.raises(ValueError,match='positive finite'):benchmark_tolerance(saved,value)
    saved['primal_dual']['tolerance']=1e-7
    with pytest.raises(ValueError,match='contradicts'):benchmark_tolerance(saved,1e-8)


def test_graph_comparison_abba_and_explicit_hybrid_prerequisite():
    orders=paired_orders(2,True)
    assert orders[0]==('reference','graph','hybrid')
    assert sum((list(row) for row in orders[1:]),[])==['hybrid','graph','graph','hybrid']
    with pytest.raises(ValueError,match='two warm'):paired_orders(1,True)
    states=[1,997,998,999];metric=dict(relative_error=2e-4,cosine=1.,absolute_kl_difference=1e-11)
    summary=dict(policy_sha256='sha',sequences=64,states=states,shrinkage_coefficient=1.,
        fused_softcap=True,prefetch_checkpoints=True,kl_backend='tiled',
        linear_backend='tf32x3',dx_backend='cublas3_cached',activation_backend='fused',
        original_layout_action={f'{n}:True':dict(relative_error=0.) for n in states},
        qualification={f'{n}:{raw}':metric.copy() for n in states for raw in (True,False)})
    summary['qualification'].update({k:metric.copy() for k in ('accum:True','accum:False','adam_update')})
    kwargs=dict(kl_backend='tiled',prefetch_checkpoints=True,
                linear_backend='tf32x3',dx_backend='cublas3_cached',activation_backend='fused')
    validate_prerequisite(summary,'sha',1.,True,**kwargs)
    with pytest.raises(ValueError,match='backend prerequisite'):
        validate_prerequisite(summary,'sha',1.,True,kl_backend='tiled',prefetch_checkpoints=True)
    summary['qualification']['adam_update']['relative_error']=.01
    with pytest.raises(ValueError,match='Adam'):
        validate_prerequisite(summary,'sha',1.,True,**kwargs)


def test_realized_norm_readout_keeps_groups_units_and_detects_missing_or_nonfinite():
    from rewriter.grouped_norm import GROUP_NAMES
    stats=dict(group_l2_percentage_reduction=dict.fromkeys(GROUP_NAMES,.2),
               group_new_l2_norm=dict.fromkeys(GROUP_NAMES,10.),
               mean_group_l2_percentage_reduction=.2,realized_edit_norm=3.)
    parameter=torch.nn.Parameter(torch.tensor([.3]))
    modes=[]
    def proposal():
        modes.append(torch.is_grad_enabled())
        return {'p':parameter*2},parameter.square().sum(),stats,None,None
    with torch.no_grad():
        edited,_,readout=detached_fitting_proposal(proposal,include_norm_readout=True)
    assert modes==[True] and not edited['p'].requires_grad
    assert readout==stats
    a,_=signatures(); b=copy.deepcopy(a)
    a['post'][0]['realized_norms']=readout
    with pytest.raises(ValueError,match='missing in one arm'):row_errors(a['post'],b['post'])
    b['post'][0]['realized_norms']=copy.deepcopy(readout)
    b['post'][0]['realized_norms']['group_l2_percentage_reduction']['head.bias']+=.15
    b['post'][0]['realized_norms']['mean_group_l2_percentage_reduction']+=.01
    row=row_errors(a['post'],b['post'])[0]
    assert row['realized_norms']['reference']==readout
    assert row['realized_norms']['candidate']==b['post'][0]['realized_norms']
    difference=row['realized_norm_differences']
    assert difference['mean_reduction_percentage_points']==pytest.approx(.01)
    assert difference['group_reduction_percentage_points']['head.bias']==pytest.approx(.15)
    for key,value in [('realized_edit_norm',float('nan')),('mean_group_l2_percentage_reduction',.4)]:
        with pytest.raises(ValueError):realized_norm_readout(dict(stats,**{key:value}))
    bad=copy.deepcopy(stats);del bad['group_new_l2_norm']['head.bias']
    with pytest.raises(ValueError,match='fifteen'):realized_norm_readout(bad)


@pytest.mark.parametrize('failure',['schedule','take','consumer'])
def test_benchmark_staging_closes_before_propagating_failure(failure):
    events=[]
    class Stage:
        def __init__(self):events.append('create')
        def schedule(self,states):
            events.append(('schedule',states))
            if failure=='schedule':raise RuntimeError(failure)
        def take(self,state):
            events.append(('take',state))
            if failure=='take':raise RuntimeError(failure)
            return 'record','panel',{}
        def close(self):events.append('close')
    with pytest.raises(RuntimeError,match=failure):
        with benchmark_records([3,1],None,stager_factory=Stage) as rows:
            next(rows)
            raise RuntimeError('consumer')
    assert events[0]=='create' and events[-1]=='close'


def test_benchmark_staging_preserves_actual_verified_records_and_panels(tmp_path):
    from tests.chisel_fixtures import tiny_bank_and_cache
    from rewriter.state_bank import StateBankReader
    from rewriter.teacher_cache import TeacherCacheReader
    from rewriter.graph_stager import GraphCheckpointStager
    from rewriter.static_snapshot_store import _leaves
    bank,targets=tiny_bank_and_cache(tmp_path)
    reader,cache=StateBankReader(bank),TeacherCacheReader(targets)
    worker_reader,worker_cache=StateBankReader(bank),TeacherCacheReader(targets)
    made=[]
    def factory():
        stage=GraphCheckpointStager(worker_reader,worker_cache,pin=False)
        made.append(stage);return stage
    def read(state):return reader.read(state),cache.read(state,'train')
    states=[4,1,3];diagnostics=[]
    with benchmark_records(states,read) as rows:reference=list(rows)
    with benchmark_records(states,read,stager_factory=factory,staging_rows=diagnostics) as rows:
        for (sid,a,ap),(tid,b,bp) in zip(reference,rows,strict=True):
            assert sid==tid and a.assert_verified()==b.assert_verified()
            assert ap.sequence_ids==bp.sequence_ids and ap.metadata==bp.metadata
            assert torch.equal(ap.inputs,bp.inputs) and torch.equal(ap.features,bp.features)
            assert a.model.keys()==b.model.keys()
            assert all(torch.equal(a.model[n],b.model[n]) for n in a.model)
            for (an,av),(bn,bv) in zip(_leaves(a.snapshot,a.train),_leaves(b.snapshot,b.train),strict=True):
                assert an==bn and av.stride()==bv.stride() and torch.equal(av,bv)
    assert [r['state_id'] for r in diagnostics]==states
    assert not made[0]._thread.is_alive()


def test_explicit_fresh_initialization_preserves_saved_checkpoint():
    from scripts.benchmark_chisel_graph_updates import benchmark_initialization
    saved={'optimizer':{'state':{0:{'step':torch.tensor(8.),'exp_avg':torch.tensor([.3])}},'param_groups':[{'lr':.1}]}}
    fresh={'state':{},'param_groups':[{'lr':.2}]}
    dual={'multipliers':{1:.3,2:.4},'updates':8,'tolerance':1e-8}
    old_opt,old_dual,old_meta=benchmark_initialization(saved,fresh,dual)
    opt,new_dual,meta=benchmark_initialization(saved,fresh,dual,True)
    assert old_meta['initial_optimizer_steps']==[8.]
    assert old_dual==dual and old_opt['state']
    assert not opt['state'] and opt['param_groups']==fresh['param_groups']
    assert new_dual['multipliers']=={1:0.,2:0.} and new_dual['updates']==0
    assert saved['optimizer']['state'][0]['step']==8 and dual['updates']==8
    assert meta['initial_optimizer_sha256']!=old_meta['initial_optimizer_sha256']
    assert meta['initial_dual_sha256']!=old_meta['initial_dual_sha256']
