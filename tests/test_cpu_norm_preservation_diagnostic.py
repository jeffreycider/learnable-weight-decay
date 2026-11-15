import torch
import pytest
from baseline.model import GPT
from rewriter.function_preservation import FunctionalPreservation
from scripts.diagnose_cpu_norm_preservation import DiskTeacherCache,load_bound_state,LocalCPUContexts,objective_config
from rewriter.tokenwise_reference import TokenwiseDual
from rewriter.edit_scope import editable_names,editable_weight_norm_sq


def test_cpu_teacher_disk_cache_rejects_wrong_state_input_and_reload(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(71)
    model=GPT(16,1,128).float().requires_grad_(False)
    engine=FunctionalPreservation(model,microbatch_sequences=1,kl_dtype=torch.float64,gpt_precision='fp32')
    state=model.state_dict();load_bound_state(engine,state,'state-a');inputs=torch.randint(16,(2,4));ids=[10,11]
    DiskTeacherCache.create(engine,inputs,ids,'state-a',tmp_path/'cache')
    provider=DiskTeacherCache(tmp_path/'cache',engine,inputs,ids,'state-a')
    assert provider(0,1).log_probs.dtype==torch.float64
    assert engine.compute.teacher_forwards==2
    with pytest.raises(ValueError,match='identity'):DiskTeacherCache(tmp_path/'cache',engine,inputs.flip(0),ids,'state-a')
    load_bound_state(engine,state,'state-b')
    with pytest.raises(ValueError,match='stale'):provider(0,1)
    with pytest.raises(ValueError,match='identity'):DiskTeacherCache(tmp_path/'cache',engine,inputs,ids,'state-b')
    load_bound_state(engine,state,'state-a')
    reused=DiskTeacherCache(tmp_path/'cache',engine,inputs,ids,'state-a')
    assert engine.compute.teacher_forwards==0 and reused.meta['teacher_compute']['teacher_forwards']==2
    assert torch.equal(reused(0,1).inputs,inputs[:1])
    with (tmp_path/'cache'/'0000.pt').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='file changed'):DiskTeacherCache(tmp_path/'cache',engine,inputs,ids,'state-a')


@pytest.mark.parametrize('rank_mode',[False,True])
@pytest.mark.parametrize('panel_backend',['reference','direct_streamed'])
def test_actual_main_loss_one_step_tiny_gpt_gradients_and_counts(tmp_path,rank_mode,panel_backend):
    from tests.test_adaptive_preservation_actor import actual_actor_case
    engine,actor,private,snapshot,grouped,inputs=actual_actor_case()
    if panel_backend=='direct_streamed':
        from tests.test_adaptive_fp64 import actor_for
        actor,private=actor_for(snapshot)
        with torch.no_grad():
            for head in actor.output_heads:head.weight.normal_(std=.001)
    # Fixture modified teacher parameters intentionally; bind its actual current state.
    state={**engine.parameters,**engine.buffers};load_bound_state(engine,state,'tiny')
    from rewriter.grouped_norm import GroupedNormObjective
    grouped=GroupedNormObjective(engine.parameters,zero_reference_policy='freeze')
    inputs=inputs[:1];ids=[4]
    DiskTeacherCache.create(engine,inputs,ids,'tiny',tmp_path/'cache')
    provider=DiskTeacherCache(tmp_path/'cache',engine,inputs,ids,'tiny')
    editable=editable_names(engine.parameters,'grouped')-grouped.frozen_parameter_names
    context=dict(engine=engine,domain='gpt',local_id=6,snapshot=snapshot,grouped=grouped,inputs=inputs,ids=ids,teacher=provider,norm_squared=editable_weight_norm_sq(engine.parameters,editable))
    from rewriter.ranking_dual import RankingDual
    config=objective_config();dual=(RankingDual([6],config=config,dual_lr=.01,rank_scale=1e-5,rank_dual_lr=1e-4) if rank_mode else TokenwiseDual([6],config=config,dual_lr=.01))
    if rank_mode:
        dual.step([(6,0.)],expected_ids=[6],ranking_measurements=[(6,2e-5)])
        config=dual.config_for_state(6)
    contexts=LocalCPUContexts(panel_backend);before={n:p.clone() for n,p in actor.named_parameters()}
    opt=torch.optim.Adam(actor.parameters(),lr=1e-8)
    row=contexts.measure(context,actor,private,config,dual,learning=True,weight=.5)
    assert row['edit_size_penalty_enabled'] is False
    assert row['objective']==pytest.approx(row['norm_objective']+row['dual_linear_term']+row['tail_quadratic_penalty']+row['ranking_penalty'])
    assert row['norm_objective']==row['unweighted_norm_term']
    assert engine.compute.student_forwards==1 and engine.compute.student_backwards==1
    assert all(torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None)
    assert any(p.grad is not None and p.grad.abs().max()>0 for p in private.parameters())
    opt.step();assert any(not torch.equal(p,before[n]) for n,p in actor.named_parameters())


def test_config_matches_declared_main_objective_and_dual_mode():
    config=objective_config()
    assert config.dual_update_mode=='clipped_relative' and config.ranking_weight==100
    assert config.mean_tolerance==config.tail_tolerance==1e-10 and config.scale==1e-8
    assert config.rho==.01 and config.ranking_k==100
    assert config.ranking_tie_policy=='exact_teacher_ties' and config.ranking_loss_mode=='relative_gap'


@pytest.mark.parametrize('update_accumulation', ['none', 'fp32_remainder'])
@pytest.mark.parametrize('rank_mode',[False,True])
def test_exact_resume_two_private_steps_duals_rng_and_rejections(tmp_path, update_accumulation,rank_mode):
    import copy
    from scripts.diagnose_cpu_norm_preservation import training_state,restore_training_state,make_diagnostic_adam,make_diagnostic_dual
    torch.set_num_threads(1)
    dtype=torch.float32 if update_accumulation=='fp32_remainder' else torch.float64
    def fixture():
        torch.manual_seed(92)
        actor=torch.nn.Linear(3,2).to(dtype=dtype)
        cases=[dict(uid=i,private=torch.nn.Linear(2,1).to(dtype=dtype)) for i in (6,7)]
        opt=make_diagnostic_adam(actor.parameters(),update_accumulation=update_accumulation,lr=1e-7,betas=(.9,.99),eps=1e-8)
        private={c['uid']:make_diagnostic_adam(c['private'].parameters(),update_accumulation=update_accumulation,lr=1e-6,betas=(.9,.99),eps=1e-12) for c in cases}
        dual=make_diagnostic_dual([6,7],objective_config(),**(dict(rank_scale=1e-5,rank_dual_lr=1e-4) if rank_mode else {}))
        return actor,cases,opt,private,dual
    metadata=dict(args=dict(sequences=2,policy='full',shared_lr=1e-7,body_lr=1e-7,private_lr=1e-6,clip=1.,seed=92,gradient_profile=False),
        objective={'test':True},dual_lr=.01,state_weights=[.5,.5],private_gradient_normalization='undo',
        cases=[dict(state_id=i,sha256={'state':str(i),'panel':'panel'},sequence_ids=[1,2],input_sha256='input',private_identity={'uid':i}) for i in (6,7)])
    if update_accumulation!='none':metadata['args']['update_accumulation']=update_accumulation
    if rank_mode:metadata['ranking_constraint']=dict(scale=1e-5,dual_lr=1e-4,tolerance=0.,measurement='mean_nonnegative_relative_gap_margin_loss')
    def step(parts):
        actor,cases,opt,private,dual=parts;opt.zero_grad(set_to_none=True)
        for c in cases:
            po=private[c['uid']];po.zero_grad(set_to_none=True)
            loss=c['private'](actor(torch.randn(4,3,dtype=dtype))).square().mean()*.5
            loss.backward()
            for p in c['private'].parameters():p.grad.mul_(2)
            torch.nn.utils.clip_grad_norm_(c['private'].parameters(),1.);po.step()
        torch.nn.utils.clip_grad_norm_(actor.parameters(),1.);opt.step()
        dual.step([(6,2e-10),(7,3e-10)],expected_ids=[6,7],**(dict(ranking_measurements=[(6,2e-5),(7,1e-5)]) if rank_mode else {}))
    full=fixture();step(full)
    saved=copy.deepcopy(training_state(*full,metadata,1))
    torch.save(saved,tmp_path/'checkpoint.pt')
    step(full);expected=copy.deepcopy(training_state(*full,metadata,2))
    resumed=fixture();torch.rand(19)
    loaded=torch.load(tmp_path/'checkpoint.pt',weights_only=False)
    assert restore_training_state(loaded,*resumed,metadata,2)==1
    step(resumed);actual=training_state(*resumed,metadata,2)
    def equal(a,b):
        if torch.is_tensor(a):assert torch.equal(a,b)
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:equal(a[k],b[k])
        elif isinstance(a,(tuple,list)):
            assert len(a)==len(b)
            for x,y in zip(a,b):equal(x,y)
        else:assert a==b
    equal(expected,actual)
    for key in ('policy','private_lr','clip'):
        wrong=copy.deepcopy(metadata);wrong['args'][key]='changed'
        with pytest.raises(ValueError,match='identity'):restore_training_state(loaded,*fixture(),wrong,2)
    wrong=copy.deepcopy(metadata);wrong['args']['update_accumulation']='none' if update_accumulation!='none' else 'fp32_remainder'
    with pytest.raises(ValueError,match='identity'):restore_training_state(loaded,*fixture(),wrong,2)
    wrong=copy.deepcopy(metadata);wrong['cases'][0]['sha256']['state']='wrong'
    with pytest.raises(ValueError,match='identity'):restore_training_state(loaded,*fixture(),wrong,2)
    wrong=copy.deepcopy(loaded);wrong['shared_optimizer']['param_groups'][0]['eps']=1e-12
    with pytest.raises(ValueError,match='optimizer'):restore_training_state(wrong,*fixture(),metadata,2)
    with pytest.raises(ValueError,match='clock'):restore_training_state(loaded,*fixture(),metadata,0)
    # Old metadata omitted epsilon; explicit legacy default is exact-resume compatible.
    explicit=copy.deepcopy(metadata);explicit['args']['private_eps']=1e-12
    assert restore_training_state(loaded,*fixture(),explicit,2)==1
    changed=copy.deepcopy(metadata);changed['args']['private_eps']=1e-8
    with pytest.raises(ValueError,match='identity'):restore_training_state(loaded,*fixture(),changed,2)
    wrong=copy.deepcopy(loaded);wrong['private_optimizers'][6]['param_groups'][0]['eps']=1e-8
    with pytest.raises(ValueError,match='optimizer'):restore_training_state(wrong,*fixture(),metadata,2)



@pytest.mark.parametrize('value',[0.,-1e-12,float('nan'),float('inf'),-float('inf')])
def test_invalid_private_epsilon_rejected(value):
    from scripts.diagnose_cpu_norm_preservation import validate_private_epsilon
    with pytest.raises(ValueError,match='epsilon'):validate_private_epsilon(value)


def test_seed_provenance_and_actual_optimizer_configuration():
    from scripts.diagnose_cpu_norm_preservation import seed_provenance,optimizer_configuration,validate_private_epsilon
    original=dict(diagnostic_control='random_shared_zero_heads_fresh_private',initialization={'imitation':False,'shared_seed':940067})
    retained=seed_provenance(original)
    assert retained['initialization']==original['initialization'] and retained['diagnostic_control']==original['diagnostic_control']
    original['initialization']['imitation']=True
    assert retained['initialization']['imitation'] is False
    assert seed_provenance({})['initialization'] is None
    parameter=torch.nn.Parameter(torch.ones(1));opt=torch.optim.Adam([parameter],lr=1e-6,eps=validate_private_epsilon(1e-8))
    actual=optimizer_configuration(opt)
    assert actual[0]['eps']==1e-8 and actual[0]['lr']==1e-6 and 'params' not in actual[0]


def test_role_clipping_does_not_suppress_small_body_gradients():
    from scripts.diagnose_cpu_norm_preservation import clip_shared_gradients
    def fixture():
        head=torch.nn.Parameter(torch.ones(2,dtype=torch.float64))
        body=torch.nn.Parameter(torch.ones(2,dtype=torch.float64))
        head.grad=torch.tensor([6000.,8000.],dtype=torch.float64)
        body.grad=torch.tensor([.0003,.0004],dtype=torch.float64)
        return head,body
    h,b=fixture();joint=clip_shared_gradients([h],[b],1.,'joint')
    joint_body=b.grad.norm().item()
    h,b=fixture();separate=clip_shared_gradients([h],[b],1.,'by_role')
    assert joint==pytest.approx(separate)
    assert h.grad.norm().item()==pytest.approx(1.)
    assert b.grad.norm().item()==pytest.approx(.0005)
    assert b.grad.norm().item()/joint_body==pytest.approx(10000.)
    # Profile-only behavior must leave both roles unchanged.
    h,b=fixture();old=[h.grad.clone(),b.grad.clone()]
    clip_shared_gradients([h],[b],float('inf'),'by_role')
    assert torch.equal(h.grad,old[0]) and torch.equal(b.grad,old[1])
    with pytest.raises(ValueError,match='policy'):clip_shared_gradients([h],[b],1.,'unknown')


def test_clipping_policy_is_part_of_resume_identity():
    from scripts.diagnose_cpu_norm_preservation import resume_contract
    import copy
    metadata=dict(args={},objective={},dual_lr=.01,cases=[],state_weights=[.5,.5],private_gradient_normalization='undo')
    explicit=copy.deepcopy(metadata);explicit['args']['shared_clip_policy']='joint'
    assert resume_contract(metadata)==resume_contract(explicit)
    explicit['args']['shared_clip_policy']='by_role'
    assert resume_contract(metadata)!=resume_contract(explicit)


def test_accumulation_is_opt_in_and_legacy_resume_means_ordinary_adam():
    import copy
    from scripts.diagnose_cpu_norm_preservation import make_diagnostic_adam,resume_contract
    from rewriter.update_accumulation import CompensatedAdam
    parameter=torch.nn.Parameter(torch.ones(2))
    assert type(make_diagnostic_adam([parameter],lr=1e-10)) is torch.optim.Adam
    assert type(make_diagnostic_adam([parameter],lr=1e-10,update_accumulation='fp32_remainder')) is CompensatedAdam
    with pytest.raises(ValueError,match='mode'):make_diagnostic_adam([parameter],update_accumulation='unknown')
    metadata=dict(args={},objective={},dual_lr=.01,cases=[],state_weights=[.5,.5],private_gradient_normalization='undo')
    explicit=copy.deepcopy(metadata);explicit['args']['update_accumulation']='none'
    assert resume_contract(metadata)==resume_contract(explicit)
    explicit['args']['update_accumulation']='fp32_remainder'
    assert resume_contract(metadata)!=resume_contract(explicit)
