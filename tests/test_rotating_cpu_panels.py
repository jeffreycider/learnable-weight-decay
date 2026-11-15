import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from scripts import diagnose_cpu_norm_preservation as d


def test_explicit_pool_plan_cycles_without_reserved_indices_and_resumes_absolutely():
    path=Path('runs/main-objective-local/pool-plans/pool120-v1.json')
    plan,schedule=d.read_pool_indices(path,8)
    provenance=json.loads(path.with_name('pool120-v1-provenance.json').read_text())
    assert schedule['plan_sha256']==provenance['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    for indices in plan.values():
        assert len(indices)==120 and set(indices).isdisjoint(provenance['reserved_panel_indices'])
        cycle=[indices[p] for step in range(15) for p in d.panel_positions(120,8,step)]
        assert cycle==indices
        assert d.panel_positions(120,8,15)==d.panel_positions(120,8,0)
    uninterrupted=[d.panel_positions(120,8,i) for i in range(40)]
    for resume in (0,7,15,29):
        assert uninterrupted==uninterrupted[:resume]+[d.panel_positions(120,8,i) for i in range(resume,40)]
    assert d.panel_positions(120,8,60,final=True)==tuple(range(120))


@pytest.mark.parametrize('plan',[{'6':[0,1],'7':[0,1,2,3]}, {'6':[0,0],'7':[0,1]}, {'6':[0,1,2],'7':[0,1,2]}, {'6':[-1,0],'7':[0,1]}, {'6':[False,1],'7':[0,1]}])
def test_invalid_pool_plans_rejected(tmp_path,plan):
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):d.read_pool_indices(path,2)


def tiny_cache(tmp_path):
    inputs=torch.arange(16).reshape(4,4);ids=[101,102,103,104]
    engine=SimpleNamespace(teacher_id='load',_local_cpu_binding=('state','load'),microbatch_sequences=1,
        kl_dtype=torch.float64,gpt_precision='fp32',device=torch.device('cpu'))
    files=[]
    for i in range(4):
        path=tmp_path/f'{i}.pt';torch.save(dict(inputs=inputs[i:i+1],sequence_ids=ids[i:i+1],log_probs=torch.full((1,4,20),float(i),dtype=torch.float64)),path)
        files.append(dict(file=path.name,sha256=d.sha(path)))
    meta=dict(state_sha256='state',input_sha256=d.tensor_sha(inputs),sequence_ids=ids,files=files,
        precision='CPU FP32 GPT with FP64 log_softmax',torch_version=torch.__version__)
    (tmp_path/'manifest.json').write_text(json.dumps(meta))
    return d.DiskTeacherCache(tmp_path,engine,inputs,ids,'state'),engine


def test_subset_teacher_alignment_order_validation_and_staleness(tmp_path):
    cache,engine=tiny_cache(tmp_path);subset=d.SubsetTeacherProvider(cache,[3,1])
    assert subset.ids==[104,102] and torch.equal(subset.inputs,cache.inputs[[3,1]])
    for i,position in enumerate([3,1]):
        batch=subset(i,i+1)
        assert batch.sequence_ids==(cache.ids[position],)
        assert torch.equal(batch.inputs,subset.inputs[i:i+1])
        assert batch.log_probs.unique().item()==position
    for bad in ([1,1],[4],[-1],[]):
        with pytest.raises(ValueError):d.SubsetTeacherProvider(cache,bad)
    engine.teacher_id='reload'
    with pytest.raises(ValueError,match='stale'):subset(0,1)


def test_resume_contract_legacy_fixed_and_ordered_pool_changes():
    meta=dict(args={},objective={},dual_lr=.01,cases=[],state_weights=[.5,.5],private_gradient_normalization='undo')
    explicit=copy.deepcopy(meta);explicit['pool_schedule']=None
    assert d.resume_contract(meta)==d.resume_contract(explicit)
    _,schedule=d.read_pool_indices('runs/main-objective-local/pool-plans/pool120-v1.json',8)
    pooled=copy.deepcopy(meta);pooled['pool_schedule']=schedule
    assert d.resume_contract(meta)!=d.resume_contract(pooled)
    reordered=copy.deepcopy(pooled);reordered['pool_schedule']['indices_by_state']['6'].reverse()
    with pytest.raises(ValueError,match='identity'):
        d.restore_training_state({'metadata':pooled},None,[],None,{},None,reordered,60)


def test_single_panel_subset_actual_tiny_gpt_loss_gradients_and_final_readout(tmp_path):
    from tests.test_adaptive_preservation_actor import actual_actor_case
    from rewriter.grouped_norm import GroupedNormObjective
    from rewriter.tokenwise_reference import TokenwiseConfig,TokenwiseDual
    from rewriter.edit_scope import editable_names,editable_weight_norm_sq
    torch.set_num_threads(1);rows=[];gradients=[]
    for subset_mode in (False,True):
        engine,actor,private,snapshot,_,inputs=actual_actor_case();inputs=inputs[:,:4]
        d.load_bound_state(engine,{**engine.parameters,**engine.buffers},'tiny')
        grouped=GroupedNormObjective(engine.parameters,zero_reference_policy='freeze')
        folder=tmp_path/str(subset_mode);ids=[8,9]
        d.DiskTeacherCache.create(engine,inputs,ids,'tiny',folder)
        provider=d.DiskTeacherCache(folder,engine,inputs,ids,'tiny')
        if subset_mode:provider=d.SubsetTeacherProvider(provider,d.panel_positions(2,2,19))
        editable=editable_names(engine.parameters,'grouped')-grouped.frozen_parameter_names
        context=dict(engine=engine,domain='gpt',local_id=6,snapshot=snapshot,grouped=grouped,inputs=provider.inputs,ids=provider.ids,teacher=provider,norm_squared=editable_weight_norm_sq(engine.parameters,editable))
        config=TokenwiseConfig(mean_tolerance=1e-10,tail_tolerance=1e-10,scale=1e-5,rho=.01,ranking_weight=.01,ranking_k=10,ranking_tie_policy='exact_teacher_ties',ranking_loss_mode='relative_gap')
        dual=TokenwiseDual([6],config=config,dual_lr=.01);contexts=d.LocalCPUContexts()
        training=contexts.measure(context,actor,private,config,dual,learning=True,weight=.5)
        gradients.append([None if p.grad is None else p.grad.clone() for m in (actor,private) for p in m.parameters()])
        before=engine.compute.as_dict()
        final=contexts.measure(context,actor,private,config,dual,learning=False,weight=.5)
        assert final==training and final['sequence_count']==2 and final['token_count']==8
        assert engine.compute.student_forwards-before['student_forwards']==2
        assert engine.compute.student_backwards==before['student_backwards']
        rows.append((training,final))
    assert rows[0]==rows[1]
    for a,b in zip(*gradients):
        assert (a is None)==(b is None)
        if a is not None:assert torch.equal(a,b)


def test_multiple_panels_full_readout_is_weighted_mean_with_additive_counts(tmp_path):
    from tests.test_adaptive_preservation_actor import actual_actor_case
    from rewriter.function_preservation import PreservationCompute
    from rewriter.grouped_norm import GroupedNormObjective
    from rewriter.tokenwise_reference import TokenwiseConfig,TokenwiseDual
    from rewriter.edit_scope import editable_names,editable_weight_norm_sq
    torch.set_num_threads(1)
    engine,actor,private,snapshot,_,inputs=actual_actor_case()
    inputs=torch.cat((inputs[:,:4],inputs[:,:4].roll(1,dims=1)));ids=[21,22,23,24]
    d.load_bound_state(engine,{**engine.parameters,**engine.buffers},'tiny-multi')
    grouped=GroupedNormObjective(engine.parameters,zero_reference_policy='freeze')
    d.DiskTeacherCache.create(engine,inputs,ids,'tiny-multi',tmp_path/'pool')
    cache=d.DiskTeacherCache(tmp_path/'pool',engine,inputs,ids,'tiny-multi')
    setup=engine.compute.as_dict();assert setup['teacher_forwards']==4
    editable=editable_names(engine.parameters,'grouped')-grouped.frozen_parameter_names
    config=TokenwiseConfig(mean_tolerance=1e-10,tail_tolerance=1e-10,scale=1e-5,rho=.01,ranking_weight=.01,ranking_k=10,ranking_tie_policy='exact_teacher_ties',ranking_loss_mode='relative_gap')
    dual=TokenwiseDual([6],config=config,dual_lr=.01);dual.multipliers[6]=.1
    contexts=d.LocalCPUContexts();rows=[]
    for positions in (d.panel_positions(4,2,0),d.panel_positions(4,2,1),d.panel_positions(4,2,2,final=True)):
        provider=d.SubsetTeacherProvider(cache,positions)
        # Main's state reload resets these counters; keep the same verified
        # teacher binding here while isolating each measure's compute delta.
        engine.compute=PreservationCompute()
        context=dict(engine=engine,domain='gpt',local_id=6,snapshot=snapshot,grouped=grouped,inputs=provider.inputs,ids=provider.ids,teacher=provider,norm_squared=editable_weight_norm_sq(engine.parameters,editable))
        rows.append(contexts.measure(context,actor,private,config,dual,learning=False,weight=.5))
    left,right,full=rows
    for key in ('mean_kl','kl_penalty','objective','dual_linear_term','augmented_penalty','tail_quadratic_penalty','ranking_penalty','norm_objective','mean_nonnegative_token_kl'):
        weighted=(left[key]*left['sequence_count']+right[key]*right['sequence_count'])/full['sequence_count']
        assert full[key]==pytest.approx(weighted,rel=1e-10,abs=1e-12),key
    assert full['token_kl_quantiles']['max']==max(left['token_kl_quantiles']['max'],right['token_kl_quantiles']['max'])
    for key in ('strict_topk_changes','strict_order_changes','strict_membership_changes','tie_aware_argmax_changes','token_count','sequence_count'):
        assert full[key]==left[key]+right[key],key
    assert full['context_ids']==ids and full['sequence_count']==4 and full['token_count']==16
    assert contexts.compute['student_forwards']==8 and contexts.compute['student_backwards']==0
    assert setup['teacher_forwards']+contexts.compute['student_forwards']==12
    assert all(p.grad is None for m in (actor,private) for p in m.parameters())
