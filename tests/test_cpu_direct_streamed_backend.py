import copy
import sys
import pytest
import torch
from scripts import diagnose_cpu_norm_preservation as diagnostic
from rewriter.adaptive_coordinate_chisel import AdaptiveCoordinateChisel
from rewriter.edit_scope import editable_names,editable_weight_norm_sq
from rewriter.tokenwise_reference import TokenwiseConfig,TokenwiseDual
from tests.test_adaptive_preservation_actor import actual_actor_case


@pytest.mark.parametrize('learning',[False,True])
def test_local_measure_direct_streamed_matches_reference_with_live_sensing(learning):
    torch.set_num_threads(1)
    rows=[];counts=[];gradients=[]
    for backend in ('reference','direct_streamed'):
        engine,_,_,snapshot,grouped,inputs=actual_actor_case();inputs=inputs[:,:4]
        torch.manual_seed(659)
        actor=AdaptiveCoordinateChisel(rank=2,trajectory_dim=12,model_dim=128,num_layers=1,latent_tokens=2,
            fresh_weight_core=True,factor_precision='fp64',residual_precision='fp32')
        private=actor.make_private(snapshot,state_key=('test',0,6),seed=77)
        with torch.no_grad():
            for head in actor.output_heads:head.weight.normal_(std=.001)
        cached=[engine.cache_teacher(x[None],data_role='synthetic')[0] for x in inputs]
        editable=editable_names(engine.parameters,'grouped')-grouped.frozen_parameter_names
        context=dict(engine=engine,domain='gpt',local_id=6,snapshot=snapshot,grouped=grouped,inputs=inputs,
            ids=[1,2],teacher=lambda a,b:cached[a],norm_squared=editable_weight_norm_sq(engine.parameters,editable))
        config=TokenwiseConfig(mean_tolerance=1e-10,tail_tolerance=1e-10,scale=1e-5,rho=.01,ranking_weight=.01,
            ranking_k=10,ranking_tie_policy='exact_teacher_ties',ranking_loss_mode='relative_gap')
        dual=TokenwiseDual([6],config=config,dual_lr=.01);dual.multipliers[6]=.1
        contexts=diagnostic.LocalCPUContexts(backend)
        rows.append(contexts.measure(context,actor,private,config,dual,learning=learning,weight=.5));counts.append(contexts.compute)
        gradients.append({prefix+n:None if p.grad is None else p.grad.clone() for prefix,m in [('actor.',actor),('private.',private)] for n,p in m.named_parameters()})
    assert rows[0]==rows[1] and counts[0]==counts[1]
    assert counts[1]['fresh_weight_core_calls']==1 and counts[1]['student_forwards']==2
    assert counts[1]['student_backwards']==(2 if learning else 0)
    for n,a in gradients[0].items():
        b=gradients[1][n];assert (a is None)==(b is None),n
        if a is not None:torch.testing.assert_close(a,b,rtol=5e-4,atol=3e-5,msg=n)
    if learning:assert gradients[1]['actor.vector_output.weight'].norm()>0


def test_backend_resume_identity_legacy_reference_default_and_rejection():
    metadata=dict(args={},objective={},dual_lr=.01,cases=[],state_weights=[.5,.5],private_gradient_normalization='undo')
    explicit=copy.deepcopy(metadata);explicit['args']['panel_backend']='reference'
    assert diagnostic.resume_contract(metadata)==diagnostic.resume_contract(explicit)
    explicit['args']['panel_backend']='direct_streamed'
    assert diagnostic.resume_contract(metadata)!=diagnostic.resume_contract(explicit)


@pytest.mark.parametrize('domain,family',[('cifar10','adaptive_ucv_v1'),('gpt','fixed_anchored_v1')])
def test_direct_mode_rejects_unsupported_context_before_model_work(domain,family):
    contexts=diagnostic.LocalCPUContexts('direct_streamed');contexts.action_family=family
    with pytest.raises(ValueError,match='GPT adaptive'):
        contexts.measure(dict(engine=object(),domain=domain,local_id=6),object(),None,None,None,learning=True)


@pytest.mark.parametrize('updates',[60,200,201])
def test_cli_bounded_longer_horizon_and_early_source_pin(monkeypatch,tmp_path,updates):
    class StopBeforeModel(Exception):pass
    def pin(output):
        assert 'rewriter.experimental_factor_norm' in sys.modules
        assert 'rewriter.experimental_factor_panel' in sys.modules
        raise StopBeforeModel
    monkeypatch.setattr(diagnostic,'pin_sources',pin)
    monkeypatch.setattr(sys,'argv',['diagnose','--cases','missing.json','--output',str(tmp_path/'out'),
        '--shared-lr','1e-9','--private-lr','1e-4','--private-eps','1e-8','--updates',str(updates),
        '--sequences','8','--panel-backend','direct_streamed'])
    with pytest.raises(StopBeforeModel if updates<=200 else ValueError):diagnostic.main()
    if updates>200:assert not (tmp_path/'out').exists()
