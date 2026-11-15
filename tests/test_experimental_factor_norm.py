import pytest
import torch
from rewriter.adaptive_action import AdaptiveAction,AdaptiveFactors
from rewriter.adaptive_preservation import group_rms_scales,build_adaptive_proposal
from rewriter.experimental_factor_norm import streamed_grouped_norm
from rewriter.grouped_norm import GroupedNormObjective


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16,torch.float64])
@pytest.mark.parametrize('chunk',[1,4,128])
@pytest.mark.parametrize('saturated',[False,True])
def test_streamed_norm_matches_rounded_dense_shared_factors_vectors_guard(dtype,chunk,saturated):
    torch.set_num_threads(1);torch.manual_seed(639)
    base={'proj.weight':torch.randn(5,7).to(dtype),'embed.weight':torch.randn(11,7).to(dtype),
          'proj.bias':torch.randn(5).to(dtype),'blocks.0.attn.q.bias':torch.zeros(3,dtype=dtype)}
    groups={'head':('proj.weight',),'embedding':('embed.weight',),'bias':('proj.bias',),'zero':('blocks.0.attn.q.bias',)}
    grouped=GroupedNormObjective(base,groups,zero_reference_policy='freeze',chunk_rows=3)
    tensors=[torch.randn(*shape,dtype=torch.float64,requires_grad=True) for shape in [(5,2),(2,3),(7,3),(3,2),(11,2),(5,)]]
    def run(thin):
        u,c,shared,ce,ve,bias=tensors
        action=AdaptiveAction({'proj.weight':AdaptiveFactors(u,c,shared),'embed.weight':AdaptiveFactors(shared,ce,ve)},
            {'proj.bias':bias,'blocks.0.attn.q.bias':torch.zeros(3,dtype=torch.float64)})
        rms=group_rms_scales(grouped,dtype=torch.float64)
        action=AdaptiveAction({n:f.scaled(rms[n]) for n,f in action.matrices.items()},{n:v*rms[n] for n,v in action.vectors.items()})
        raw=action.norm_sq().clamp_min(1e-24).sqrt();scale=torch.minimum(raw.new_ones(()),raw.new_tensor(.001 if saturated else 1e6)/(raw+1e-12))
        params={}
        for n,w in base.items():
            delta=action.matrices[n].dense() if w.ndim==2 else action.vectors[n]
            if n=='embed.weight':delta=delta.T
            params[n]=(w.double()+scale*delta).to(dtype)
        if thin:
            values,loss=streamed_grouped_norm(params,grouped,action,scale,transpose_storage={'embed.weight':True},chunk_rows=chunk,shrinkage_coefficient=.37)
            assert all(not p.requires_grad and p.grad_fn is None for p in values.values())
        else:loss=.37*(100*grouped.evaluate(params).mean_relative_change)
        grads=torch.autograd.grad(loss,tensors,loss.new_tensor(.5))
        return loss.detach(),grads
    expected,eg=run(False);actual,ag=run(True)
    assert torch.equal(actual,expected)
    for a,e in zip(ag,eg):torch.testing.assert_close(a,e,rtol=2e-11,atol=2e-12)


def test_vector_cotangent_rounds_before_fp64_scale_multiplication():
    base={'proj.bias':torch.tensor([.11331,.77311,-.92341],dtype=torch.float32)}
    grouped=GroupedNormObjective(base,{'bias':('proj.bias',)})
    edit=torch.tensor([.01623,-.00231,.07129],dtype=torch.float64,requires_grad=True)
    scale=torch.tensor(.17321,dtype=torch.float64,requires_grad=True)
    params={'proj.bias':(base['proj.bias'].double()+scale*edit).float()}
    action=AdaptiveAction({}, {'proj.bias':edit})
    _,loss=streamed_grouped_norm(params,grouped,action,scale,chunk_rows=1)
    grad=torch.autograd.grad(loss,edit)[0]
    result=grouped.evaluate(params);old=grouped.reference_norms['bias'];new=result.new_norms['bias']
    norm_credit=100/(2*old*new)
    expected=(2*norm_credit*params['proj.bias'].double()).float().double()*scale
    wrongly_commuted=(2*norm_credit*params['proj.bias'].double())*scale
    assert torch.equal(grad,expected)
    assert not torch.equal(grad,wrongly_commuted)


@pytest.mark.parametrize('radius',[1e-6,1.])
def test_actual_fp64_proposal_actor_private_norm_credit(radius):
    from tests.test_adaptive_preservation_actor import actual_actor_case
    from rewriter.adaptive_coordinate_chisel import AdaptiveCoordinateChisel
    results=[];gradients=[]
    for thin in (False,True):
        engine,_,_,snapshot,grouped,_=actual_actor_case()
        torch.manual_seed(641)
        actor=AdaptiveCoordinateChisel(rank=2,trajectory_dim=12,model_dim=128,num_layers=1,latent_tokens=2,factor_precision='fp64',residual_precision='fp32')
        private=actor.make_private(snapshot,state_key=('test',0,6),seed=77)
        with torch.no_grad():
            for head in actor.output_heads:head.weight.normal_(std=.001)
        captured=[]
        params,regularizer,stats,_,credit=build_adaptive_proposal(lambda snap:actor(snap,private),snapshot,engine,grouped,radius_fraction=radius,execution_capture=lambda a,s:captured.append((a,s)))
        if thin:
            params,regularizer=streamed_grouped_norm(params,grouped,*captured[0],transpose_storage=engine.transpose_storage,chunk_rows=3)
        regularizer.backward(regularizer.new_tensor(.5))
        original,leaves=credit;pairs=[(o,l.grad) for o,l in zip(original,leaves) if o.requires_grad and l.grad is not None]
        torch.autograd.backward([o for o,g in pairs],[g for o,g in pairs])
        results.append(float(regularizer.detach()))
        gradients.append([None if p.grad is None else p.grad.clone() for m in (actor,private) for p in m.parameters()])
    assert results[0]==results[1]
    for a,e in zip(gradients[1],gradients[0]):
        assert (a is None)==(e is None)
        if a is not None:torch.testing.assert_close(a,e,rtol=3e-5,atol=2e-7)


def test_adapter_releases_dense_fp64_edit_graph_and_saves_only_thin_tensors():
    import gc,weakref
    torch.manual_seed(643)
    base={'proj.weight':torch.randn(9,13)}
    grouped=GroupedNormObjective(base,{'head':('proj.weight',)})
    u=torch.randn(9,2,dtype=torch.float64,requires_grad=True)
    c=torch.randn(2,3,dtype=torch.float64,requires_grad=True)
    v=torch.randn(13,3,dtype=torch.float64,requires_grad=True)
    scale=torch.tensor(.01,dtype=torch.float64,requires_grad=True)
    action=AdaptiveAction({'proj.weight':AdaptiveFactors(u,c,v)}, {})
    delta=u@c@v.T;old={'proj.weight':(base['proj.weight'].double()+scale*delta).float()}
    original_norm=100*grouped.evaluate(old).mean_relative_change
    dense_ref=weakref.ref(delta);replacement_ref=weakref.ref(old['proj.weight'])
    values,thin=streamed_grouped_norm(old,grouped,action,scale,chunk_rows=2)
    assert thin.item()==original_norm.item()
    del delta,old,original_norm
    gc.collect()
    assert dense_ref() is None and replacement_ref() is None
    assert all(t.numel()<9*13 for t in thin.grad_fn.saved_tensors)
    assert values['proj.weight'].dtype==torch.float32 and values['proj.weight'].grad_fn is None
    from torch.utils._python_dispatch import TorchDispatchMode
    produced=[]
    class Shapes(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            result=func(*args,**(kwargs or {}))
            def record(value):
                if isinstance(value,torch.Tensor):produced.append(tuple(value.shape))
                elif isinstance(value,(tuple,list)):
                    for child in value:record(child)
            record(result)
            return result
    with Shapes():thin.backward()
    assert (9,13) not in produced and (13,9) not in produced
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in (u,c,v,scale))


def test_replacement_mutation_rejected():
    base={'proj.bias':torch.ones(5)}
    grouped=GroupedNormObjective(base,{'bias':('proj.bias',)})
    vector=torch.randn(5,dtype=torch.float64,requires_grad=True);scale=torch.tensor(.1,dtype=torch.float64,requires_grad=True)
    params={'proj.bias':(base['proj.bias'].double()+scale*vector).float()}
    values,loss=streamed_grouped_norm(params,grouped,AdaptiveAction({}, {'proj.bias':vector}),scale)
    values['proj.bias'].add_(1.)
    with pytest.raises(RuntimeError,match='mutated'):loss.backward()


@pytest.mark.parametrize('radius',[1e-6,1.])
@pytest.mark.parametrize('zero',[False,True])
def test_prepared_panel_full_loss_actor_private_and_vector_credit(radius,zero):
    import gc,weakref
    from tests.test_adaptive_preservation_actor import actual_actor_case
    from rewriter.adaptive_coordinate_chisel import AdaptiveCoordinateChisel
    from rewriter.experimental_factor_panel import FactorPanelCapture
    from rewriter.experimental_factor_norm import prepare_factor_panel
    from rewriter.tokenwise_reference import TokenwiseConfig,TokenwiseDual,tokenwise_panel
    rows=[];grads=[];counts=[]
    for direct in (False,True):
        engine,_,_,snapshot,grouped,inputs=actual_actor_case();inputs=inputs[:,:4]
        norm_evaluations=[0];original_evaluate=grouped.evaluate
        def evaluate(parameters):
            norm_evaluations[0]+=1
            return original_evaluate(parameters)
        grouped.evaluate=evaluate
        torch.manual_seed(647)
        actor=AdaptiveCoordinateChisel(rank=2,trajectory_dim=12,model_dim=128,num_layers=1,latent_tokens=2,factor_precision='fp64',residual_precision='fp32')
        private=actor.make_private(snapshot,state_key=('test',0,6),seed=77)
        with torch.no_grad():
            for head in actor.output_heads:
                if zero:head.weight.zero_();head.bias.zero_()
                else:head.weight.normal_(std=.001)
        teacher=[engine.cache_teacher(x[None],data_role='synthetic')[0] for x in inputs]
        capture=FactorPanelCapture(chunk_tokens=3)
        parameters,regularizer,stats,_,credit=build_adaptive_proposal(lambda snap:actor(snap,private),snapshot,engine,grouped,
            radius_fraction=radius,execution_capture=capture if direct else None)
        assert stats['radius_saturated']==(radius<1e-5 and not zero)
        if direct:
            original_matrix=weakref.ref(parameters['embed.weight'])
            original_bias=parameters['proj.bias']
            parameters,regularizer=prepare_factor_panel(parameters,grouped,capture,transpose_storage=engine.transpose_storage,chunk_rows=3)
            gc.collect()
            assert capture.norm_measurement is None
            assert norm_evaluations==[1]
            assert original_matrix() is None
            assert parameters['embed.weight'].is_leaf and parameters['embed.weight'].requires_grad
            assert parameters['proj.bias'] is original_bias
            with pytest.raises(RuntimeError,match='fresh'):prepare_factor_panel(parameters,grouped,capture)
        config=TokenwiseConfig(mean_tolerance=1e-10,tail_tolerance=1e-10,scale=1e-5,rho=.01,ranking_weight=.01,
            ranking_k=10,ranking_tie_policy='exact_teacher_ties',ranking_loss_mode='relative_gap')
        dual=TokenwiseDual([6],config=config,dual_lr=.01);dual.multipliers[6]=.1
        run=capture.run if direct else tokenwise_panel
        rows.append(run(engine,parameters,regularizer,inputs,config=config,dual=dual,state_id=6,state_weight=.5,actor_credit=credit,teacher_provider=lambda a,b:teacher[a]))
        counts.append(engine.compute.as_dict())
        assert norm_evaluations==[1]
        if direct:assert capture.norm_measurement is None
        grads.append({prefix+n:None if p.grad is None else p.grad.clone() for prefix,m in [('actor.',actor),('private.',private)] for n,p in m.named_parameters()})
        if direct:assert all(p.grad is None for p in parameters.values() if p.ndim==2)
    assert rows[0]==rows[1] and counts[0]==counts[1]
    assert grads[1]['actor.vector_output.weight'].norm()>0
    for n,a in grads[0].items():
        b=grads[1][n];assert (a is None)==(b is None),n
        if a is not None:torch.testing.assert_close(a,b,rtol=5e-4,atol=3e-5,msg=n)


@pytest.mark.parametrize('mutation',['replacement','clone','reference','group_object','measurement'])
def test_measured_norm_rejects_stale_values_and_consumes_capture(mutation):
    from rewriter.experimental_factor_panel import FactorPanelCapture
    from rewriter.experimental_factor_norm import prepare_factor_panel
    base={'proj.weight':torch.randn(9,13)}
    grouped=GroupedNormObjective(base,{'head':('proj.weight',)})
    u=torch.randn(9,2,dtype=torch.float64,requires_grad=True)
    c=torch.randn(2,3,dtype=torch.float64,requires_grad=True)
    v=torch.randn(13,3,dtype=torch.float64,requires_grad=True)
    scale=torch.tensor(.01,dtype=torch.float64,requires_grad=True)
    action=AdaptiveAction({'proj.weight':AdaptiveFactors(u,c,v)}, {})
    params={'proj.weight':(base['proj.weight'].double()+scale*action.matrices['proj.weight'].dense()).float()}
    capture=FactorPanelCapture();capture(action,scale)
    capture.capture_norm_measurement(params,grouped,grouped.evaluate(params))
    cached=capture.norm_measurement
    assert cached.mean.grad_fn is None and all(t.grad_fn is None for t in cached.norms.values())
    if mutation=='replacement':
        with torch.no_grad():params['proj.weight'].add_(.1)
    elif mutation=='clone':params={'proj.weight':params['proj.weight'].clone()}
    elif mutation=='reference':base['proj.weight'].add_(.1)
    elif mutation=='measurement':cached.norms['head'].add_(.1)
    else:grouped=GroupedNormObjective(base,{'head':('proj.weight',)})
    with pytest.raises(RuntimeError,match='stale norm'):prepare_factor_panel(params,grouped,capture)
    assert capture.phase=='consumed' and capture.norm_measurement is None and capture.action is None
    with pytest.raises(RuntimeError,match='consumed'):cached.consume(params,grouped)


def test_measurement_does_not_retain_any_dense_graph_and_is_one_shot():
    import gc,weakref
    from rewriter.experimental_factor_norm import MeasuredGroupedNorm
    base={'proj.bias':torch.ones(7)};grouped=GroupedNormObjective(base,{'mean':('proj.bias',)})
    x=torch.randn(7,dtype=torch.float64,requires_grad=True)
    params={'proj.bias':(base['proj.bias'].double()+x).float()}
    weak=weakref.ref(params['proj.bias']);result=grouped.evaluate(params)
    cached=MeasuredGroupedNorm(params,grouped,result)
    del params,result;gc.collect()
    assert weak() is None
    assert cached.mean.grad_fn is None
    cached.mean.add_(1.)
    with pytest.raises(RuntimeError,match='stale norm scalar'):cached.consume({},grouped)
