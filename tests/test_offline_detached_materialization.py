import pytest
import torch
from rewriter.adaptive_preservation import build_adaptive_proposal
from rewriter.adaptive_action import AdaptiveAction,AdaptiveFactors
from rewriter.adaptive_coordinate_chisel import AdaptiveCoordinateChisel
from rewriter.adaptive_sensing import query_weight_cores
from rewriter.experimental_factor_panel import FactorPanelCapture
from rewriter.experimental_factor_norm import prepare_factor_panel
from rewriter.tokenwise_reference import TokenwiseConfig,TokenwiseDual
from tests.test_adaptive_preservation_actor import actual_actor_case


@pytest.mark.parametrize('radius',[1e-6,1.])
@pytest.mark.parametrize('zero',[False,True])
def test_detached_build_preserves_live_read_guard_vector_embedding_and_actor_credit(monkeypatch,radius,zero):
    torch.set_num_threads(1);rows=[];gradients=[];stored=[];statistics=[]
    for detached in (False,True):
        engine,_,_,snapshot,grouped,inputs=actual_actor_case();inputs=inputs[:,:4]
        # Repeated embedding IDs, while each forward is still a separate microbatch.
        inputs[:,1]=inputs[:,0]
        torch.manual_seed(991)
        actor=AdaptiveCoordinateChisel(rank=2,trajectory_dim=12,model_dim=128,num_layers=1,latent_tokens=2,fresh_weight_core=True,factor_precision='fp64',residual_precision='fp32')
        private=actor.make_private(snapshot,state_key=('test',0,6),seed=77)
        with torch.no_grad():
            for head in actor.output_heads:
                if not zero:head.weight.normal_(std=.001)
            actor.blocks[0].attn.proj.weight.normal_(std=.001)
        def proposal(snap):
            cores=query_weight_cores(engine.parameters,private,embedding_names=('embed.weight',))
            action=actor(snap,private,weight_cores=cores)
            matrices=dict(action.matrices)
            # Same factor object contributes through two separate GPT uses.
            matrices['blocks.0.attn.k.weight']=matrices['blocks.0.attn.q.weight']
            return AdaptiveAction(matrices,action.vectors)
        capture=FactorPanelCapture(chunk_tokens=3);dense_grad_modes=[];original=AdaptiveFactors.dense
        def dense(factor):
            dense_grad_modes.append(torch.is_grad_enabled());return original(factor)
        with monkeypatch.context() as patch:
            patch.setattr(AdaptiveFactors,'dense',dense)
            parameters,norm,stats,editable,credit=build_adaptive_proposal(proposal,snapshot,engine,grouped,radius_fraction=radius,execution_capture=capture,materialization_grad=not detached)
        assert dense_grad_modes and all(mode==(not detached) for mode in dense_grad_modes)
        assert credit is not None
        if detached:
            assert not norm.requires_grad
            assert all(p.grad_fn is None and not p.requires_grad for p in parameters.values())
        authoritative={n:p.detach().clone() for n,p in parameters.items()}
        parameters,norm=prepare_factor_panel(parameters,grouped,capture,transpose_storage=engine.transpose_storage,chunk_rows=3)
        assert norm.requires_grad
        assert all(torch.equal(parameters[n],v) for n,v in authoritative.items())
        assert parameters['proj.bias'].requires_grad
        assert parameters['embed.weight'].requires_grad and parameters['embed.weight'].grad_fn is None
        stored.append(authoritative);statistics.append(stats)
        teacher=[engine.cache_teacher(x[None],data_role='synthetic')[0] for x in inputs]
        config=TokenwiseConfig(mean_tolerance=1e-10,tail_tolerance=1e-10,scale=1e-5,rho=.01,ranking_weight=.01,ranking_k=10,ranking_tie_policy='exact_teacher_ties',ranking_loss_mode='relative_gap')
        dual=TokenwiseDual([6],config=config,dual_lr=.01);dual.multipliers[6]=.1
        rows.append(capture.run(engine,parameters,norm,inputs,config=config,dual=dual,state_id=6,state_weight=.5,actor_credit=credit,teacher_provider=lambda a,b:teacher[a]))
        gradients.append({prefix+n:None if p.grad is None else p.grad.clone() for prefix,m in [('actor.',actor),('private.',private)] for n,p in m.named_parameters()})
        assert capture.detached_editable is None and capture.phase=='consumed'
    assert rows[0]==rows[1] and statistics[0]==statistics[1]
    assert statistics[1]['radius_saturated']==(radius<1e-5 and not zero)
    assert all(torch.equal(stored[0][n],stored[1][n]) for n in stored[0])
    for n,a in gradients[0].items():
        b=gradients[1][n];assert (a is None)==(b is None),n
        if a is not None:torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10,msg=n)
    assert gradients[1]['actor.vector_output.weight'].norm()>0
    if not zero:
        assert any(g is not None and g.norm()>0 for n,g in gradients[1].items() if n.startswith('private.matrix_rows'))


def test_detached_capture_refuses_unprepared_run():
    capture=FactorPanelCapture();capture.phase='captured';capture.detached_editable=frozenset()
    with pytest.raises(RuntimeError,match='prepare_factor_panel'):capture.run(None,{},None,None)
    assert capture.phase=='consumed' and capture.detached_editable is None


def test_detached_vector_mutation_rejected_before_reconstruction():
    from rewriter.adaptive_action import AdaptiveAction
    capture=FactorPanelCapture();vector=torch.ones(3,dtype=torch.float64,requires_grad=True)
    capture(AdaptiveAction({}, {'proj.bias':vector}),torch.ones((),dtype=torch.float64))
    capture.capture_detached_materialization({'proj.bias'})
    with torch.no_grad():vector.add_(1.)
    with pytest.raises(RuntimeError,match='mutated'):
        prepare_factor_panel({},None,capture)
    assert capture.phase=='consumed'
