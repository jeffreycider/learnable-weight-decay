import dataclasses

import pytest
import torch

from baseline.model import GPT
from rewriter.algebra import Anchors,Factors,RawAction
from rewriter.controller import ControllerSnapshot,GlobalRewriter,MatrixObservation,VectorObservation
from rewriter.edit_scope import (editable_names,editable_weight_norm_sq,scoped_action,scoped_snapshot,
    keep_readonly_parameters,readonly_hashes,verify_readonly)
from rewriter.function_preservation import FunctionalPreservation
from scripts.evaluate_frozen_norm_control import norm_only_action


def small_observation():
    generator=torch.Generator().manual_seed(718)
    weights={n:torch.randn(7,5,generator=generator) for n in ('embed.weight','proj.weight','blocks.0.weight')}
    vectors={n:torch.randn(7,generator=generator) for n in ('proj.bias','blocks.0.bias')}
    matrices={}
    for name,weight in weights.items():
        anchors=Anchors.random(7,5,2,generator=generator)
        matrices[name]=MatrixObservation(anchors,torch.randn(7,25,generator=generator),
            torch.randn(5,25,generator=generator),(weight@anchors.B,weight.T@anchors.A))
    snapshot=ControllerSnapshot(matrices,{n:VectorObservation(torch.randn(7,18,generator=generator),v)
        for n,v in vectors.items()},torch.randn(3,12,generator=generator),torch.tensor(50.),torch.tensor(.5))
    return snapshot,dict(weights,**vectors)


def test_excluded_weights_do_not_contribute_to_norm_radius_or_global_guard():
    snapshot,parameters=small_observation()
    editable=editable_names(parameters,'hidden')
    norm=editable_weight_norm_sq(parameters,editable)
    expected=sum(v.double().square().sum() for n,v in parameters.items() if n.startswith('blocks.'))
    torch.testing.assert_close(norm,expected,rtol=0,atol=0)
    larger={n:v if n in editable else v*1e6 for n,v in parameters.items()}
    assert editable_weight_norm_sq(larger,editable)==norm
    scoped=scoped_snapshot(snapshot,norm,.01,scope='hidden')
    assert scoped.matrices is snapshot.matrices and scoped.vectors is snapshot.vectors
    assert scoped.trajectories is snapshot.trajectories
    torch.testing.assert_close(scoped.radius,norm.sqrt().float()*.01)
    raw,_=norm_only_action(snapshot)
    masked=scoped_action(raw,editable)
    inflated=RawAction({n:f if n in editable else Factors(f.P*1e6,f.Q*1e6,f.anchors)
                        for n,f in raw.matrices.items()},
                       {n:v if n in editable else v*1e6 for n,v in raw.vectors.items()})
    assert scoped_action(inflated,editable).norm_sq()==masked.norm_sq()
    assert masked.matrices['embed.weight'].norm_sq()==0
    assert masked.matrices['proj.weight'].norm_sq()==0
    assert masked.vectors['proj.bias'].abs().sum()==0
    # Historical all-scope snapshot keeps precisely the old FP32 radius.
    assert scoped_snapshot(snapshot,norm,.01,scope='all') is snapshot


def test_masked_head_and_embedding_still_influence_hidden_actions_globally():
    snapshot,parameters=small_observation()
    actor=GlobalRewriter(rank=2,trajectory_dim=12,coordinate_width=8,global_width=16,
        pool_queries=2,latents=4,blocks=1,heads=2,ff_width=32)
    with torch.no_grad():
        actor.row_decoder[-1].weight.normal_(std=.2)
        actor.column_decoder[-1].weight.normal_(std=.2)
    names=editable_names(parameters,'hidden')
    first=scoped_action(actor(snapshot),names)
    matrices=dict(snapshot.matrices)
    matrices['embed.weight']=dataclasses.replace(matrices['embed.weight'],rows=matrices['embed.weight'].rows+7)
    second=scoped_action(actor(dataclasses.replace(snapshot,matrices=matrices)),names)
    assert not torch.allclose(first.matrices['blocks.0.weight'].P,second.matrices['blocks.0.weight'].P)
    assert second.matrices['embed.weight'].norm_sq()==0


def test_actual_stored_embedding_and_entire_head_unchanged_with_hidden_gradients():
    model=GPT(vocab_size=8,num_layers=1,model_dim=128)
    engine=FunctionalPreservation(model)
    names=editable_names(engine.parameters,'hidden')
    hashes=readonly_hashes(engine.parameters,names)
    generator=torch.Generator().manual_seed(55)
    matrices={};vectors={};leaves={}
    for name,parameter in engine.parameters.items():
        if parameter.ndim==2:
            shape=parameter.T.shape if name=='embed.weight' else parameter.shape
            anchor=Anchors.random(*shape,2,generator=generator)
            p=(torch.randn(anchor.A.shape,generator=generator)*.001).requires_grad_()
            q=(torch.randn(anchor.B.shape,generator=generator)*.001).requires_grad_()
            matrices[name]=Factors.canonical(p,q,anchor);leaves[name]=p
        else:
            vectors[name]=(torch.randn(parameter.shape,generator=generator)*.001).requires_grad_()
            leaves[name]=vectors[name]
    action=scoped_action(RawAction(matrices,vectors),names)
    replacements=keep_readonly_parameters(engine.edited_parameters(action),engine.parameters,names)
    for name in ('embed.weight','proj.weight','proj.bias'):
        assert replacements[name] is engine.parameters[name]
    assert verify_readonly(replacements,names,hashes)['unchanged']
    change,edit_sq=engine.realized_parameter_norm_change(replacements)
    (change+edit_sq).backward()
    assert all(leaves[name].grad is None for name in ('embed.weight','proj.weight','proj.bias'))
    assert leaves['blocks.0.mlp.fc.weight'].grad.abs().sum()>0
    broken=dict(replacements,**{'proj.bias':replacements['proj.bias']+1})
    with pytest.raises(RuntimeError,match='stored bits changed'):
        verify_readonly(broken,names,hashes)


def test_norm_only_control_uses_same_scope_and_budget():
    snapshot,parameters=small_observation()
    names=editable_names(parameters,'hidden')
    snapshot=scoped_snapshot(snapshot,editable_weight_norm_sq(parameters,names),.01,scope='hidden')
    action,stats=norm_only_action(snapshot,names)
    assert action.matrices['embed.weight'].norm_sq()==0
    assert action.matrices['proj.weight'].norm_sq()==0
    assert action.vectors['proj.bias'].abs().sum()==0
    assert stats['raw_edit_norm']==pytest.approx(float(snapshot.radius),rel=1e-6)
    # Whole contraction equals editable-only contraction because excluded writes are zero.
    full=action.norm_change(snapshot.weights,snapshot.vector_weights)
    selected=RawAction({n:f for n,f in action.matrices.items() if n in names},
                       {n:v for n,v in action.vectors.items() if n in names})
    torch.testing.assert_close(full,selected.norm_change(snapshot.weights,snapshot.vector_weights))


def test_scope_validation_and_explicit_all_preserve_outputs():
    snapshot,parameters=small_observation()
    action,_=norm_only_action(snapshot)
    result=scoped_action(action,editable_names(parameters,'all'))
    for name,f in action.matrices.items():assert result.matrices[name] is f
    for name,v in action.vectors.items():assert result.vectors[name] is v
    with pytest.raises(ValueError):editable_names(parameters,'mystery')
    with pytest.raises(ValueError):editable_names({'other.weight':torch.ones(2,2)},'hidden')
