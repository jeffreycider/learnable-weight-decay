import dataclasses
from types import SimpleNamespace

import pytest
import torch

from rewriter.algebra import Anchors, Factors, Measurements
from rewriter.controller import ControllerSnapshot, GlobalRewriter, MatrixObservation, VectorObservation
from scripts.profile_controller import (compare_signatures, profile, signatures,
                                        synthetic_banks, validate_measurements,run,_load)


def fixture():
    generator=torch.Generator().manual_seed(891)
    anchors=Anchors.random(7,5,2,generator=generator)
    snapshot=ControllerSnapshot({'matrix':MatrixObservation(anchors,
        torch.randn(7,25,generator=generator),torch.randn(5,25,generator=generator),
        (torch.randn(7,2,generator=generator),torch.randn(5,2,generator=generator)))},
        {'bias':VectorObservation(torch.randn(7,18,generator=generator),torch.ones(7))},
        torch.randn(3,12,generator=generator),torch.tensor(10.),torch.tensor(.1))
    with torch.random.fork_rng():
        torch.manual_seed(144)
        actor=GlobalRewriter(rank=2,trajectory_dim=12,coordinate_width=8,global_width=16,
                             pool_queries=2,latents=4,blocks=1,heads=2,ff_width=32)
        with torch.no_grad():
            for decoder in (actor.row_decoder,actor.column_decoder,actor.vector_decoder):
                decoder[-1].weight.normal_(std=.02)
    return actor,snapshot


@pytest.mark.parametrize('with_measurements',[False,True])
def test_profile_is_thin_only_counts_iterations_and_never_updates_weights(monkeypatch,with_measurements):
    actor,snapshot=fixture()
    original={n:p.detach().clone() for n,p in actor.named_parameters()}
    rng=torch.random.get_rng_state().clone()
    measurements=synthetic_banks(snapshot) if with_measurements else None
    if measurements:validate_measurements(snapshot,measurements)
    forwards=[]
    actor.register_forward_hook(lambda *args:forwards.append(1))
    def forbidden(*args,**kwargs):raise AssertionError('dense edit forbidden in cached timing')
    monkeypatch.setattr(Factors,'dense',forbidden)
    result=profile(actor,snapshot,measurements,warmup=1,iterations=3)
    assert len(forwards)==4 and result['iterations']==3
    assert result['gpt_forwards']==result['gpt_backwards']==result['optimizer_steps']==0
    assert result['controller_gradient_norm']>0
    assert result['timing_ms']['total_wall_ms']['mean']>0
    assert result['peak_allocated_bytes'] is None
    assert torch.equal(rng,torch.random.get_rng_state())
    for n,p in actor.named_parameters():assert torch.equal(p,original[n])


def test_signature_comparison_checks_full_actions_and_gradients():
    actor,snapshot=fixture()
    first=signatures(actor,snapshot,synthetic_banks(snapshot))
    second=signatures(actor,snapshot,synthetic_banks(snapshot))
    assert compare_signatures(first,second)['within_tolerance']
    corrupted=dict(second,gradient=second['gradient'].clone())
    corrupted['gradient'][0]+=1
    comparison=compare_signatures(first,corrupted)
    assert not comparison['within_tolerance']
    assert comparison['action']['within_tolerance']
    assert not comparison['gradient']['within_tolerance']


def test_rejects_audit_plus_training_bank_and_unbounded_clock():
    actor,snapshot=fixture()
    measurements=synthetic_banks(snapshot)
    wrong=Measurements({n:tuple(x.repeat(2,1,1) for x in values)
                         for n,values in measurements.matrices.items()},measurements.vectors)
    with pytest.raises(ValueError,match='four training probes'):
        validate_measurements(snapshot,wrong)
    with pytest.raises(ValueError,match='bounded diagnostic'):
        profile(actor,snapshot,measurements,iterations=1000)


def test_hidden_scope_inferred_from_saved_observation_and_actor_and_cost_labels(tmp_path):
    _,snapshot=fixture()
    matrix=snapshot.matrices['matrix'];vector=snapshot.vectors['bias']
    snapshot=dataclasses.replace(snapshot,
        matrices={'embed.weight':matrix,'proj.weight':matrix,'blocks.0.weight':matrix},
        vectors={'proj.bias':vector,'blocks.0.bias':vector})
    observation=tmp_path/'observation.pt';checkpoint=tmp_path/'controller.pt'
    torch.save(dict(snapshot=snapshot,calibration=dict(editable_scope='hidden')),observation)
    actor=GlobalRewriter(rank=2,trajectory_dim=12)
    torch.save(dict(controller=actor.state_dict(),config=dict(editable_scope='hidden')),checkpoint)
    args=SimpleNamespace(observation=observation,actor_checkpoint=checkpoint,measurements=None,
        synthetic_measurements=True,device='cpu',compile_worker=False,warmup=0,iterations=1)
    result=run(args)
    assert result['editable_scope']=='hidden'
    assert result['objective_banks']=='synthetic_four_probe_banks_for_performance_only'
    assert result['optimizer_steps']==result['gpt_forwards']==result['gpt_backwards']==0
    assert 'excludes Adam' in result['timing_scope'] and 'GPT' in result['timing_scope']
    loaded,observed,measurements,_,_= _load(args)
    editable={'blocks.0.weight','blocks.0.bias'}
    output=signatures(loaded,observed,measurements,editable=editable)
    assert torch.isfinite(output['gradient']).all()
    torch.save(dict(controller=actor.state_dict(),config=dict(editable_scope='all')),checkpoint)
    with pytest.raises(ValueError,match='editable scopes differ'):
        _load(args)


def test_historical_observation_without_scope_still_means_all(tmp_path):
    _,snapshot=fixture()
    observation=tmp_path/'observation.pt'
    torch.save(dict(snapshot=snapshot,calibration={}),observation)
    args=SimpleNamespace(observation=observation,actor_checkpoint=None,measurements=None,
        synthetic_measurements=False,device='cpu')
    _,_,measurements,label,scope=_load(args)
    assert scope=='all' and measurements is None
    assert label=='norm_and_trust_only_no_preservation_banks'


def test_grouped_profile_fails_before_substituting_old_cached_objective(tmp_path):
    _,snapshot=fixture()
    observation=tmp_path/'observation.pt'
    torch.save(dict(snapshot=snapshot,calibration={'editable_scope':'grouped'}),observation)
    args=SimpleNamespace(observation=observation)
    with pytest.raises(NotImplementedError,match='per-group reference'):
        _load(args)
