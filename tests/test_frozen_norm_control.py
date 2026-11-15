import json
import struct

import numpy as np

import pytest
import torch

from baseline.model import GPT
from rewriter.algebra import Anchors, Factors, RawAction
from rewriter.controller import ControllerSnapshot, MatrixObservation, VectorObservation
from rewriter.function_preservation import FunctionalPreservation
from rewriter.replay_buffer import PastReplay, SequenceBatch, sample_past_sequences
from scripts.evaluate_frozen_norm_control import evaluate, matched_replay, norm_only_action,matching_norm_control


def snapshot(weights, vectors, radius):
    generator = torch.Generator().manual_seed(970)
    matrices = {}
    for name, weight in weights.items():
        anchors = Anchors.random(*weight.shape, 2, generator=generator, dtype=weight.dtype)
        matrices[name] = MatrixObservation(anchors, torch.empty(0), torch.empty(0),
                                          (weight@anchors.B, weight.T@anchors.A))
    squared_norm = sum(w.square().sum() for w in [*weights.values(), *vectors.values()])
    return ControllerSnapshot(matrices, {n:VectorObservation(torch.empty(0),v) for n,v in vectors.items()},
                              torch.empty(0), squared_norm.sqrt(), torch.tensor(radius, dtype=torch.float64))


@pytest.mark.parametrize('radius', [0., .4, 100.])
def test_continuous_optimum_matches_dense_projection_and_beats_feasible_edits(radius):
    generator = torch.Generator().manual_seed(41)
    weights = {'rect':torch.randn(7, 5, generator=generator, dtype=torch.float64),
               'embedding_orientation':torch.randn(4, 9, generator=generator, dtype=torch.float64)}
    vectors = {'bias':torch.randn(7, generator=generator, dtype=torch.float64)}
    observation = snapshot(weights, vectors, radius)
    action, stats = norm_only_action(observation)
    dense_projection = {}
    for name, weight in weights.items():
        a, b = observation.matrices[name].anchors.A, observation.matrices[name].anchors.B
        # Dense projectors are allowed in this separate tiny algebra test.
        dense_projection[name] = weight@b@b.T + a@a.T@weight - a@a.T@weight@b@b.T
        torch.testing.assert_close(action.matrices[name].dense(), -stats['global_coefficient']*dense_projection[name])
        torch.testing.assert_close(a.T@action.matrices[name].P, torch.zeros(2,2,dtype=torch.float64),atol=1e-14,rtol=0)
    actual_change = sum(((weights[n]+f.dense()).square()-weights[n].square()).sum() for n,f in action.matrices.items())
    actual_change += sum(((vectors[n]+v).square()-vectors[n].square()).sum() for n,v in action.vectors.items())
    assert float(actual_change) == pytest.approx(stats['ideal_continuous_squared_norm_change'], abs=1e-12)
    assert stats['raw_edit_norm'] <= radius + 1e-12
    for _ in range(20):
        candidate = RawAction({n:Factors.canonical(torch.randn(o.anchors.A.shape,generator=generator,dtype=torch.float64),
            torch.randn(o.anchors.B.shape,generator=generator,dtype=torch.float64),o.anchors)
            for n,o in observation.matrices.items()},
            {n:torch.randn(v.shape,generator=generator,dtype=torch.float64) for n,v in vectors.items()})
        scale = radius * torch.rand((),generator=generator) / candidate.norm_sq().sqrt()
        change = 2*scale*candidate.contraction(observation.weights,observation.vector_weights)+scale.square()*candidate.norm_sq()
        assert float(change) >= float(actual_change) - 1e-12


def test_zero_weight_projection_and_invalid_anchors():
    observation = snapshot({'zero':torch.zeros(4,5,dtype=torch.float64)}, {'zero_bias':torch.zeros(4,dtype=torch.float64)}, 1.)
    action, stats = norm_only_action(observation)
    assert action.norm_sq() == 0
    assert stats['global_coefficient'] == 0
    observation.matrices['zero'].anchors.A.mul_(2)
    with pytest.raises(ValueError,match='orthonormal'):
        norm_only_action(observation)


def test_finite_audit_evaluation_counts_only_forward_passes_and_preserves_model():
    torch.manual_seed(890)
    model = GPT(vocab_size=8,num_layers=1,model_dim=128).float()
    original = {n:p.detach().clone() for n,p in model.named_parameters()}
    weights = {n:p.detach().T if n=='embed.weight' else p.detach() for n,p in model.named_parameters() if p.ndim==2}
    vectors = {n:p.detach() for n,p in model.named_parameters() if p.ndim==1}
    observation = snapshot(weights,vectors,.1)
    action,_ = norm_only_action(observation)
    engine = FunctionalPreservation(model,kl_dtype=torch.float64)
    with torch.no_grad():
        parameters = engine.edited_parameters(action)
        inputs = torch.tensor([[1,2,3],[4,5,6]],dtype=torch.int32)
        batch = SequenceBatch(inputs,inputs.long(),[])
        replay = PastReplay(batch,batch,{})
        source = dict(base_config=dict(seq_len=3,model=dict(vocab_size=8)),
                      final_audit_sequences=2,synthetic_audit_sequences=2)
        results = evaluate(engine,parameters,replay,source)
    assert set(results)=={'historical_audit','uniform','repeated','alternating'}
    assert all(result['sequences']==2 and result['mean_kl']>=-1e-14 for result in results.values())
    assert engine.compute.as_dict()==dict(teacher_forwards=8,teacher_tokens=24,
        student_forwards=8,student_tokens=24,student_backwards=0,student_backward_tokens=0)
    for name,p in model.named_parameters():
        assert torch.equal(p,original[name])
        assert p.grad is None


def test_audit_ids_order_and_dataset_shards_match_source_run(tmp_path):
    path = tmp_path/'fineweb_train_000000.bin'
    path.write_bytes(struct.pack('<256i',20240520,1,100,*([0]*253))+
                     (np.arange(100)%8).astype('<u2').tobytes())
    replay = sample_past_sequences([path],4,batch_tokens=8,seq_len=2,count=8)
    source = dict(base_config=dict(batch_tokens=8,seq_len=2),checkpoint_step=4,buffer_sequences=8)
    provenance = dict(metadata=replay.metadata,train=replay.train.provenance,audit=replay.audit.provenance)
    record = tmp_path/'replay_provenance.json'
    record.write_text(json.dumps(provenance))
    restored = matched_replay(source,tmp_path,[path])
    assert torch.equal(restored.audit.inputs,replay.audit.inputs)
    provenance['audit'].reverse()
    record.write_text(json.dumps(provenance))
    with pytest.raises(ValueError,match='audit replay IDs/order/shards'):
        matched_replay(source,tmp_path,[path])


def test_grouped_frontend_selects_matched_solver_and_preserves_gain_scope():
    model=GPT(vocab_size=8,num_layers=1,model_dim=128).float()
    base={n:p.detach() for n,p in model.named_parameters()}
    matrices={n:v.T if n=='embed.weight' else v for n,v in base.items() if v.ndim==2}
    vectors={n:v for n,v in base.items() if v.ndim==1}
    observed=snapshot(matrices,vectors,.1)
    action,stats,objective=matching_norm_control(observed,base,
        {'editable_scope':'grouped','norm_objective':'grouped_l2'})
    assert len(objective.groups)==15
    assert stats['ideal_mean_group_l2_percentage_reduction']>0
    assert action.matrices['embed.weight'].norm_sq()>0
    assert action.matrices['proj.weight'].norm_sq()>0
    assert action.vectors['norm1.gains'].abs().sum()==0
    assert stats['raw_edit_norm']<=float(observed.radius)*(1+1e-12)
    for wrong in ({'editable_scope':'grouped'}, {'editable_scope':'hidden','norm_objective':'grouped_l2'}):
        with pytest.raises(ValueError,match='do not match'):
            matching_norm_control(observed,base,wrong)
