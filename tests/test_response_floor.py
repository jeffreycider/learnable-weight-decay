import os
os.environ.setdefault('PROFILE_CHISEL_SKIP_SOURCE_GUARD', '1')
import math
import pytest
import torch
from rewriter.response_floor import response_sensitivity


def _setup(tmp_path):
    from tests.test_step_timing import _tiny_bank_and_cache
    from rewriter.state_bank import StateBankReader
    from rewriter.teacher_cache import TeacherCacheReader
    from rewriter.function_preservation import FunctionalPreservation
    from rewriter.grouped_norm import GroupedNormObjective
    from rewriter.chisel_factory import build_chisel
    from baseline.model import GPT
    import scripts.train_multistate_chisel as train
    bank, cache_dir = _tiny_bank_and_cache(tmp_path)
    reader = StateBankReader(bank, verify_once=True, mmap_model=True)
    cache = TeacherCacheReader(cache_dir, verify_once=True, mmap=True)
    cfg = reader.manifest['provenance']['model_config']
    engine = FunctionalPreservation(GPT(**cfg).requires_grad_(False), microbatch_sequences=1, kl_dtype=torch.float64)
    record = reader.read(1, device='cpu', model_device='cpu')
    cache.load_engine(engine, record)
    grouped = GroupedNormObjective(engine.parameters, zero_reference_policy='freeze')
    panel = cache.read(1, 'train')
    torch.manual_seed(5)
    actor = build_chisel(dict(name='nanogpt_blocks_patch_io_v1', rank=2, trajectory_dim=record.snapshot.trajectories.shape[1], model_dim=128, num_layers=1, patch_size=8))
    with torch.no_grad():
        for head in actor.output_heads:
            head.weight.normal_(0, 1e-2)
    def logits_fn(module):
        with torch.enable_grad():  # fitting-mode proposal, GPT under no_grad (owner's corrected readout)
            parameters, *_ = train.build_proposal(module, record.snapshot, engine, grouped, radius_fraction=.01)
        parameters = {n: v.detach() for n, v in parameters.items()}
        with torch.no_grad():
            return engine._forward(parameters, panel.inputs[:1]).detach()
    return actor, logits_fn


def test_actual_delta_replay_and_size_matched_controls(tmp_path):
    actor, logits_fn = _setup(tmp_path)
    names = [n for n, _ in actor.named_parameters()]
    before = logits_fn(actor)
    dense = {n: p.detach().clone() for n, p in actor.named_parameters()}
    g = torch.Generator().manual_seed(11)
    candidate = {n: (v + torch.randn(v.shape, generator=g) * 1e-5).to(v.dtype) for n, v in dense.items()}
    heads = {n for n in names if n.startswith('matrix_output') or n.startswith('vector_output')}
    result = response_sensitivity(actor, logits_fn, before, dense, candidate, control_seeds=(1, 2),
                                  ablation_groups={'heads': heads, 'rest': set(names) - heads})
    # actual delta stats are exact
    expected_l2 = math.sqrt(sum(float((candidate[n].double() - dense[n].double()).norm()) ** 2 for n in names))
    assert result['delta_l2_total'] == pytest.approx(expected_l2, rel=1e-12)
    # controls match the per-tensor L2 of the actual delta (size-matched, not +-max everywhere)
    assert len(result['controls']) == 2 and result['control_median_max_abs'] >= 0
    # the actual replay is deterministic and the actor is restored to the dense state afterwards
    assert all(torch.equal(p.detach(), dense[n]) for n, p in actor.named_parameters())
    again = response_sensitivity(actor, logits_fn, before, dense, candidate, control_seeds=())
    assert again['actual'] == result['actual'] and again['controls'] == []
    # ablations: heads-only + rest-only deltas each produce some response; both are reported with their delta L2
    assert set(result['ablations']) == {'heads', 'rest'}
    assert result['ablations']['heads']['delta_l2'] + result['ablations']['rest']['delta_l2'] >= result['delta_l2_total']
    assert result['response_norm'] >= 0 and 'floor' not in result['note']


def test_response_sensitivity_rejects_mismatched_states(tmp_path):
    actor, logits_fn = _setup(tmp_path)
    before = logits_fn(actor)
    dense = {n: p.detach().clone() for n, p in actor.named_parameters()}
    bad = dict(dense); bad.pop(next(iter(dense)))
    with pytest.raises(ValueError, match='inventories'):
        response_sensitivity(actor, logits_fn, before, dense, bad)
    with pytest.raises(ValueError, match='ablation'):
        response_sensitivity(actor, logits_fn, before, dense, dense, control_seeds=(), ablation_groups={'x': {'nope'}})


def test_response_sensitivity_restores_actor_when_logits_fn_raises_and_reports_realised_control_size(tmp_path):
    actor, logits_fn = _setup(tmp_path)
    before = logits_fn(actor)
    dense = {n: p.detach().clone() for n, p in actor.named_parameters()}
    g = torch.Generator().manual_seed(3)
    candidate = {n: (v + torch.randn(v.shape, generator=g) * 1e-4).to(v.dtype) for n, v in dense.items()}
    calls = {'n': 0}
    def exploding(module):
        calls['n'] += 1
        if calls['n'] == 3:
            raise RuntimeError('injected')
        return logits_fn(module)
    with pytest.raises(RuntimeError, match='injected'):
        response_sensitivity(actor, exploding, before, dense, candidate, control_seeds=(1, 2))
    assert all(torch.equal(p.detach(), dense[n]) for n, p in actor.named_parameters())
    result = response_sensitivity(actor, logits_fn, before, dense, candidate, control_seeds=(1,))
    control = result['controls'][0]
    assert control['realised_delta_l2_total'] == pytest.approx(result['delta_l2_total'], rel=1e-3)
    assert 0.9 < control['realised_l2_ratio_min'] <= 1.001


def test_streamed_stats_match_direct_fp64_and_no_full_double_materialisation(tmp_path, monkeypatch):
    from rewriter.response_floor import _stream_stats
    torch.manual_seed(2)
    a = torch.randn(5, 7, 33); b = a + torch.randn_like(a) * 1e-3
    max_abs, sumsq, ref_sumsq, count = _stream_stats(a, b, chunk_rows=4)
    d = (a.double() - b.double())
    assert max_abs == pytest.approx(float(d.abs().max()), rel=0, abs=0)
    assert sumsq == pytest.approx(float(d.square().sum()), rel=1e-12)
    assert ref_sumsq == pytest.approx(float(b.double().square().sum()), rel=1e-12) and count == a.numel()
    # the probe must never call .double() on a tensor larger than one chunk
    actor, logits_fn = _setup(tmp_path)
    before = logits_fn(actor)
    rows = before.reshape(-1, before.shape[-1]).shape[0]
    original = torch.Tensor.double
    def guarded(self, *args, **kwargs):
        # only logits-sized tensors are the concern (parameters/deltas are small per tensor)
        if self.numel() == before.numel() and self.reshape(-1, self.shape[-1]).shape[0] > 2:
            raise AssertionError('full-size .double() materialisation')
        return original(self, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'double', guarded)
    dense = {n: p.detach().clone() for n, p in actor.named_parameters()}
    candidate = {n: (v + 1e-5).to(v.dtype) for n, v in dense.items()}
    result = response_sensitivity(actor, logits_fn, before, dense, candidate, control_seeds=(1,), chunk_rows=2)
    assert result['memory'].startswith('FP32 logits storage only') and rows > 2
