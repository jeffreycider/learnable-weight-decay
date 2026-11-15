import json

import pytest
import torch

from rewriter.prediction_changes import PredictionChanges


def test_identity_and_additive_offsets_leave_predictions_unchanged():
    t = torch.tensor([[3., 2., 1., 0.], [0., 1., 2., 3.]])
    metric = PredictionChanges(top_k=3)
    metric.update(t, t + 10)
    assert metric.tokens == 2
    assert all(v == 0 for v in metric.counts.values())
    assert not metric.summary()['strict_ranking_witnesses']


def test_small_kl_can_flip_argmax_and_records_probability_margin():
    t = torch.tensor([[1e-4, 0., -2., -3.]], requires_grad=True)
    s = torch.tensor([[0., 1e-4, -2., -3.]], requires_grad=True)
    metric = PredictionChanges(top_k=3)
    metric.update(t, s, token_ids=[['sequence_7', 19]])
    q, p = t.double().log_softmax(-1), s.double().log_softmax(-1)
    assert float((q.exp() * (q - p)).sum().detach()) < 1e-8
    out = metric.summary()
    assert out['argmax_changes'] == 1 and out['topk_membership_changes'] == 0
    assert out['strict_ranking_witness_tokens'] == 1
    assert out['strict_ranking_witnesses'][0]['context_id'] == ['sequence_7', 19]
    expected = float((q.exp()[0, 0] - q.exp()[0, 1]).detach())
    assert out['teacher_margins_at_argmax_changes']['top1_top2_probability_gap']['min'] == pytest.approx(expected)
    assert t.grad is None and s.grad is None
    json.dumps(out, allow_nan=False)


def test_membership_crossing_and_inner_ranking_changes_without_argmax_change():
    t = torch.tensor([[5., 4., 3., 2.], [5., 4., 3., 2.]])
    s = torch.tensor([[5., 4., 1., 2.], [5., 3., 4., 2.]])
    one = PredictionChanges(top_k=3)
    one.update(t, s)
    assert one.counts['argmax_changes'] == 0
    assert one.counts['topk_membership_changes'] == 1
    assert one.counts['topk_order_changes'] == 2
    assert one.counts['strict_ranking_witness_tokens'] == 2
    chunks = PredictionChanges(top_k=3)
    for a, b in zip(t, s):
        chunks.update(a[None], b[None])
    assert one.summary() == chunks.summary()
    for w in one.witnesses:
        assert w['teacher_logit_a'] > w['teacher_logit_b']
        assert w['student_logit_a'] < w['student_logit_b']


def test_tied_teacher_winners_are_not_strict_inversions():
    metric = PredictionChanges(top_k=2)
    metric.update(torch.tensor([[1., 1., 0.]]), torch.tensor([[1., 1.1, 0.]]))
    assert metric.counts['argmax_changes'] == 1
    assert metric.counts['argmax_changes_with_teacher_top_tie'] == 1
    assert metric.counts['strict_ranking_witness_tokens'] == 0


def test_invalid_input_is_rejected():
    metric = PredictionChanges(top_k=2)
    with pytest.raises(ValueError, match='finite'):
        metric.update(torch.tensor([[float('nan'), 0.]]), torch.zeros(1, 2))
    with pytest.raises(ValueError, match='context ID'):
        metric.update(torch.zeros(1, 2), torch.zeros(1, 2), token_ids=[])


def test_probability_margin_chunking_handles_pervasive_flips():
    t = torch.tensor([[2., 1., 0.]]).repeat(600, 1)
    s = torch.tensor([[1., 2., 0.]]).repeat(600, 1)
    one, chunks = PredictionChanges(top_k=2), PredictionChanges(top_k=2)
    one.update(t, s)
    for a, b in zip(t.split(73), s.split(73)):
        chunks.update(a, b)
    assert one.summary() == chunks.summary()
    assert one.counts['argmax_changes'] == 600
