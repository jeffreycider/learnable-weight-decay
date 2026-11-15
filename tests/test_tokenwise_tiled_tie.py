import pytest
import torch

from rewriter.tokenwise_reference import TokenwiseConfig, logit_terms_and_cotangent as reference
from rewriter.tokenwise_tiled_tie import logit_terms_and_cotangent as candidate


@pytest.mark.parametrize('shape', [(2, 3, 137), (7, 10)])
@pytest.mark.parametrize('backward,balance', [(True, False), (True, True), (False, False), (False, True)])
@pytest.mark.parametrize('multiplier', [0., .37])
def test_combined_tie_objective_all17_metrics_and_weighted_cotangent(shape, backward, balance, multiplier):
    torch.manual_seed(103)
    teacher = torch.randint(0, 8, shape).double().log_softmax(-1)
    student = torch.randn(shape)
    config = TokenwiseConfig(.01, .02, .1, .01, ranking_weight=3.7,
                             ranking_k=100, ranking_tie_policy='exact_teacher_ties')
    inputs = torch.randint(shape[-1], shape[:-1]) if len(shape)==3 else None
    kwargs = dict(config=config, multiplier=multiplier, panel_tokens=11, chunk_tokens=2,
                  backward=backward, balance=balance, inputs=inputs)
    ga, sa, ka = candidate(student, teacher, **kwargs)
    gr, sr, kr = reference(student, teacher, **kwargs)
    assert len(sa) == len(sr) == 17
    torch.testing.assert_close(ka, kr, atol=5e-14, rtol=1e-12)
    torch.testing.assert_close(sa, sr, atol=1e-10, rtol=1e-6)
    assert torch.equal(sa[3:6], sr[3:6]) and torch.equal(sa[13:17], sr[13:17])
    if backward: torch.testing.assert_close(ga, gr, atol=1e-7, rtol=3e-6)
    else: assert ga is None and gr is None


def test_zero_rank_weight_still_measures_strict_counts():
    teacher = torch.tensor([[4., 3., 3., 0.]], dtype=torch.float64).log_softmax(-1)
    student = torch.tensor([[1., 4., 3., 0.]])
    config = TokenwiseConfig(.01, .02, .1, .01, ranking_weight=0.,
                             ranking_k=3, ranking_tie_policy='exact_teacher_ties')
    kwargs = dict(config=config, multiplier=.5, panel_tokens=1, backward=True)
    ga, sa, ka = candidate(student, teacher, **kwargs)
    gr, sr, kr = reference(student, teacher, **kwargs)
    torch.testing.assert_close(sa, sr, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(ga, gr, atol=1e-7, rtol=1e-6)
    assert sa[14]>0 and sa[16]>0


def test_uneven_image_microbatch_normalization_uses_full_panel_count_once():
    torch.manual_seed(711)
    teacher = torch.randint(0, 4, (7, 10)).double().log_softmax(-1)
    student = torch.randn(7, 10)
    config = TokenwiseConfig(.01, .02, .1, .01, ranking_weight=3.7,
                             ranking_k=100, ranking_tie_policy='exact_teacher_ties')
    kwargs = dict(config=config, multiplier=.37, panel_tokens=7, chunk_tokens=2)
    whole = candidate(student, teacher, **kwargs)
    pieces = [candidate(student[a:b], teacher[a:b], **kwargs) for a,b in ((0,3),(3,7))]
    torch.testing.assert_close(torch.cat([p[0] for p in pieces]), whole[0], rtol=0, atol=0)
    torch.testing.assert_close(sum(p[1] for p in pieces), whole[1], rtol=1e-14, atol=1e-12)
    torch.testing.assert_close(torch.cat([p[2] for p in pieces]), whole[2], rtol=0, atol=0)


def test_new_helper_rejects_legacy_semantics():
    config = TokenwiseConfig(.01, .02, .1, .01)
    with pytest.raises(ValueError, match='exact_teacher_ties'):
        candidate(torch.zeros(1, 10), torch.zeros(1, 10, dtype=torch.float64),
                  config=config, multiplier=0., panel_tokens=1)
