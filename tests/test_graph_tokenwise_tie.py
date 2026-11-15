from types import SimpleNamespace

import pytest
import torch

from rewriter.graph_tokenwise import TokenwiseThinGradientGraph
from rewriter.graph_tokenwise_tie import TieAwareTokenwiseGraph, TieAwareTokenwiseGraphAccumulator
from rewriter.rank_preservation import topk_preservation_flags
from rewriter.ranking_loss import tie_aware_topk_margin_loss
from rewriter.tokenwise_reference import TokenwiseConfig
from rewriter.tokenwise_reference import TokenwiseDual
from rewriter.offline_graph_backend import OfflineGraphAccumulator


@pytest.mark.parametrize('vocabulary,k', [(137, 100), (10, 100)])
def test_graph_loss_and_all_literal_strict_counts_use_effective_vocabulary(vocabulary, k):
    torch.manual_seed(18)
    teacher = torch.randint(0, 8, (2, 3, vocabulary)).double()
    student = torch.randn(2, 3, vocabulary, requires_grad=True)
    graph = object.__new__(TieAwareTokenwiseGraph)
    graph.token_config = TokenwiseConfig(.01,.01,.1,.01,ranking_k=k,ranking_tie_policy='exact_teacher_ties')
    loss, selected = graph._ranking_loss(teacher, student)
    expected = tie_aware_topk_margin_loss(teacher, student.double(), k=min(k,vocabulary))
    torch.testing.assert_close(loss, expected, atol=1e-14, rtol=1e-13)
    counts = graph._ranking_counts(teacher, student, selected)
    flags = topk_preservation_flags(teacher, student, k=min(k,vocabulary))
    keys = ('literal_argmax_changed','literal_order_changed','literal_membership_changed',
            'tie_aware_argmax_changed','strict_order_changed','strict_membership_changed','strict_topk_changed')
    assert counts.tolist() == [int(flags[name].sum()) for name in keys]


def test_graph_scope_guards_and_adapter_type_are_explicit():
    legacy = TokenwiseConfig(.01,.01,.1,.01)
    tied = TokenwiseConfig(.01,.01,.1,.01,ranking_k=100,ranking_tie_policy='exact_teacher_ties')
    with pytest.raises(ValueError, match='exact_teacher_ties'):
        TieAwareTokenwiseGraph(token_config=legacy)
    with pytest.raises(ValueError, match='legacy top10'):
        TokenwiseThinGradientGraph(token_config=tied)
    with pytest.raises(ValueError, match='exact cached FP64'):
        TieAwareTokenwiseGraph(token_config=tied, teacher_representation='logits')
    assert TieAwareTokenwiseGraphAccumulator.graph_type is TieAwareTokenwiseGraph


@pytest.mark.parametrize('vocabulary', [10, 137])
def test_adapter_publishes_measured_strict_counts_and_never_labels100_as10(monkeypatch, vocabulary):
    config=TokenwiseConfig(.01,.01,.1,.01,ranking_k=100,ranking_tie_policy='exact_teacher_ties')
    dual=TokenwiseDual([6],config=config,dual_lr=.01)
    adapter=object.__new__(TieAwareTokenwiseGraphAccumulator)
    adapter.graph=SimpleNamespace(inputs=torch.ones(2,3,dtype=torch.long),
        engine=SimpleNamespace(parameters={'proj.weight':torch.empty(vocabulary,2)}))
    monkeypatch.setattr(OfflineGraphAccumulator,'_metrics',lambda *args:({},{}))
    output=dict(token_kl=torch.arange(6,dtype=torch.float64)*.01,
        token_loss_components=torch.tensor([.1,.2,.3]),rank_change_counts=torch.tensor([1,5,4,1,3,2,4]),
        regularizer=torch.tensor(-.1))
    measured,stats=adapter._metrics(output,dual,6)
    assert measured['ranking_k']==min(100,vocabulary)
    assert measured['topk_order_changes']==5 and measured['topk_membership_changes']==4
    assert measured['strict_order_changes']==3 and measured['strict_membership_changes']==2
    assert measured['strict_topk_changes']==4
    assert ('top10_order_changes' in measured)==(vocabulary==10)
    assert stats['offline_backend']=='tokenwise_tie_graph'


def test_absolute_graph_rejects_relative_objective_before_allocating():
    config=TokenwiseConfig(.01,.01,.1,.01,ranking_k=100,ranking_tie_policy='exact_teacher_ties',
        ranking_loss_mode='relative_gap',loss_backend='tiled_relative_fp64_v1')
    with pytest.raises(ValueError,match='only for absolute ranking'):
        TieAwareTokenwiseGraph(token_config=config)
