from types import SimpleNamespace

import pytest
import torch

from rewriter.graph_tokenwise import TokenwiseThinGradientGraph
from rewriter.tokenwise_tiled import logit_terms_and_cotangent


@pytest.mark.parametrize('config', [
    SimpleNamespace(ranking_k=100, ranking_tie_policy='legacy'),
    SimpleNamespace(ranking_k=10, ranking_tie_policy='exact_teacher_ties'),
])
def test_old_tiled_and_graph_backends_reject_new_unqualified_ranking_scope(config):
    with pytest.raises(ValueError, match='legacy top10'):
        TokenwiseThinGradientGraph(token_config=config)
    with pytest.raises(ValueError, match='legacy top10'):
        logit_terms_and_cotangent(torch.empty(1, 137), torch.empty(1, 137, dtype=torch.float64),
                                 config=config, multiplier=0., panel_tokens=1)
