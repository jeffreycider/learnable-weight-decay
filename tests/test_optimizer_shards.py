import copy
import pytest
import torch
from rewriter.optimizer_shards import merge_muon_shards


def fixture():
    groups=[dict(params=list(range(7)),lr=.02,mu=.95,weight_decay=.05)]
    return [dict(param_groups=copy.deepcopy(groups),state={i:dict(momentum=torch.full((2,3),float(i))) for i in range(rank,7,3)}) for rank in range(3)]


def test_merge_retains_actual_owned_moments_without_reset_or_input_mutation():
    shards=fixture();merged=merge_muon_shards(shards)
    assert set(merged['state'])==set(range(7))
    for i,state in merged['state'].items():assert state['momentum'] is shards[i%3]['state'][i]['momentum']
    merged['param_groups'][0]['lr']=.5
    assert shards[0]['param_groups'][0]['lr']==.02


@pytest.mark.parametrize('fault',['missing','wrong_owner','settings'])
def test_merge_rejects_incomplete_or_inconsistent_rank_states(fault):
    shards=fixture()
    if fault=='missing':del shards[1]['state'][1]
    if fault=='wrong_owner':shards[0]['state'][1]=shards[1]['state'].pop(1)
    if fault=='settings':shards[2]['param_groups'][0]['mu']=.9
    with pytest.raises(ValueError):merge_muon_shards(shards)
