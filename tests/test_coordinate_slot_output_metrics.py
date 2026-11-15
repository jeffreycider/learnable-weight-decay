import torch
from scripts.qualify_coordinate_slot_compile import compare_outputs


def test_mutual_metrics_use_effective10_and_zero_tied_identity():
    x=torch.zeros(4,10)
    row=compare_outputs(x,x)
    assert row['ranking_k']==10 and row['contexts']==4
    assert row['mean_forward_kl']==row['mean_reverse_kl']==0
    assert not any(row['rank_counts'].values())


def test_top100_rank_witness_is_reported_even_when_functional_difference_is_tiny():
    x=(1.-torch.arange(137)*1e-6)[None]
    y=x.clone();y[:,101]=x[:,99]+1e-7
    row=compare_outputs(x,y)
    assert row['ranking_k']==100
    assert row['rank_counts']['strict_membership_changed']==1
    assert row['rank_counts']['literal_membership_changed']==1
    assert row['max_forward_kl']<1e-10 and row['centered_logit_rms']>0
