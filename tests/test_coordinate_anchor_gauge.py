import numpy as np
import pytest
import torch
from rewriter.coordinate_anchor_gauge import capped_binary_greedy
from rewriter.algebra import Anchors,Factors


def test_matrix_progress_filter_rejects_bias_only_reward():
    # Shrinking the first channel wins on bias yet grows the combined matrices.
    energies=([1.,1.],[100.,0.],[1.,1.])
    ordinary=capped_binary_greedy(*energies,['a','a'],inverse_power=1,rank=1)
    constrained=capped_binary_greedy(*energies,['a','a'],inverse_power=1,rank=1,
                                    require_matrix_improvement=True)
    assert ordinary['selected_channels']==1
    assert sum(ordinary['group_l2_percentage_reductions'][i] for i in (0,2))<0
    assert constrained['selected_channels']==0


def test_matrix_progress_target_improves_requested_reward_and_matrix_contribution():
    energies=([100.,1.],[1.,1.],[1.,100.])
    result=capped_binary_greedy(*energies,['a','a'],inverse_power=1,rank=2,
                                require_matrix_improvement=True)
    assert result['selected_channels']>0
    assert result['mean_group_l2_percentage_reduction']>0
    assert sum(result['group_l2_percentage_reductions'][i] for i in (0,2))>0
    for a,b in zip(result['moves'],result['moves'][1:]):
        assert b['matrix_contribution_percent']>a['matrix_contribution_percent']
        assert b['mean_group_reduction_percent']>a['mean_group_reduction_percent']
    # Existing controls retain their exact default behavior.
    assert capped_binary_greedy(*energies,['a','a'],inverse_power=1,rank=2)==capped_binary_greedy(
        *energies,['a','a'],inverse_power=1,rank=2,require_matrix_improvement=False)


def test_capacity_and_exact_energy_contraction():
    g=np.random.default_rng(51);energies=[g.lognormal(0,1,21) for _ in range(3)];blocks=['a']*7+['b']*7+['c']*7
    result=capped_binary_greedy(*energies,blocks,inverse_power=2,rank=2)
    assert all(n<=2 for n in result['selected_per_block'].values())
    assert result['mean_group_l2_percentage_reduction']>=-1e-12
    d=np.exp2(result['channel_exponents']);cost=sum(np.sqrt(np.dot(e,d**power)/sum(e)) for e,power in zip(energies,(2,2,-4)))
    assert result['mean_group_l2_percentage_reduction']==pytest.approx((3-cost)*100/15)
    assert all(b['mean_group_reduction_percent']>a['mean_group_reduction_percent'] for a,b in zip(result['moves'],result['moves'][1:]))


@pytest.mark.parametrize('power',[1,2])
def test_selected_coordinate_anchors_exactly_represent_both_gauge_matrices(power):
    torch.manual_seed(9);n,m,o,r=7,5,6,2;dtype=torch.float64
    first=torch.randn(n,m,dtype=dtype);bias=torch.randn(n,dtype=dtype);second=torch.randn(o,n,dtype=dtype)
    ids=torch.tensor([1,5]);E=torch.eye(n,dtype=dtype)[:,ids];d=torch.tensor([.5,2.],dtype=dtype)
    Bfirst=torch.linalg.qr(torch.randn(m,r,dtype=dtype)).Q;Asecond=torch.linalg.qr(torch.randn(o,r,dtype=dtype)).Q
    first_anchor=Anchors(E,Bfirst);second_anchor=Anchors(Asecond,E)
    q=first.T@E@torch.diag(d-1)
    first_action=Factors.canonical(torch.zeros(n,r,dtype=dtype),q,first_anchor)
    rawp=second@E@torch.diag(d.pow(-power)-1)
    # canonical() projects P only; compensate the removed component in Q.
    compensated_q=E@(Asecond.T@rawp).T
    second_action=Factors.canonical(rawp,compensated_q,second_anchor)
    transform=torch.ones(n,dtype=dtype);transform[ids]=d
    expected_first=transform[:,None]*first-first;expected_second=second*transform.pow(-power)[None,:]-second
    torch.testing.assert_close(first_action.dense(),expected_first,atol=2e-15,rtol=2e-15)
    torch.testing.assert_close(second_action.dense(),expected_second,atol=3e-15,rtol=3e-15)
    torch.testing.assert_close(Asecond.T@second_action.P,torch.zeros(r,r,dtype=dtype),atol=3e-15,rtol=0)
    x=torch.randn(m,11,dtype=dtype)
    activation=lambda z:z if power==1 else z.relu().square()
    before=second@activation(first@x+bias[:,None])
    after=(second+second_action.dense())@activation((first+first_action.dense())@x+(bias*transform)[:,None])
    torch.testing.assert_close(after,before,atol=2e-13,rtol=2e-13)


def test_zero_reference_groups_are_not_given_fake_denominators():
    r=capped_binary_greedy(np.zeros(4),np.zeros(4),np.ones(4),['a']*4,inverse_power=1,rank=2)
    assert r['selected_channels']==2
    assert r['group_l2_percentage_reductions'][:2]==[0.,0.]
