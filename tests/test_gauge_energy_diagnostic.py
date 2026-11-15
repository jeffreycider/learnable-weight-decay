import itertools
import numpy as np
import pytest
import torch
from rewriter.gauge_energy_diagnostic import normalized_energies,group_cost,analytic_reward_upper_bound,binary_coordinate_search

@pytest.mark.parametrize('p',[1,2])
@pytest.mark.parametrize('zero_bias',[False,True])
def test_bounds_enclose_exhaustive_discrete_optimum(p,zero_bias):
    rng=np.random.default_rng(8);a,b,c=rng.random((3,5))
    if zero_bias:b*=0
    e,_,active=normalized_energies(a,b,c)
    exact=max(100*(active.sum()-group_cost(e,x,p).sum())/15 for x in itertools.product((-1,0,1),repeat=5))
    result=binary_coordinate_search(a,b,c,inverse_power=p,max_sweeps=10)
    assert result['best']['mean_group_l2_percentage_reduction']<=exact+1e-10
    assert result['upper_bound']['mean_group_l2_percentage_reduction_upper_bound']>=exact-1e-10
    assert result['best']['mean_group_l2_percentage_reduction']>=-1e-10

@pytest.mark.parametrize('p',[1,2])
def test_exact_formula_matches_dense_weight_and_bias_edits(p):
    torch.manual_seed(5);first=torch.randn(5,3,dtype=torch.double);bias=torch.randn(5,dtype=torch.double);second=torch.randn(4,5,dtype=torch.double)
    energies=[first.square().sum(1).numpy(),bias.square().numpy(),second.square().sum(0).numpy()]
    result=binary_coordinate_search(*energies,inverse_power=p)
    d=torch.tensor(2.**np.asarray(result['channel_exponents']),dtype=torch.double)
    before=[first.norm(),bias.norm(),second.norm()]
    after=[(first*d[:,None]).norm(),(bias*d).norm(),(second/d[None,:]**p).norm()]
    direct=sum(100*(1-float(y/x))/15 for x,y in zip(before,after))
    assert direct==pytest.approx(result['best']['mean_group_l2_percentage_reduction'],abs=1e-12)
    if p==2:
        x=torch.randn(9,3,dtype=torch.double)
        old=(x@first.T+bias).relu().square()@second.T
        new=(x@(first*d[:,None]).T+bias*d).relu().square()@(second/d[None,:]**2).T
        torch.testing.assert_close(old,new,atol=0,rtol=0)


def test_uniform_holders_bounds_and_zero_groups():
    e=np.ones((3,10))/10
    assert analytic_reward_upper_bound(e,inverse_power=1)['normalized_cost_lower_bound']==pytest.approx(2*np.sqrt(2))
    assert analytic_reward_upper_bound(e,inverse_power=2)['normalized_cost_lower_bound']==pytest.approx(3.)
    result=binary_coordinate_search(np.zeros(2),np.zeros(2),np.zeros(2),inverse_power=2)
    assert result['best']['mean_group_l2_percentage_reduction']==0
