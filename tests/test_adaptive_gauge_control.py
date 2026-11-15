import pytest
import torch

from rewriter.adaptive_gauge_control import orthonormal_product


@pytest.mark.parametrize('rank',[2,5])
@pytest.mark.parametrize('degenerate',[False,True])
def test_thin_qr_representation_including_completion_and_rank_deficiency(rank,degenerate):
    torch.set_num_threads(1);generator=torch.Generator().manual_seed(43)
    x=torch.randn(9,2,generator=generator,dtype=torch.float64)
    y=torch.randn(7,2,generator=generator,dtype=torch.float64)
    if degenerate:x[:,1]=0;y[:,0]=0
    rng=torch.get_rng_state().clone()
    f=orthonormal_product(x,y,rank=rank)
    assert torch.equal(rng,torch.get_rng_state())
    torch.testing.assert_close(f.U.T@f.U,torch.eye(rank,dtype=x.dtype),atol=1e-14,rtol=1e-14)
    torch.testing.assert_close(f.V.T@f.V,torch.eye(rank,dtype=x.dtype),atol=1e-14,rtol=1e-14)
    torch.testing.assert_close(f.dense(),x@y.T,atol=1e-14,rtol=1e-14)
    torch.testing.assert_close(f.norm_sq(),(x@y.T).square().sum(),atol=1e-12,rtol=1e-12)


@pytest.mark.parametrize('power',[1,2])
def test_paired_channel_gauge_and_bias_are_in_rank_cap_family(power):
    generator=torch.Generator().manual_seed(44)
    first=torch.randn(12,9,generator=generator,dtype=torch.float64)
    second=torch.randn(7,12,generator=generator,dtype=torch.float64)
    bias=torch.randn(12,generator=generator,dtype=torch.float64)
    indices=torch.tensor([1,7]);scales=torch.tensor([.5,2.],dtype=torch.float64)
    e=torch.zeros(12,2,dtype=torch.float64);e[indices,torch.arange(2)]=1
    up=orthonormal_product(e*(scales-1),first[indices].T,rank=5)
    down=orthonormal_product(second[:,indices]*(scales.pow(-power)-1),e,rank=5)
    direct_first=first.clone();direct_first[indices]*=scales[:,None]
    direct_second=second.clone();direct_second[:,indices]*=scales.pow(-power)[None,:]
    torch.testing.assert_close(first+up.dense(),direct_first,atol=1e-14,rtol=1e-14)
    torch.testing.assert_close(second+down.dense(),direct_second,atol=1e-14,rtol=1e-14)
    # Existing private vector decoder can use a coordinate frame for the same
    # selected indices; normalize away its explicit sqrt(n/L) decoder scale.
    q=torch.linalg.qr(torch.cat((e,torch.eye(12,dtype=e.dtype)[:,:3]),dim=1),mode='reduced').Q
    delta=e@((scales-1)*bias[indices])
    coefficients=q.T@delta/(12/5)**.5
    torch.testing.assert_close((12/5)**.5*q@coefficients,delta,atol=1e-14,rtol=1e-14)


def test_invalid_rank_rejected():
    with pytest.raises(ValueError,match='rank'):
        orthonormal_product(torch.ones(4,3),torch.ones(5,3),rank=2)
