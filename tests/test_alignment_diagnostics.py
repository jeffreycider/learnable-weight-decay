import torch
from rewriter.algebra import Anchors
from rewriter.alignment_diagnostics import tensor_alignment,summarize_alignment


def test_alignment_matches_dense_projection_and_exposes_rounding():
    torch.manual_seed(15);w=torch.randn(7,5,dtype=torch.float64)
    a=Anchors.random(7,5,2,generator=torch.Generator().manual_seed(13),dtype=torch.float64)
    projected=w@a.B@a.B.T+a.A@(a.A.T@w)-a.A@(a.A.T@w@a.B)@a.B.T
    edited=w-.03*projected
    actual=summarize_alignment(tensor_alignment(w,edited,a,chunk_elements=8))
    assert abs(actual['cosine_to_negative_projected_weight']-1)<1e-12
    assert abs(actual['realized_off_subspace_fraction'])<1e-12
    noisy=edited.clone();noisy[0,0]+=.01
    result=summarize_alignment(tensor_alignment(w,noisy,a))
    expected=-((noisy-w)*projected).sum()/((noisy-w).norm()*projected.norm())
    assert abs(result['cosine_to_negative_projected_weight']-float(expected))<1e-12
    assert result['realized_off_subspace_fraction']>0
    assert abs(result['norm_squared_change']-float(noisy.square().sum()-w.square().sum()))<1e-12


def test_zero_and_vector_alignment():
    w=torch.zeros(4);result=summarize_alignment(tensor_alignment(w,w))
    assert result['cosine_to_negative_projected_weight'] is None
    v=torch.tensor([2.,-1.]);result=summarize_alignment(tensor_alignment(v,v*.9))
    assert abs(result['cosine_to_negative_projected_weight']-1)<1e-6
