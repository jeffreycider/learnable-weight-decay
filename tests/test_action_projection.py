import torch

from rewriter.algebra import Anchors, Factors


def test_projection_is_nearest_orthogonal_and_idempotent():
    gen = torch.Generator().manual_seed(47)
    anchors = Anchors.random(13, 9, 3, generator=gen, dtype=torch.float64)
    edit = torch.randn(13, 9, generator=gen, dtype=torch.float64)
    projected = Factors.project_dense(edit, anchors).dense()
    residual = edit - projected
    torch.testing.assert_close(residual @ anchors.B, torch.zeros(13, 3, dtype=edit.dtype), atol=1e-14, rtol=0)
    torch.testing.assert_close(residual.T @ anchors.A, torch.zeros(9, 3, dtype=edit.dtype), atol=1e-14, rtol=0)
    torch.testing.assert_close(Factors.project_dense(projected, anchors).dense(), projected)
    for _ in range(10):
        other = Factors.canonical(torch.randn(13, 3, generator=gen, dtype=edit.dtype),
                                  torch.randn(9, 3, generator=gen, dtype=edit.dtype), anchors).dense()
        torch.testing.assert_close((residual * other).sum(), torch.tensor(0., dtype=edit.dtype), atol=1e-13, rtol=0)
        assert (edit - other).square().sum() >= residual.square().sum()
    torch.testing.assert_close(edit.square().sum(), projected.square().sum() + residual.square().sum())


def test_full_rank_anchors_recover_dense_edit():
    gen = torch.Generator().manual_seed(48)
    anchors = Anchors.random(5, 8, 5, generator=gen, dtype=torch.float64)
    edit = torch.randn(5, 8, generator=gen, dtype=torch.float64)
    torch.testing.assert_close(Factors.project_dense(edit, anchors).dense(), edit)
