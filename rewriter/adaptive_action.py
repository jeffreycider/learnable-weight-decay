"""Experimental adaptive rank-L edits; no fixed-anchor contraction identities.

U and V may be column-orthonormal or row-orthonormal Parseval frames. The
general Gram formulas below also support unconstrained factors and therefore
retain the correct ambient derivatives before a differentiable QR chain.
Dense cotangents must come from an actual permitted finite-function evaluation
or other explicitly supplied dense quantity. Old (G B, G.T A) sensor sketches
are NOT sufficient to evaluate these contractions for moving U/V.
"""
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor


@dataclass(frozen=True)
class AdaptiveCotangents:
    U: Tensor
    C: Tensor
    V: Tensor


@dataclass(frozen=True)
class AdaptiveFactors:
    U: Tensor
    C: Tensor
    V: Tensor
    family: str = 'adaptive_ucv_v1'

    def __post_init__(self):
        if self.family != 'adaptive_ucv_v1':
            raise ValueError('adaptive action family mismatch')
        if any(not isinstance(x, Tensor) or x.ndim != 2 for x in (self.U, self.C, self.V)):
            raise ValueError('U, C, V must be matrices')
        if (self.C.shape != (self.U.shape[1], self.V.shape[1])
                or min(*self.U.shape, *self.V.shape) < 1):
            raise ValueError('adaptive factor shapes do not match')
        if any(x.dtype != self.U.dtype or x.device != self.U.device for x in (self.C, self.V)):
            raise ValueError('adaptive factor dtype/device mismatch')

    def dense(self):
        return self.U @ self.C @ self.V.T

    def norm_sq(self):
        # Do not simplify to ||C||²: wide frames have U U.T=I rather than
        # U.T U=I. The general formula also preserves gradients with respect
        # to unconstrained U/V, before QR's VJP is applied.
        return (self.C * ((self.U.T @ self.U) @ self.C @ (self.V.T @ self.V))).sum()

    def _validate_dense(self, matrix):
        if not isinstance(matrix, Tensor) or matrix.shape[-2:] != (self.U.shape[0], self.V.shape[0]):
            raise ValueError('a dense matrix cotangent is required; fixed-anchor sketches are insufficient')
        if matrix.dtype != self.U.dtype or matrix.device != self.U.device:
            raise ValueError('dense contraction dtype/device mismatch')

    def contract(self, matrix):
        """<matrix, U C V.T>; leading batch dimensions are preserved.

        Recompute U.T @ matrix @ V with LIVE frames. Caching that projection
        while U/V learn would silently remove their derivative paths.
        """
        self._validate_dense(matrix)
        return ((self.U.T @ matrix @ self.V) * self.C).sum(dim=(-2, -1))

    def cotangents(self, matrix):
        """Exact ambient VJP from dL/d(dense edit), before QR/scaling chains.

        Batched inputs return one cotangent per batch item; callers choose the
        explicit SUM/mean normalization. This method never silently averages.
        """
        self._validate_dense(matrix)
        return AdaptiveCotangents(matrix @ self.V @ self.C.T,
                                  self.U.T @ matrix @ self.V,
                                  matrix.transpose(-2, -1) @ self.U @ self.C)

    def norm_change(self, weight):
        return 2 * self.contract(weight) + self.norm_sq()

    def scaled(self, scale):
        """Put any differentiable global guard or group scale in the core."""
        return AdaptiveFactors(self.U, self.C * scale, self.V)


@dataclass(frozen=True)
class AdaptiveAction:
    matrices: Mapping[str, AdaptiveFactors]
    vectors: Mapping[str, Tensor]
    family: str = 'adaptive_ucv_v1'

    def __post_init__(self):
        if self.family != 'adaptive_ucv_v1' or any(not isinstance(v, AdaptiveFactors) for v in self.matrices.values()):
            raise ValueError('adaptive action family mismatch')
        if not self.matrices and not self.vectors:
            raise ValueError('empty adaptive action')
        if set(self.matrices) & set(self.vectors):
            raise ValueError('matrix/vector names overlap')

    def dense(self):
        return {**{name: value.dense() for name, value in self.matrices.items()}, **self.vectors}

    def norm_sq(self):
        return sum([value.norm_sq() for value in self.matrices.values()]
                   + [value.square().sum() for value in self.vectors.values()])

    def contraction(self, matrices, vectors):
        if matrices.keys() != self.matrices.keys() or vectors.keys() != self.vectors.keys():
            raise ValueError('adaptive dense contraction keys differ')
        return sum([value.contract(matrices[name]) for name, value in self.matrices.items()]
                   + [(value * vectors[name]).sum(-1) for name, value in self.vectors.items()])

    def norm_change(self, matrices, vectors):
        return 2 * self.contraction(matrices, vectors) + self.norm_sq()

    def scaled(self, scale):
        return AdaptiveAction({n: v.scaled(scale) for n, v in self.matrices.items()},
                              {n: v * scale for n, v in self.vectors.items()})
