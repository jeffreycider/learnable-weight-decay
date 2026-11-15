"""Fixed-checkpoint free-factor control, never a deployable Chisel policy.

This asks whether the anchored action space admits a better finite-KL edit than
the learned conditional network found. It contains no observation encoder and
cannot generalize to another GPT checkpoint.
"""
import torch
from torch import nn

from .algebra import Factors, RawAction


class DiagnosticAnchoredAction(nn.Module):
    architecture = "diagnostic_fixed_checkpoint_factors_v1"

    def __init__(self, snapshot):
        super().__init__()
        self.matrix_names = tuple(snapshot.matrices)
        self.vector_names = tuple(snapshot.vectors)
        self.anchors = tuple(o.anchors for o in snapshot.matrices.values())
        self.rows = nn.ParameterList([
            nn.Parameter(torch.zeros_like(a.A, dtype=torch.float32)) for a in self.anchors])
        self.columns = nn.ParameterList([
            nn.Parameter(torch.zeros_like(a.B, dtype=torch.float32)) for a in self.anchors])
        self.vectors = nn.ParameterList([
            nn.Parameter(torch.zeros_like(o.weight, dtype=torch.float32))
            for o in snapshot.vectors.values()])

    def forward(self, snapshot):
        if tuple(snapshot.matrices) != self.matrix_names or tuple(snapshot.vectors) != self.vector_names:
            raise ValueError("diagnostic factors are bound to one snapshot schema")
        if any(snapshot.matrices[n].anchors is not a
               for n, a in zip(self.matrix_names, self.anchors)):
            raise ValueError("diagnostic factors cannot be transferred to another snapshot")
        return RawAction(
            {n: Factors.canonical(p, q, a) for n, p, q, a in
             zip(self.matrix_names, self.rows, self.columns, self.anchors)},
            dict(zip(self.vector_names, self.vectors)))
