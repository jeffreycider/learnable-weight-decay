"""Exact thin algebra for PB^T + AQ^T; never build a dense projector."""
from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass(frozen=True)
class Anchors:
    A: Tensor
    B: Tensor

    @classmethod
    def random(cls, n, m, rank, *, generator, device='cpu', dtype=torch.float32):
        if rank > min(n, m):
            raise ValueError('probe rank exceeds matrix dimensions')
        # CPU stream is independent of the model RNG and portable across devices.
        def basis(dim):
            x = torch.randn(dim, rank, generator=generator, dtype=torch.float64)
            return torch.linalg.qr(x, mode='reduced').Q.to(device=device, dtype=dtype)
        return cls(basis(n), basis(m))


@dataclass(frozen=True)
class Factors:
    P: Tensor
    Q: Tensor
    anchors: Anchors
    family: str = 'anchored_pb_aq'

    def __post_init__(self):
        if self.family != 'anchored_pb_aq':
            raise ValueError('free PQ^T is not supported by the cached identities')
        if self.P.shape != self.anchors.A.shape or self.Q.shape != self.anchors.B.shape:
            raise ValueError('factor shape must match action-time anchors')

    @classmethod
    def canonical(cls, p_tilde, q, anchors):
        p = p_tilde - anchors.A @ (anchors.A.T @ p_tilde)
        return cls(p, q, anchors)

    @classmethod
    def project_dense(cls, edit, anchors):
        """Orthogonal Frobenius projection into the anchored action space.

        Diagnostic use only: reads the supplied dense edit once per probe bank.
        With orthonormal anchors, the residual is orthogonal to every accessible
        action. Projection does NOT preserve a dense edit's functional identity.
        No square projector is constructed.
        """
        if edit.shape != (anchors.A.shape[0], anchors.B.shape[0]):
            raise ValueError('dense edit shape does not match anchor coordinates')
        return cls.canonical(edit @ anchors.B, edit.T @ anchors.A, anchors)

    def dense(self):
        """Only for committing and small tests, not auxiliary optimization."""
        return self.P @ self.anchors.B.T + self.anchors.A @ self.Q.T

    def norm_sq(self):
        return self.P.square().sum() + self.Q.square().sum()

    def contract(self, xb, xta):
        return (xb * self.P).sum(dim=(-2, -1)) + (xta * self.Q).sum(dim=(-2, -1))


@dataclass(frozen=True)
class RawAction:
    matrices: dict[str, Factors]
    vectors: dict[str, Tensor]

    def norm_sq(self):
        return sum([f.norm_sq() for f in self.matrices.values()] +
                   [v.square().sum() for v in self.vectors.values()])

    def contraction(self, matrices, vectors):
        return sum([f.contract(*matrices[n]) for n, f in self.matrices.items()] +
                   [(vectors[n] * v).sum(-1) for n, v in self.vectors.items()])

    def norm_change(self, weights, vectors):
        return 2 * self.contraction(weights, vectors) + self.norm_sq()


@dataclass(frozen=True)
class Measurements:
    """Separate train/audit storage. First dimension indexes sequence signs."""
    matrices: dict[str, tuple[Tensor, Tensor]]
    vectors: dict[str, Tensor]

    def values(self, action):
        # Whole-model sum BEFORE square preserves cross-layer cancellation.
        return action.contraction(self.matrices, self.vectors)

    def penalty(self, action):
        return self.values(action).square().mean()


def guard(action, audit, radius, loss_budget, warmup=1., *, norm_floor=1e-12,
          loss_floor=1e-12, return_measurements=False):
    """One differentiable global multiplier; stable derivative at zero action.

    norm_floor has parameter-norm units; loss_floor has nats/token units.
    Clamp below sqrt avoids 0 * inf NaNs from sqrt's derivative at zero.
    """
    norm = action.norm_sq().clamp_min(norm_floor**2).sqrt()
    audit_value = audit.penalty(action)
    damage = audit_value.clamp_min(loss_floor**2).sqrt()
    one = norm.new_ones(())
    kappa = warmup * torch.minimum(one, torch.minimum(radius / (norm + norm_floor),
                                                     loss_budget / (damage + loss_floor)))
    return (kappa, audit_value) if return_measurements else kappa


def auxiliary_objective(action, weights, vectors, train, *, weight_norm, radius,
                        loss_budget=1e-3, beta=1., gamma=.01, squared_norm_floor=1e-24):
    c = 2 * weight_norm * radius + radius.square() + squared_norm_floor
    return (action.norm_change(weights, vectors) / c +
            beta * train.penalty(action) / loss_budget**2 +
            gamma * action.norm_sq() / (radius.square() + squared_norm_floor))
