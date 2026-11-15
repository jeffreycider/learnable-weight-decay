"""Norm-only diagnostic matched to the mean functional-group relative L2 norm.

For each group, decompose W into its accessible projection w and an orthogonal
residual. At a fixed edit length, Cauchy--Schwarz makes the best direction -w.
An optimum therefore has d_g=-alpha_g w_g, with 0<=alpha_g<=1. Writing
N_g=||W_g|| and p_g=||w_g|| leaves the convex allocation problem

    min mean_g sqrt(N_g²-p_g²+p_g²(1-alpha_g)²)/N_g,
    subject to sum_g alpha_g² p_g² <= R².

This ignores function preservation and never replaces the learned controller.
The optimum is continuous and assumes exact orthogonal probes. Cached FP32
probes and parameter storage rounding make its realized edit only approximate;
neither its measured objective nor KL is a certified bound on learned edits.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .algebra import Anchors, Factors, RawAction


def solve_grouped_allocation(norms, projected_norms, radius, *, projection_rtol=1e-5):
    """Return FP64 alpha coefficients and serializable continuous diagnostics.

    Inputs are one-dimensional positive reference norms and nonnegative
    projected norms, with a finite nonnegative scalar radius. A small p>N from
    rounded probes is clipped and explicitly reported. Material excess fails.
    The reported dual is for the *mean* objective with constraint ||d||²-R².
    """
    def as_array(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().double().numpy()
        return np.asarray(value, dtype=np.float64)

    n, p = as_array(norms), as_array(projected_norms)
    r = float(radius)
    if n.ndim != 1 or not len(n) or n.shape != p.shape:
        raise ValueError('nonempty matching one-dimensional group norms required')
    if not np.isfinite(n).all() or not np.isfinite(p).all() or (n <= 0).any() or (p < 0).any():
        raise ValueError('positive finite reference norms and nonnegative finite projections required')
    if not math.isfinite(r) or r < 0 or not math.isfinite(projection_rtol) or projection_rtol < 0:
        raise ValueError('finite nonnegative radius and projection tolerance required')
    excess = np.maximum(p / n - 1, 0)
    if (excess > projection_rtol).any():
        raise ValueError('accessible projection exceeds reference group norm')
    p = np.minimum(p, n)
    # Rescale before squaring. This also makes the dual bracket insensitive to
    # whether callers express norms in tiny or large units.
    scale = max(float(n.max()), r)
    ns, ps, rs = n / scale, p / scale, r / scale
    accessible = ps > 0
    residual_sq = (ns - ps) * (ns + ps)
    full = accessible & (residual_sq == 0)

    def coefficients(tau):
        # tau=2*G*lambda*scale². For p=N the norm is N(1-alpha),
        # including its boundary kink: alpha=min(1,1/(tau*n²)).
        lo, hi = np.zeros_like(ns), np.ones_like(ns)
        for _ in range(64):
            mid = (lo + hi) / 2
            remaining = np.sqrt(residual_sq + ps * ps * (1 - mid)**2)
            positive = (1 - mid) > tau * mid * ns * remaining
            lo = np.where(positive, mid, lo)
            hi = np.where(positive, hi, mid)
        alpha = (lo + hi) / 2
        alpha[full] = np.minimum(1, 1 / (tau * ns[full]**2))
        alpha[~accessible] = 0
        return alpha

    total_projection = float(np.linalg.norm(ps))
    if r == 0 or not accessible.any():
        alpha = np.zeros_like(ns)
        tau = None if r == 0 and accessible.any() else 0.
    elif rs >= total_projection:
        alpha = accessible.astype(np.float64)
        tau = 0.
    else:
        low, high = 0., 1.
        while np.linalg.norm(coefficients(high) * ps) > rs:
            high *= 2
            if not math.isfinite(high):
                raise ValueError('dual bracket overflow: group norm scales too disparate')
        # Keep the feasible side of the bracket; changing lambda decreases all
        # edit lengths monotonically, including fully canceled vector groups.
        for _ in range(80):
            middle = (low + high) / 2
            if np.linalg.norm(coefficients(middle) * ps) > rs:
                low = middle
            else:
                high = middle
        tau = high
        alpha = coefficients(tau)
    remaining = np.sqrt(residual_sq + ps**2 * (1 - alpha)**2) / ns
    dual = None if tau is None else tau / scale / scale / (2 * len(n))
    stats = dict(
        radius=r, group_count=len(n), coefficients=alpha.tolist(),
        projected_weight_norm=float(np.linalg.norm(ps) * scale),
        ideal_edit_norm=float(np.linalg.norm(alpha * ps) * scale),
        ideal_group_relative_changes=(remaining - 1).tolist(),
        ideal_mean_relative_l2_change=float((remaining - 1).mean()),
        ideal_mean_group_l2_percentage_reduction=float(100 * (1 - remaining).mean()),
        dual_lambda=dual,
        radius_active=bool(accessible.any() and rs < total_projection),
        projection_norm_relative_excess=float(excess.max()),
        projection_norm_clipped=bool((excess > 0).any()))
    return torch.from_numpy(alpha.copy()), stats


@torch.no_grad()
def grouped_norm_only_action(snapshot, parameters, groups=None):
    """Construct the matched continuous action using only cached thin probes.

    ``parameters`` supplies reference group norms; matrices may be stored in
    their native orientation because Frobenius norms are transpose invariant.
    Snapshot.radius must already match the caller's selected editable scope.
    All coordinates outside ``groups`` receive exact zero actions.
    """
    from .grouped_norm import GroupedNormObjective

    objective = GroupedNormObjective(parameters, groups=groups)
    groups = objective.groups
    selected = [name for names in groups.values() for name in names]
    if len(selected) != len(set(selected)):
        raise ValueError('functional parameter groups must be disjoint')
    available = set(snapshot.matrices) | set(snapshot.vectors)
    if not set(selected).issubset(available):
        raise ValueError('grouped parameter missing from snapshot')
    matrices, vectors = {}, {}
    orthogonality_error = 0.
    for name, obs in snapshot.matrices.items():
        anchors = Anchors(obs.anchors.A.double(), obs.anchors.B.double())
        if name in selected:
            for basis in (anchors.A, anchors.B):
                error = (basis.T @ basis - torch.eye(basis.shape[1], device=basis.device,
                                                   dtype=basis.dtype)).abs().max()
                orthogonality_error = max(orthogonality_error, float(error))
            if orthogonality_error > 1e-5:
                raise ValueError('saved anchors are not orthonormal within the FP32 tolerance')
            matrices[name] = Factors.canonical(*(x.double() for x in obs.weight_probes), anchors)
        else:
            matrices[name] = Factors(torch.zeros_like(anchors.A), torch.zeros_like(anchors.B), anchors)
    for name, obs in snapshot.vectors.items():
        vectors[name] = obs.weight.double().clone() if name in selected else torch.zeros_like(obs.weight, dtype=torch.float64)
    energies = {name: factor.norm_sq() for name, factor in matrices.items()}
    energies.update({name: value.square().sum() for name, value in vectors.items()})
    reference = torch.stack([objective.reference_norms[g] for g in groups])
    projected = torch.stack([sum(energies[n] for n in names).sqrt() for names in groups.values()])
    alpha, stats = solve_grouped_allocation(reference, projected, snapshot.radius)
    for index, names in enumerate(groups.values()):
        for name in names:
            coefficient = float(alpha[index])
            if name in matrices:
                f = matrices[name]
                matrices[name] = Factors(-coefficient * f.P, -coefficient * f.Q, f.anchors)
            else:
                vectors[name] = -coefficient * vectors[name]
    action = RawAction(matrices, vectors)
    # Tolerance-level clipping of p>N otherwise risks a slightly oversized
    # action when alpha is applied to the original cached factors.
    raw_norm = float(action.norm_sq().sqrt())
    radius = float(snapshot.radius)
    realization_scale = min(1., radius / raw_norm) if raw_norm else 1.
    if realization_scale < 1:
        action = RawAction({n: Factors(realization_scale*f.P, realization_scale*f.Q, f.anchors)
                            for n, f in matrices.items()},
                           {n: realization_scale*v for n, v in vectors.items()})
    stats.update(group_names=list(groups), group_parameters={g: list(names) for g, names in groups.items()},
                 raw_edit_norm=float(action.norm_sq().sqrt()),
                 realization_feasibility_scale=realization_scale,
                 anchor_orthogonality_max_error=orthogonality_error,
                 interpretation='Mean functional-group relative L2 norm-only continuous optimum; '
                 'ignores preservation. Rounded storage, FP32 probes and tolerance-level projection '
                 'clipping do not constitute a certified bound.')
    return action, stats
