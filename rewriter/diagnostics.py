"""Optional analytic control for the fixed-snapshot auxiliary quadratic.

This module is deliberately not imported by the trainer or online controller.
It supplies a diagnostic optimum for the *raw*, ungated surrogate, not a learned
policy or a replacement for online task credit. It never sees audit data and
never evaluates GPT.

With canonical coordinates x=(P,Q,dv), A^T P=0, the objective is
    2 b^T x + lambda ||x||^2 + alpha ||Zx||^2,
where b=Wcoords/c, lambda=1/c+gamma/(R^2+eps), and
alpha=beta/(K*loss_budget^2). Projecting row banks onto A-perp makes b and every
row of Z lie in the canonical subspace. A K-by-K Woodbury solve then suffices;
all other arrays have thin-bank dimensions. No dense GPT matrix, full coordinate
Hessian, or n-by-n projector is constructed.

For each n-by-m matrix the accessible dimension is r(n+m-r), and action rank is
at most 2r. The invisible (I-AA^T)W(I-BB^T) component cannot be changed. When
r=min(n,m), the action space covers the entire matrix space; smaller fixed ranks
generally cannot express isotropic decay. More solver iterations do not enlarge
that space. The returned action and diagnostics use FP64, including the small
solve, and remain detached. Any subsequent cast or safety contraction changes
the optimum and must be accounted for separately.
"""
from __future__ import annotations

import math
import torch
from .algebra import Anchors, Factors, RawAction


@torch.no_grad()
def solve_auxiliary_quadratic(snapshot, train, *, beta=1., gamma=.01,
                              loss_budget=1e-3, squared_norm_floor=1e-24,
                              orthogonality_tolerance=1e-5):
    """Return ``(RawAction, diagnostics)`` for the unconstrained raw surrogate.

    ``snapshot`` is a ControllerSnapshot and ``train`` contains only training
    sign probes, with the same names and a positive common probe count. Audit
    banks, the controller, GPT parameters and optimizer state are not arguments.
    Inputs are read-only. The solve is exact in real arithmetic; residuals expose
    floating-point error. A failed small SPD solve raises rather than adding an
    undeclared ridge or changing the objective.
    """
    for name, value, strictly_positive in (
            ('beta', beta, False), ('gamma', gamma, False),
            ('loss_budget', loss_budget, True), ('squared_norm_floor', squared_norm_floor, True),
            ('orthogonality_tolerance', orthogonality_tolerance, True)):
        if not math.isfinite(float(value)) or (value <= 0 if strictly_positive else value < 0):
            raise ValueError(f'{name} must be finite and {"positive" if strictly_positive else "nonnegative"}')
    if set(snapshot.matrices) != set(train.matrices) or set(snapshot.vectors) != set(train.vectors):
        raise ValueError('training banks must cover exactly the editable snapshot parameters')
    if not snapshot.matrices and not snapshot.vectors:
        raise ValueError('at least one editable parameter is required')
    reference = snapshot.weight_norm
    device = reference.device
    weight_norm = reference.detach().to(dtype=torch.float64)
    radius = snapshot.radius.detach().to(device=device, dtype=torch.float64)
    if weight_norm.numel() != 1 or radius.numel() != 1:
        raise ValueError('weight_norm and radius must be scalars')
    if not torch.isfinite(weight_norm) or not torch.isfinite(radius) or weight_norm < 0 or radius < 0:
        raise ValueError('weight_norm and radius must be finite and nonnegative')
    c = 2 * weight_norm * radius + radius.square() + squared_norm_floor
    lam = c.reciprocal() + gamma / (radius.square() + squared_norm_floor)
    if not torch.isfinite(c) or not torch.isfinite(lam) or c <= 0 or lam <= 0:
        raise ValueError('nonfinite or nonpositive quadratic normalization; rescale diagnostic inputs')
    # Blocks are (kind, name, weight coordinates, signed-gradient coordinates).
    blocks, converted_anchors = [], {}
    probes = None
    accessible = 0

    def convert(value):
        if value.device != device:
            raise ValueError('all snapshot and measurement tensors must share one device')
        result = value.detach().to(dtype=torch.float64)
        if not torch.isfinite(result).all():
            raise ValueError('nonfinite cached coordinate or measurement')
        return result

    def append(kind, name, weight, measured):
        nonlocal probes
        if measured.ndim != weight.ndim + 1 or measured.shape[1:] != weight.shape:
            raise ValueError(f'signed-gradient shape mismatch for {name}/{kind}')
        if measured.shape[0] < 1:
            raise ValueError('at least one training sign probe is required')
        if probes is None:
            probes = measured.shape[0]
        elif probes != measured.shape[0]:
            raise ValueError('all training banks must have the same probe count')
        blocks.append((kind, name, weight, measured))

    for name, obs in snapshot.matrices.items():
        a, anchor_b = convert(obs.anchors.A), convert(obs.anchors.B)
        if a.ndim != 2 or anchor_b.ndim != 2 or a.shape[1] != anchor_b.shape[1]:
            raise ValueError(f'invalid two-sided anchors for {name}')
        rank = a.shape[1]
        if rank < 1 or rank > min(a.shape[0], anchor_b.shape[0]):
            raise ValueError(f'invalid anchor rank for {name}')
        identity = torch.eye(rank, dtype=torch.float64, device=device)
        if max(float((a.T @ a - identity).abs().max()),
               float((anchor_b.T @ anchor_b - identity).abs().max())) > orthogonality_tolerance:
            raise ValueError(f'orthonormal anchors required for {name}')
        wp, wq = (convert(x) for x in obs.weight_probes)
        zp, zq = (convert(x) for x in train.matrices[name])
        if wp.shape != a.shape or wq.shape != anchor_b.shape:
            raise ValueError(f'weight-probe shape mismatch for {name}')
        if zp.ndim != 3 or zp.shape[1:] != a.shape:
            raise ValueError(f'signed row-probe shape mismatch for {name}')
        # Thin canonical projection, including all K signed row banks.
        wp = wp - a @ (a.T @ wp)
        zp = zp - a @ (a.T @ zp)
        append('P', name, wp, zp)
        append('Q', name, wq, zq)
        converted_anchors[name] = Anchors(a, anchor_b)
        accessible += rank * (a.shape[0] + anchor_b.shape[0] - rank)
    for name, obs in snapshot.vectors.items():
        w, z = convert(obs.weight), convert(train.vectors[name])
        if w.ndim != 1:
            raise ValueError(f'editable vector must be one-dimensional: {name}')
        append('vector', name, w, z)
        accessible += w.numel()
    alpha = beta / (probes * loss_budget**2)
    gram = torch.zeros(probes, probes, dtype=torch.float64, device=device)
    zb = torch.zeros(probes, dtype=torch.float64, device=device)
    for _, _, w, z in blocks:
        flat = z.reshape(probes, -1)
        gram.add_(flat @ flat.T)
        zb.add_(flat @ (w / c).reshape(-1))
    identity = torch.eye(probes, dtype=torch.float64, device=device)
    small = lam * identity + alpha * gram
    if not torch.isfinite(small).all() or not torch.isfinite(zb).all():
        raise ValueError('nonfinite small-system coefficients; rescale the diagnostic inputs')
    if alpha:
        cholesky, info = torch.linalg.cholesky_ex(small)
        if int(info) != 0:
            raise RuntimeError('FP64 Woodbury system lost positive definiteness; no implicit regularization applied')
        y = torch.cholesky_solve(zb[:, None], cholesky).squeeze(-1)
    else:
        y = zb / lam
    solutions = []
    for kind, name, w, z in blocks:
        correction = (z.reshape(probes, -1).T @ y).reshape(w.shape)
        x = (-w / c + alpha * correction) / lam
        if not torch.isfinite(x).all():
            raise RuntimeError('nonfinite FP64 solver action; no approximate fallback applied')
        solutions.append((kind, name, x))
    by_matrix, vectors = {}, {}
    for kind, name, x in solutions:
        if kind == 'vector':
            vectors[name] = x
        else:
            by_matrix.setdefault(name, {})[kind] = x
    action = RawAction({n: Factors(v['P'], v['Q'], converted_anchors[n]) for n,v in by_matrix.items()}, vectors)
    # Diagnose the same FP64 quadratic without materializing a concatenated Z.
    z_action = torch.zeros_like(zb)
    linear, norm_sq, b_norm_sq = (torch.zeros((), dtype=torch.float64, device=device) for _ in range(3))
    for (_, _, w, z), (_, _, x) in zip(blocks, solutions):
        z_action.add_(z.reshape(probes, -1) @ x.reshape(-1))
        linear += (w * x).sum()
        norm_sq += x.square().sum()
        b_norm_sq += (w / c).square().sum()
    residual_sq = torch.zeros_like(norm_sq)
    for (_, _, w, z), (_, _, x) in zip(blocks, solutions):
        residual = lam * x + w / c + alpha * (z.reshape(probes, -1).T @ z_action).reshape(x.shape)
        residual_sq += residual.square().sum()
    proxy = z_action.square().mean()
    norm_change = 2 * linear + norm_sq
    objective = norm_change / c + beta * proxy / loss_budget**2 + gamma * norm_sq / (radius.square() + squared_norm_floor)
    tiny = torch.finfo(torch.float64).tiny
    diagnostics = dict(diagnostic_only=True, extra_gpt_evaluations=0,
        dtype='float64', training_probes=probes, lambda_coefficient=float(lam), alpha_coefficient=float(alpha),
        normalization_c=float(c), objective=float(objective), raw_norm_change=float(norm_change),
        action_norm_sq=float(norm_sq), train_proxy=float(proxy),
        stationarity_relative_norm=float(residual_sq.sqrt() / b_norm_sq.sqrt().clamp_min(tiny)),
        small_system_relative_residual=float((small @ y-zb).norm() / zb.norm().clamp_min(tiny)),
        canonical_accessible_dimension=accessible,
        stored_action_coordinates=sum(x.numel() for _,_,x in solutions),
        max_anchor_constraint_error=max((float((f.anchors.A.T @ f.P).abs().max())
                                          for f in action.matrices.values()), default=0.))
    return action, diagnostics
