"""Independent CPU qualification of the optional action-time norm auxiliary (Fable-owned).

Checks, on the production fixture (15 PR323 groups, embedding transpose, a zero
reference group, QR anchors, canonical factors): (1) the thin probe/Gram
finite-edit objective J_norm = mean_g(||W_g + s D_g|| / R_g - 1) equals a dense
FP64 oracle in value and in autograd gradient w.r.t. the factors, vectors and the
differentiable global radius scale; (2) frozen zero-reference groups contribute
zero and receive no gradient; (3) the sign of descent (outward edits raise J,
inward edits lower it; reported shrink percent is -100 J); (4) the per-group gap
between the continuous objective and the FP32-stored realized metric at a
production-like per-element edit scale, plus the sub-ulp boundary where the
stored metric is exactly zero; (5) two Adam ticks with a shared state are not one
combined step (root's chosen semantics) and the step counter advances by two.
No GPT, no actor training, no scalar search.
"""
import math
import pytest
import torch
from rewriter.algebra import Factors, RawAction
from rewriter.grouped_norm import GroupedNormObjective
from test_transformer_action import fixture


def thin_and_dense(parameters, snapshot, grouped, factors, vectors, radius):
    """Return (J_thin, J_dense, per-group thin, per-group dense, scale s)."""
    raw_sq = sum(f.norm_sq() for f in factors.values()) + sum(v.square().sum() for v in vectors.values())
    s = torch.minimum(torch.ones((), dtype=torch.float64), radius / raw_sq.clamp_min(1e-24).sqrt())
    thin = {}; dense = {}
    for group, names in grouped.groups.items():
        R2 = grouped.reference_squared[group].double()
        d_thin = torch.zeros((), dtype=torch.float64); new_sq = torch.zeros((), dtype=torch.float64)
        for name in names:
            W = parameters[name].double()
            if name in factors:
                f = factors[name]; a = f.anchors
                Wc = W.T if name == 'embed.weight' else W            # computational orientation
                wb, wta = snapshot.matrices[name].weight_probes
                c = (wb.double() * f.P).sum() + (wta.double() * f.Q).sum()
                q = torch.trace(f.P.T @ f.P @ (a.B.T @ a.B).double()) + torch.trace(f.Q.T @ f.Q @ (a.A.T @ a.A).double()) \
                    + 2 * torch.trace(f.P.T @ a.A.double() @ f.Q.T @ a.B.double())
                d_thin = d_thin + 2 * s * c + s * s * q
                D = f.P @ a.B.T.double() + a.A.double() @ f.Q.T
                new_sq = new_sq + (Wc + s * D).square().sum()
            elif name in vectors:
                d = vectors[name]
                d_thin = d_thin + 2 * s * (W * d).sum() + s * s * d.square().sum()
                new_sq = new_sq + (W + s * d).square().sum()
            else:
                new_sq = new_sq + W.square().sum()
        if group in grouped.zero_reference_groups:
            thin[group] = torch.zeros((), dtype=torch.float64); dense[group] = torch.zeros((), dtype=torch.float64)
        else:
            thin[group] = (R2 + d_thin).sqrt() / R2.sqrt() - 1
            dense[group] = new_sq.sqrt() / R2.sqrt() - 1
    J_thin = torch.stack(list(thin.values())).mean(); J_dense = torch.stack(list(dense.values())).mean()
    return J_thin, J_dense, thin, dense, s


def build(parameters, snapshot, seed, edit_scale, radius_fraction):
    g = torch.Generator().manual_seed(seed)
    grouped = GroupedNormObjective(parameters, zero_reference_policy='freeze')
    factors = {}; vectors = {}
    for name, obs in snapshot.matrices.items():
        a = obs.anchors
        p = (edit_scale * torch.randn(a.A.shape, generator=g, dtype=torch.float64)).requires_grad_()
        q = (edit_scale * torch.randn(a.B.shape, generator=g, dtype=torch.float64)).requires_grad_()
        factors[name] = (p, q, a)
    editable = {n for names in grouped.groups.values() for n in names}
    for name, obs in snapshot.vectors.items():
        if name in grouped.frozen_parameter_names or name not in editable:
            continue  # zero-reference groups stay frozen; gains/scalars are not editable
        vectors[name] = (edit_scale * torch.randn(obs.weight.shape, generator=g, dtype=torch.float64)).requires_grad_()
    canonical = {n: Factors.canonical(p, q, a) for n, (p, q, a) in factors.items()}
    radius = snapshot.weight_norm.double() * radius_fraction
    return grouped, factors, canonical, vectors, radius


@pytest.mark.parametrize('radius_fraction', [1e-4, 10.])   # radius active (s<1) and inactive (s=1)
def test_thin_objective_matches_dense_fp64_oracle_in_value_and_gradient(radius_fraction):
    parameters, snapshot = fixture(torch.float64)
    grouped, leaves, canonical, vectors, radius = build(parameters, snapshot, 11, .3, radius_fraction)
    J_thin, J_dense, thin, dense, s = thin_and_dense(parameters, snapshot, grouped, canonical, vectors, radius)
    assert (float(s.detach()) < 1) == (radius_fraction < 1)
    assert len(thin) == 15 and 'head.bias' in grouped.zero_reference_groups
    assert float(J_thin) == pytest.approx(float(J_dense), rel=1e-12, abs=1e-14)
    for group in thin:
        assert float(thin[group]) == pytest.approx(float(dense[group]), rel=1e-12, abs=1e-14)
    inputs = [t for p, q, _ in leaves.values() for t in (p, q)] + list(vectors.values())
    g_thin = torch.autograd.grad(J_thin, inputs, retain_graph=True)
    g_dense = torch.autograd.grad(J_dense, inputs)
    for a, b in zip(g_thin, g_dense):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-13)
    assert all(g.abs().sum() > 0 for g in g_dense)   # every editable leaf receives a gradient
    assert thin['head.bias'].item() == 0.0            # frozen zero-reference group contributes exactly zero


def test_sign_convention_outward_raises_objective_inward_lowers_it():
    parameters, snapshot = fixture(torch.float64)
    grouped = GroupedNormObjective(parameters, zero_reference_policy='freeze')
    name = 'blocks.0.attn.q.bias'; w = parameters[name].double(); group = 'attention.q.bias'
    R = grouped.reference_norms[group].double()
    for eps, expected_sign in ((1e-3, +1), (-1e-3, -1)):
        J = ((w + eps * w).norm() / R - 1).item()
        assert math.copysign(1, J) == expected_sign
    eps = torch.zeros((), dtype=torch.float64, requires_grad=True)
    J = ((w + eps * w).norm() / R - 1).mean()
    (dJ,) = torch.autograd.grad(J, eps)
    assert float(dJ) > 0                    # descent on J pushes eps negative = inward = shrink
    assert -100 * J.item() == pytest.approx(0.)  # reported percent = -100 J; zero at eps = 0


def test_fp32_stored_metric_gap_per_group_at_production_like_scale_and_sub_ulp_boundary():
    parameters, snapshot = fixture(torch.float32)
    grouped = GroupedNormObjective(parameters, zero_reference_policy='freeze')
    g = torch.Generator().manual_seed(5)
    # production-like: per-element edit RMS ~ 1e-2 of the weight RMS for every editable tensor
    report = {}
    for ratio, label in ((1e-2, 'production_like'), (1e-9, 'sub_ulp')):   # sub_ulp: relative per-element edit below half an FP32 ulp
        replacements = {}; continuous = {}
        for group, names in grouped.groups.items():
            if group in grouped.zero_reference_groups:
                continue
            new_sq = 0.
            for name in names:
                W = parameters[name]
                d = (torch.randn(W.shape, generator=g) * ratio * W.float().pow(2).mean().sqrt()) if label == 'production_like' else (W.float() * ratio)
                replacements[name] = (W.double() + d.double()).float()      # FP32 storage, as the live trainer
                new_sq += (W.double() + d.double()).square().sum().item()   # continuous FP64
            R2 = grouped.reference_squared[group].item()
            continuous[group] = -100 * (math.sqrt(new_sq) / math.sqrt(R2) - 1)
        for name in grouped.frozen_parameter_names:
            replacements[name] = parameters[name]
        for name in parameters:
            replacements.setdefault(name, parameters[name])
        stored = grouped.evaluate(replacements).metrics()['group_l2_percentage_reduction']
        gaps = {gname: abs(stored[gname] - continuous[gname]) for gname in continuous}
        report[label] = dict(max_abs_gap_percent=max(gaps.values()),
                             typical_percent=sorted(abs(v) for v in continuous.values())[len(continuous)//2],
                             stored_identity_everywhere=all(torch.equal(replacements[n], parameters[n]) for n in replacements))
    prod = report['production_like']
    assert prod['max_abs_gap_percent'] < 1e-3 * prod['typical_percent']      # stored vs continuous far below the shrink magnitude
    sub = report['sub_ulp']
    assert sub['stored_identity_everywhere'] is True                          # Astra's boundary: sub-ulp edits round to identity
    assert sub['max_abs_gap_percent'] > 0                                     # ...while the continuous objective moves
    print('fp32 stored-vs-continuous per-group gap', report)


def test_two_adam_ticks_with_shared_state_are_not_one_combined_step():
    torch.manual_seed(0)
    def run(two_ticks, lr_task=1e-3, lr_norm=5e-4):
        p = torch.nn.Parameter(torch.ones(4, dtype=torch.float64))
        opt = torch.optim.Adam([p], lr=lr_task, betas=(.9, .99), eps=1e-8)
        g_task = torch.tensor([1., -2., .5, 0.], dtype=torch.float64); g_norm = torch.tensor([.1, .1, -.3, .2], dtype=torch.float64)
        if two_ticks:
            opt.zero_grad(); p.grad = g_task.clone(); opt.step()
            for group in opt.param_groups: group['lr'] = lr_norm
            opt.zero_grad(); p.grad = g_norm.clone(); opt.step()
            steps = int(opt.state[p]['step'])
        else:
            opt.zero_grad(); p.grad = g_task + (lr_norm / lr_task) * g_norm; opt.step()
            steps = int(opt.state[p]['step'])
        return p.detach().clone(), steps
    two, n_two = run(True); one, n_one = run(False)
    assert n_two == 2 and n_one == 1
    assert not torch.allclose(two, one)      # the chosen semantics is a distinct schedule, documented, not a combined step
    # second tick honours the separately configured learning rate and shared moments
    p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64)); opt = torch.optim.Adam([p], lr=1e-3, betas=(.9, .99), eps=1e-8)
    p.grad = torch.ones(1, dtype=torch.float64); opt.step(); first = p.item()
    opt.param_groups[0]['lr'] = 1e-4; p.grad = torch.ones(1, dtype=torch.float64); opt.step()
    assert first == pytest.approx(-1e-3) and p.item() == pytest.approx(-1e-3 - 1e-4, rel=1e-6)


@pytest.mark.parametrize('radius_fraction', [1e-4, 10.])
def test_astra_continuous_objective_matches_dense_oracle_value_actor_gradient_and_sign(radius_fraction):
    """Astra's rewriter.online_norm.continuous_group_norm_objective on a real prepared action
    (group scales and global cap applied inside prepare_transformer_action) versus a dense FP64
    oracle built from the materialized proposal, including the gradient w.r.t. the actor parameter."""
    from rewriter.online_norm import continuous_group_norm_objective
    from rewriter.transformer_action import prepare_transformer_action
    from test_transformer_action import FactorActor
    parameters, snapshot = fixture(torch.float64)
    actor = FactorActor()
    prepared = prepare_transformer_action(actor, snapshot, parameters, radius_fraction=radius_fraction)
    assert (float(prepared.scale.detach()) < 1) == (radius_fraction < 1)
    grouped = GroupedNormObjective(parameters, zero_reference_policy='freeze')
    J, changes = continuous_group_norm_objective(prepared, grouped)
    # Dense FP64 reference built directly from the prepared (group-scaled) action, its global
    # cap and the FP64 base weights. NOTE: TransformerAction.materialize() casts the base to
    # float32 before adding the edit (a no-op in FP32 production, but it rounds this FP64
    # fixture by ~1e-7 relative), so materialize() is not an exact FP64 oracle.
    s = prepared.scale
    proposal = {}
    for name, base in parameters.items():
        if name in prepared.action.matrices:
            delta = prepared.action.matrices[name].dense()
            if name == 'embed.weight':
                delta = delta.T
            proposal[name] = base.double() + s * delta.double()
        elif name in prepared.action.vectors:
            proposal[name] = base.double() + s * prepared.action.vectors[name].double()
        else:
            proposal[name] = base.double()
    dense = {}
    for group, names in grouped.groups.items():
        if group in grouped.zero_reference_groups:
            dense[group] = torch.zeros((), dtype=torch.float64); continue
        new_sq = sum(proposal[n].square().sum() for n in names)
        dense[group] = new_sq.sqrt() / grouped.reference_squared[group].double().sqrt() - 1
    J_dense = torch.stack(list(dense.values())).mean()
    assert set(changes) == set(dense) and len(changes) == 15
    for group in dense:
        assert float(changes[group]) == pytest.approx(float(dense[group]), rel=1e-11, abs=1e-14)
    assert float(J) == pytest.approx(float(J_dense), rel=1e-11, abs=1e-14)
    (g_astra,) = torch.autograd.grad(J, actor.x, retain_graph=True)
    (g_dense,) = torch.autograd.grad(J_dense, actor.x)
    torch.testing.assert_close(g_astra, g_dense, rtol=1e-9, atol=1e-13)
    assert g_dense.abs().sum() > 0
    # sign/units: the offline realized metric reports -100 x this fraction (positive percent = shrink)
    realized = grouped.evaluate({n: v.detach() for n, v in proposal.items()}).metrics()['mean_group_l2_percentage_reduction']
    assert realized == pytest.approx(-100 * float(J), rel=1e-9, abs=1e-12)
