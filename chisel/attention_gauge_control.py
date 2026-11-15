"""Analytic feasibility control, not the learned Chisel policy or an optimizer."""
import math
import torch


@torch.no_grad()
def balance(snapshot, reference, head_dim, rank=16, max_iter=100):
    """Balance head-local value/output coordinates, preserving unchanged value bias.

    H = I + U diag(s-1) U.T, V' = HV, O' = OH^-1.
    U is orthonormal within each head and perpendicular to the value bias.
    Only cached directional energies enter the scalar optimization.
    """
    pairs = []
    for name, item in snapshot.items():
        if not name.endswith('.attn.v.weight'):
            continue
        other = name.replace('.attn.v.weight', '.attn.proj.weight')
        v = (item['raw'] + item['consolidated']).double()
        o = (snapshot[other]['raw'] + snapshot[other]['consolidated']).double()
        if v.shape[0] % head_dim or o.shape[1] != v.shape[0]:
            raise ValueError('head geometry does not match value/output matrices')
        bias = reference.get(name.removesuffix('weight') + 'bias', v.new_zeros(v.shape[0])).double()
        pairs.append(dict(name=name, other=other, v=v, o=o, bias=bias))
    if not pairs:
        raise ValueError('no attention value/output pairs')
    nv = sum(p['v'].square().sum() for p in pairs)
    no = sum(p['o'].square().sum() for p in pairs)
    if nv <= 0 or no <= 0:
        raise ValueError('control requires nonzero value and output families')
    for pair in pairs:
        v, o, bias = pair['v'], pair['o'], pair['bias']
        heads = v.shape[0] // head_dim
        columns = []
        for h in range(heads):
            count = rank // heads + int(h < rank % heads)
            if not count:
                continue
            start, end = h * head_dim, (h + 1) * head_dim
            b = bias[start:end]
            eye = torch.eye(head_dim, device=v.device, dtype=v.dtype)
            if b.norm() > 0:
                basis = torch.linalg.qr(torch.cat((b[:, None] / b.norm(), eye), dim=1), mode='complete').Q[:, 1:]
            else:
                basis = eye
            if count > basis.shape[1]:
                raise ValueError('requested rank exceeds bias-preserving head dimension')
            left, right = v[start:end], o[:, start:end]
            mismatch = basis.T @ (left @ left.T / nv - right.T @ right / no) @ basis
            values, vectors = torch.linalg.eigh(mismatch)
            u = basis @ vectors[:, values.abs().argsort(descending=True)[:count]]
            padded = v.new_zeros(v.shape[0], count)
            padded[start:end] = u
            columns.append(padded)
        pair['u'] = torch.cat(columns, dim=1)
        pair['a'] = (pair['u'].T @ v).square().sum(dim=1)
        pair['b'] = (o @ pair['u']).square().sum(dim=0)
    a, b = (torch.cat([p[key] for p in pairs]) for key in ('a', 'b'))
    with torch.enable_grad():
        logarithms = torch.zeros_like(a, requires_grad=True)
        optimizer = torch.optim.LBFGS([logarithms], max_iter=max_iter, line_search_fn='strong_wolfe',
                                     tolerance_grad=1e-12, tolerance_change=1e-15)
        def closure():
            optimizer.zero_grad()
            scales = logarithms.clamp(-math.log(2), math.log(2)).exp()
            loss = ((nv + (a * (scales.square() - 1)).sum()) / nv).sqrt()
            loss = loss + ((no + (b * (scales.pow(-2) - 1)).sum()) / no).sqrt()
            loss.backward()
            return loss
        optimizer.step(closure)
        scales = logarithms.detach().clamp(-math.log(2), math.log(2)).exp()
    effective = {n: (p['raw'] + p['consolidated']).clone() for n, p in snapshot.items()}
    offset = 0
    for pair in pairs:
        u, v, o = pair['u'], pair['v'], pair['o']
        s = scales[offset:offset + u.shape[1]]
        offset += len(s)
        effective[pair['name']] = (v + (u * (s - 1)) @ (u.T @ v)).to(effective[pair['name']])
        effective[pair['other']] = (o + ((o @ u) * (s.reciprocal() - 1)) @ u.T).to(effective[pair['other']])
    return effective
