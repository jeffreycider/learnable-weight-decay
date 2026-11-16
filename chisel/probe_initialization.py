"""Optional learned-probe initialization with aligned attention value/output directions."""
import torch


@torch.no_grad()
def orthonormal(probes, snapshot, seed, reference=None, actor=None):
    """Condition the initial learned maps; impose no constraint on later updates."""
    for parameter in probes.parameters():
        parameter.copy_(torch.linalg.qr(parameter, mode='reduced').Q)


@torch.no_grad()
def attention_pairs(probes, snapshot, head_dim, seed, reference=None, actor=None):
    """Initialize paired write maps; every map remains trainable after initialization.

    For y=O V x, the accessible directions include dV=u(u^T V) and
    dO=-(O u)u^T within a head. This does not prescribe Chisel's output codes
    or restrict later probe updates. Random initial maps remain on other matrices.
    """
    generator = torch.Generator(device=next(probes.parameters()).device).manual_seed(seed)
    indices = {name: index for index, name in enumerate(probes.names)}
    for name, index in indices.items():
        if not name.endswith('.attn.v.weight'):
            continue
        output = name.replace('.attn.v.weight', '.attn.proj.weight')
        if output not in indices:
            raise ValueError(f'value matrix has no paired output: {name}')
        other = indices[output]
        value = snapshot[name]['raw'] + snapshot[name]['consolidated']
        projection = snapshot[output]['raw'] + snapshot[output]['consolidated']
        if value.shape[0] % head_dim or projection.shape[1] != value.shape[0]:
            raise ValueError('paired probe head dimensions disagree with checkpoint')
        heads = value.shape[0] // head_dim
        bias = (reference or {}).get(name.removesuffix('weight') + 'bias', value.new_zeros(value.shape[0]))
        directions = value.new_zeros(value.shape[0], probes.rank)
        for head in range(heads):
            columns = list(range(head, probes.rank, heads))
            if not columns:
                continue
            if len(columns) > head_dim:
                raise ValueError('too many probe directions in one head')
            basis = torch.randn(head_dim, len(columns), device=value.device,
                                dtype=value.dtype, generator=generator)
            head_bias = bias[head * head_dim:(head + 1) * head_dim].to(value)
            if head_bias.norm() > 0:
                if len(columns) >= head_dim:
                    raise ValueError('probe rank leaves no room to preserve value bias')
                unit = head_bias / head_bias.norm()
                basis = basis - unit[:, None] * (unit @ basis)[None, :]
            basis = torch.linalg.qr(basis, mode='reduced').Q
            directions[head * head_dim:(head + 1) * head_dim, columns] = basis
        maps = [(probes.write_left[index], directions),
                (probes.write_right[index], value.T @ directions),
                (probes.write_left[other], projection @ directions),
                (probes.write_right[other], directions)]
        for parameter, initial in maps:
            # Keep random columns on dormant zero-output branches so they can learn.
            norms = initial.norm(dim=0, keepdim=True)
            parameter.copy_(torch.where(norms > 1e-12, initial / norms.clamp_min(1e-12), parameter))


@torch.no_grad()
def zero_write_left(probes, snapshot, seed, reference=None, actor=None):
    """Identity initial edit with trainable left factors for an already trained backbone."""
    if actor is None:
        raise ValueError('zero-write initialization requires the shared Chisel backbone')
    codes = actor(probes.tokens(snapshot))
    if not (codes[:, 0] + codes[:, 1]).abs().any():
        raise ValueError('zero write maps and zero effective output codes would prevent learning')
    for parameter in probes.write_left:
        parameter.zero_()
