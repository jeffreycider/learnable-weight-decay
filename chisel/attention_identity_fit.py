"""Learned attention-only feasibility diagnostic; not the unrestricted Chisel task."""
from functools import partial
import torch
from chisel.preservation import group_norm_change, effective_norm_change
from chisel.overfit_diagnostics import edited_weights


def propose(actor, probes, snapshot, reference=None, head_dim=None, parameterization='bounded_symmetric'):
    """Learn H from global transformer codes; V'=HV, O'=OH^-1, Hb=b.

    Only this diagnostic constrains edits to this exact attention symmetry.
    No analytically optimized factors or norm-minimizing solver are used.
    """
    codes = actor(probes.tokens(snapshot))
    if isinstance(codes, dict):
        raise ValueError('attention identity diagnostic expects ordinary matrix codes')
    return decode(codes, probes, snapshot, reference, head_dim, parameterization)


def decode(codes, probes, snapshot, reference=None, head_dim=None, parameterization='bounded_symmetric'):
    """Decode supplied global codes; shared by the diagnostic and augmented action."""
    if parameterization not in ('bounded_symmetric', 'exponential_consolidated'):
        raise ValueError('unknown attention identity parameterization')
    edits = {name: {bank: torch.zeros_like(item['raw']) for bank in ('raw', 'consolidated')}
             for name, item in snapshot.items()}
    for i, name in enumerate(probes.names):
        if not name.endswith('.attn.v.weight'):
            continue
        other = name.replace('.attn.v.weight', '.attn.proj.weight')
        v = (snapshot[name]['raw'] + snapshot[name]['consolidated']).double()
        o = (snapshot[other]['raw'] + snapshot[other]['consolidated']).double()
        dim = head_dim if head_dim is not None else snapshot[name]['attention_head_dim']
        heads = v.shape[0] // dim
        bias = (reference[name.removesuffix('weight') + 'bias'] if reference is not None
                else snapshot[name]['attention_value_bias']).double().reshape(heads, dim, 1)
        left = probes.write_left[i].double().reshape(heads, dim, probes.rank)
        code = codes[i].sum(0).double()
        identity = torch.eye(dim, device=v.device, dtype=v.dtype).expand(heads, -1, -1)
        if parameterization == 'exponential_consolidated':
            # A b=0 implies exp(A)b=b. Unlike the bounded symmetric diagnostic,
            # this includes shears and has no fixed cap on the transformation.
            right = probes.read_left[i].double().reshape(heads, dim, probes.rank)
            right = right - bias @ (bias.transpose(-1, -2) @ right) / bias.square().sum(-2, keepdim=True).clamp_min(1e-30)
            delta = torch.linalg.matrix_exp(left @ code @ right.transpose(-1, -2)) - identity
            # Keep R fixed: the raw-radius guard cannot rescale and break the
            # coupled identity. This is a C-only diagnostic, not the full policy.
            bank = 'consolidated'
        else:
            left = left - bias @ (bias.transpose(-1, -2) @ left) / bias.square().sum(-2, keepdim=True).clamp_min(1e-30)
            symmetric = left @ ((code + code.T) * .5) @ left.transpose(-1, -2)
            delta = .5 * symmetric / (1 + symmetric.square().sum((-1, -2), keepdim=True)).sqrt()
            bank = 'raw'
        value_delta = delta @ v.reshape(heads, dim, -1)
        # H^-1-I = -solve(H, delta) avoids subtracting nearly equal matrices.
        output_delta = o.reshape(-1, heads, dim).permute(1, 0, 2) @ (-torch.linalg.solve(identity + delta, delta))
        edits[name][bank] = value_delta.reshape_as(v).to(snapshot[name]['raw'])
        edits[other][bank] = output_delta.permute(1, 0, 2).reshape_as(o).to(snapshot[other]['raw'])
    return edits


def setup(parameterization='bounded_symmetric'):
    """Inject a norm-only diagnostic; actual GPT logit audits remain ordinary hooks."""
    def hook(config, state):
        fit = state.extras['fit']
        if config.get('diagnostic_lbfgs') or fit['kind'] != 'chisel':
            raise ValueError('attention identity fitting needs the shared/private Adam path')
        if any(item['consolidated'].count_nonzero() for item in fit['snapshot'].values()):
            raise ValueError('this feasibility diagnostic supports recorded C=0 checkpoints only')
        head_dim = fit['model'].blocks[0].attn.head_dim
        if any(block.attn.head_dim != head_dim for block in fit['model'].blocks):
            raise ValueError('all heads must share a declared dimension')
        proposal = partial(propose, reference=fit['base'], head_dim=head_dim,
                           parameterization=parameterization)
        state.extras['proposal_forward'] = torch.compile(proposal, fullgraph=True, dynamic=False) if config.get('compile_proposal') else proposal
        state.backward = backward
    return hook


def backward(config, state):
    fit = state.extras['fit']
    fit['private'].zero_grad(set_to_none=True)
    effective = edited_weights(config, state)
    loss = group_norm_change(fit['snapshot'], effective)
    loss.backward()
    shared = torch.nn.utils.clip_grad_norm_(state.model.parameters(), config['gradient_clip'])
    private = torch.nn.utils.clip_grad_norm_(fit['probes'].parameters(), config['gradient_clip'])
    fit['private'].step()
    state.extras['train_loss'] = loss.item()
    state.extras['metrics'] = dict(group_norm_reduction_pct=-100 * loss.item(),
        effective_norm_reduction_pct=-100 * effective_norm_change(fit['snapshot'], effective).item(),
        shared_gradient_norm=shared.item(), private_gradient_norm=private.item(),
        preservation_source='exact attention identity; numerical GPT audits logged separately')
