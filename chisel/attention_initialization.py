"""Explicit analytic initialization experiment; later edits are fully learned."""
import torch
from chisel.attention_gauge_control import balance


def identity_codes():
    """Initialize a constant raw-edit code, with a trainable ordinary output head."""
    @torch.no_grad()
    def hook(config, state):
        actor = state.model
        actor.proj.weight.zero_()
        actor.proj.bias.zero_()
        actor.proj.bias.view(2, actor.rank, actor.rank)[0].copy_(
            torch.eye(actor.rank, device=actor.proj.bias.device))
    return hook


def measure_initial():
    """Record step zero before any learning, without advancing the harness clock."""
    def hook(config, state):
        from chisel.overfit_diagnostics import measure
        step = state.step
        state.step = -1
        try:
            measure(every=1)(config, state)
        finally:
            state.step = step
    return hook


@torch.no_grad()
def probes(probes, snapshot, seed, reference=None, actor=None, head_dim=128):
    """Initialize private factors to a known finite attention edit, then train them.

    This is an analytic starting point, not evidence of learned simplification.
    All read/write maps and all shared Chisel parameters remain trainable.
    """
    codes = actor(probes.tokens(snapshot))
    expected = torch.zeros_like(codes)
    expected[:, 0] = torch.eye(probes.rank, device=codes.device, dtype=codes.dtype)
    torch.testing.assert_close(codes, expected, rtol=0, atol=0)
    _, factors = balance(snapshot, reference, head_dim, probes.rank, return_factors=True)
    for i, name in enumerate(probes.names):
        probes.write_left[i].zero_()
        if name not in factors:
            continue
        left, right = factors[name]
        # Balance factor magnitudes without changing their product.
        ratio = (right.norm(dim=0) / left.norm(dim=0).clamp_min(1e-30)).sqrt()
        scale = snapshot[name]['raw'].double().square().mean().sqrt().sqrt()
        probes.write_left[i].copy_(left * ratio / scale)
        probes.write_right[i].copy_(right / ratio / scale)
