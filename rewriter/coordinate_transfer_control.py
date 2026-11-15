"""Frozen residual-block ablation conditional on identical learned interfaces.

This is not an independently trained no-transformer baseline. All non-block
parameters are copied from the trained actor into both arms. The reference
blocks are restored to a supplied, pinned initialization, whose zero residual
outputs make the stack an identity map. Fresh private maps are paired exactly.
"""
from __future__ import annotations
import copy
import torch
from .coordinate_chisel import CoordinateChisel
from .distributed_probe_store import _digest


def frozen_block_pair(trained: CoordinateChisel, initialized: CoordinateChisel):
    """Return two frozen actors without mutating either input or consuming RNG.

    The caller must pin both input states/configs. This helper cannot infer an
    original initializer from a trained checkpoint or qualify transfer itself.
    """
    if trained.configuration() != initialized.configuration():
        raise ValueError('trained and initialization architecture differ')
    if trained.frame_mode != 'tied_orthonormal':
        raise ValueError('this protocol requires tied orthonormal private frames')
    for block in initialized.blocks:
        for projection in (block.attn.proj, block.mlp.proj):
            if any(torch.count_nonzero(p).item() for p in projection.parameters()):
                raise ValueError('reference is not the zero-output initial residual stack')
    if not any(torch.count_nonzero(head.weight).item() for head in trained.output_heads):
        raise ValueError('trained interfaces have dead zero output weights')
    learned = copy.deepcopy(trained)
    identity = copy.deepcopy(trained)
    identity.blocks.load_state_dict(initialized.blocks.state_dict(), strict=True)
    for actor in (learned, identity):
        actor.eval().requires_grad_(False)
        actor.zero_grad(set_to_none=True)
    outside = lambda actor: {k:v for k,v in actor.state_dict().items() if not k.startswith('blocks.')}
    if _digest(outside(learned)) != _digest(outside(identity)):
        raise RuntimeError('block ablation changed a learned interface')
    metadata = dict(
        schema='coordinate_frozen_block_ablation_v1',
        trained_actor_digest=_digest(trained.state_dict()),
        initialization_actor_digest=_digest(initialized.state_dict()),
        common_interface_digest=_digest(outside(learned)),
        reference_block_digest=_digest(initialized.blocks.state_dict()),
        interpretation='residual-block contribution conditional on learned common interfaces',
        independent_identity_stack_training=False,
    )
    return learned, identity, metadata


def fresh_paired_private(learned, identity, snapshot, *, state_key, seed):
    """Same private values/identity in each arm; optimizers must be created fresh."""
    if learned.configuration() != identity.configuration():
        raise ValueError('paired actor configuration differs')
    if any(p.requires_grad or p.grad is not None for a in (learned,identity) for p in a.parameters()):
        raise ValueError('both shared actors must be frozen and gradient-cleared')
    first = learned.make_private(snapshot, state_key=state_key, seed=seed)
    second = identity.make_private(snapshot, state_key=state_key, seed=seed)
    second.load_state_dict(first.state_dict(), strict=True)
    if _digest(first.state_dict()) != _digest(second.state_dict()):
        raise RuntimeError('private initialization differs')
    return first, second
