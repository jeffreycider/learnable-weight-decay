"""Additive per-token losses for the explicit tokenwise-v1 training recipe.

The mean constraint has one fixed multiplier per checkpoint and Adam update.
The quadratic term acts on individual tokens, not the checkpoint's mean KL.
All terms can share one GPT backward; accumulate sums divided by the COMPLETE
panel's token count, including when the final microbatch is smaller.

``scale`` is a numerical normalization in nats/token, independent of either
tolerance. Tightening a tolerance must not silently change normalization and
therefore multiply curvature. Absolute-gap ranking loss has squared-logit
units; relative-gap ranking loss is dimensionless. Its weight is independent
of the KL multiplier and stays fixed within each accumulated update.
"""
import math

import torch


def token_preservation_terms(kl, *, multiplier, mean_tolerance,
                             tail_tolerance, scale, rho,
                             ranking_loss=None, ranking_weight=0.):
    """Return unreduced terms, retaining gradients only through measurements.

Dual ascent still uses the whole-panel mean constraint after the primal update.
This helper does not update dual state or provide an evaluation acceptance rule.
``rho`` and ``ranking_weight`` are fixed throughout one accumulated update.
"""
    values = (multiplier, mean_tolerance, tail_tolerance, scale, rho, ranking_weight)
    if not all(math.isfinite(x) for x in values):
        raise ValueError('finite loss coefficients required')
    if scale <= 0 or min(multiplier, mean_tolerance, tail_tolerance, rho, ranking_weight) < 0:
        raise ValueError('positive scale and nonnegative coefficients required')
    if ranking_weight and ranking_loss is None:
        raise ValueError('ranking measurements required for a nonzero ranking weight')
    if ranking_loss is not None and ranking_loss.shape != kl.shape:
        raise ValueError('one ranking measurement per KL token required')
    measured = kl.clamp_min(0.)
    linear = multiplier * ((measured - mean_tolerance) / scale)
    tail = 0.5 * rho * ((measured - tail_tolerance) / scale).relu().square()
    ranking = torch.zeros_like(kl) if ranking_loss is None else ranking_weight * ranking_loss
    return dict(dual_linear=linear, tail_quadratic=tail, ranking=ranking,
                total=linear + tail + ranking)
