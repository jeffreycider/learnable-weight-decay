"""Observe existing checkpoint gradients without another model evaluation.

This estimates subsampling noise relative to the *observed panel mean*, not
noise over all possible data or the batch size that optimizes convergence.
Gradients must come from one unchanged actor/dual state, before clipping.
"""
import math

import torch


class CheckpointGradientMoments:
    """Only scalar second moments are retained; gradient storage is never owned.

    Call observe on each graph's private gradient buffers after replay and
    before their next overwrite. Inputs already contain state_weight. At the
    end, supply the globally SUM-reduced gradient and SUM-reduced statistics.
    No communication, optimizer update, or device synchronization is implicit.
    """

    def __init__(self, *, device, state_weight):
        if not math.isfinite(state_weight) or state_weight <= 0:
            raise ValueError('positive finite checkpoint gradient weight required')
        self.state_weight = float(state_weight)
        # [number of checkpoints, sum of unweighted squared gradient norms]
        self.statistics = torch.zeros(2, dtype=torch.float64, device=device)

    @torch.no_grad()
    def observe(self, gradients):
        gradients = tuple(gradients)
        if not gradients or any(g is None or g.device != self.statistics.device for g in gradients):
            raise ValueError('complete gradient list on the statistics device required')
        # FP64 reduction avoids cancellation artifacts for nearly identical
        # gradients. This adds arithmetic, but no GPT computation or dense
        # concatenation; actual CUDA overhead must be measured separately.
        norms = torch.stack(torch._foreach_norm(gradients, 2, dtype=torch.float64))
        squared_norm = norms.square().sum() / self.state_weight**2
        self.statistics[0].add_(1)
        self.statistics[1].add_(squared_norm)
        return squared_norm


@torch.no_grad()
def summarize_checkpoint_gradients(statistics, global_mean_gradients, *, batch_sizes):
    """Exact finite-population variance formula, with measured FP arithmetic.

    statistics must be globally summed once. global_mean_gradients must be the
    existing global mean before clipping, i.e. each of N contributions was
    weighted by 1/N. The estimate concerns uniformly sampled subsets of these
    N checkpoints and their fixed token panels. It does not model Adam, clipping,
    temporal correlations, unseen checkpoints, or different sequence counts.
    """
    if statistics.shape != (2,) or not bool(torch.isfinite(statistics).all()):
        raise ValueError('two finite globally summed statistics required')
    count, sum_norm_sq = statistics.detach().cpu().tolist()
    if count < 2 or count != int(count) or sum_norm_sq < 0:
        raise ValueError('at least two checkpoint gradients required')
    count = int(count)
    gradients = tuple(global_mean_gradients)
    if not gradients or any(g is None for g in gradients):
        raise ValueError('complete globally averaged gradient required')
    mean_norm_sq = sum(float(g.detach().double().square().sum()) for g in gradients)
    if not math.isfinite(mean_norm_sq):
        raise ValueError('finite global mean gradient required')
    centered_sum = sum_norm_sq - count * mean_norm_sq
    # Negative values larger than roundoff usually indicate wrong reduction or
    # normalization. Tiny negatives are possible for identical FP32 gradients.
    tolerance = 1e-5 * max(sum_norm_sq, count * mean_norm_sq, 1e-300)
    if centered_sum < -tolerance:
        raise ValueError('inconsistent gradient weighting or global reduction')
    sample_variance = max(0., centered_sum) / (count - 1)
    estimates = []
    for batch in batch_sizes:
        if type(batch) is not int or not 1 <= batch <= count:
            raise ValueError('subset size must be an integer within observed panel')
        error_sq = (1 / batch - 1 / count) * sample_variance
        estimates.append(dict(checkpoints=batch, expected_squared_error=error_sq,
            rms_error_relative_to_panel_mean=(math.sqrt(error_sq / mean_norm_sq)
                if mean_norm_sq else None)))
    return dict(checkpoints=count, panel_mean_gradient_norm=math.sqrt(mean_norm_sq),
        checkpoint_gradient_variance_trace=sample_variance, estimates=estimates,
        scope='Uniform subsets of this fixed checkpoint/token panel; raw gradients before clipping/Adam')
