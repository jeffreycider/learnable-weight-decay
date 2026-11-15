import itertools

import pytest
import torch

from rewriter.gradient_batch_diagnostic import (
    CheckpointGradientMoments, summarize_checkpoint_gradients,
)


def test_subset_error_matches_exhaustive_sampling_and_distributed_sum():
    gradients = torch.tensor([[1., 4., -1.], [-3., 2., 1.], [7., 0., 2.], [2., 1., 9.]], dtype=torch.float64)
    observers = [CheckpointGradientMoments(device='cpu', state_weight=.25) for _ in range(2)]
    for i, g in enumerate(gradients):
        observers[i % 2].observe([g[:2] / 4, g[2:] / 4])
    stats = sum(x.statistics for x in observers)
    mean = gradients.mean(0)
    summary = summarize_checkpoint_gradients(stats, [mean[:2], mean[2:]], batch_sizes=[1, 2, 3, 4])
    for row in summary['estimates']:
        errors = [(gradients[list(ids)].mean(0) - mean).square().sum()
                  for ids in itertools.combinations(range(4), row['checkpoints'])]
        assert row['expected_squared_error'] == pytest.approx(float(torch.stack(errors).mean()))


def test_observer_does_not_modify_or_retain_gradient_storage():
    observer = CheckpointGradientMoments(device='cpu', state_weight=.5)
    g = torch.tensor([1., 2.], requires_grad=True)
    original = g.detach().clone()
    observer.observe([g])
    assert torch.equal(g, original)
    saved = observer.statistics.clone()
    with torch.no_grad():
        g.zero_()
    assert torch.equal(observer.statistics, saved)
    assert not observer.statistics.requires_grad


def test_identical_gradients_zero_noise_and_wrong_scaling_rejected():
    observer = CheckpointGradientMoments(device='cpu', state_weight=.5)
    for _ in range(2):
        observer.observe([torch.tensor([1., 2.])])
    result = summarize_checkpoint_gradients(observer.statistics, [torch.tensor([2., 4.])], batch_sizes=[1, 2])
    assert result['checkpoint_gradient_variance_trace'] == pytest.approx(0.)
    with pytest.raises(ValueError, match='weighting'):
        summarize_checkpoint_gradients(observer.statistics, [torch.tensor([4., 8.])], batch_sizes=[1])


def test_zero_mean_has_no_fabricated_relative_noise_and_invalid_inputs_fail():
    observer = CheckpointGradientMoments(device='cpu', state_weight=.5)
    for x in (-1., 1.):
        observer.observe([torch.tensor([x / 2])])
    result = summarize_checkpoint_gradients(observer.statistics, [torch.tensor([0.])], batch_sizes=[1])
    assert result['estimates'][0]['rms_error_relative_to_panel_mean'] is None
    with pytest.raises(ValueError):
        summarize_checkpoint_gradients(observer.statistics, [torch.tensor([0.])], batch_sizes=[3])
    with pytest.raises(ValueError):
        observer.observe([None])
