# Measure prediction movement directly

Set `preservation.center_logits: false` to penalize mean squared changes in raw logits, including common shifts across classes. The default retains the previous shift-invariant objective for reproducibility. Set `preservation.ranking_weight: 0` to remove rank penalties.

`logit_diagnostics` reports per-example RMS and maximum logit changes, then their mean, 95th percentile and maximum over the fitting set. Raw errors include common shifts; centered errors remove those shifts. Probability displacement is reported both as the largest class-probability change and total variation. Near-tie rank swaps are not a preservation objective.

A diagnostic caller can omit unused KL and ranking measurements. Omitted measurements are absent from returned metrics, rather than reported as measured zeros. A nonzero KL multiplier always computes KL. The ordinary offline task still requires KL measurements for its dual update.
