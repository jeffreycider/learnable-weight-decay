# Ranking errors relative to the original gap

`ranking_normalization: relative` penalizes closing or reversing a small original logit gap more strongly than the default absolute squared-gap penalty.

For teacher gap `g > 0` and edited gap `s`, the term is

`[max(0, margin_fraction*g - s) / max(g, rank_gap_floor)] ** ranking_power`.

The same rule covers the top-k membership boundary. Exact teacher ties remain equal rank. The positive floor limits gradient magnitude; it does not exempt any positive teacher gap from the penalty or change ranking-change measurements. KL is unchanged.

The absolute default is unchanged. Relative and absolute penalties have different units and need separately declared weights. The example YAML uses a floor of `1e-6` logit units, a squared hinge, and a maximum within each microbatch before averaging microbatches. This is an optimization pilot, not a guarantee of zero ranking changes or generalization.
