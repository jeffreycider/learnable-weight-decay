# Train shrinkage subject to raw-logit preservation

The default offline recipe now minimizes group-average fractional change in effective weight norm, with an adaptive multiplier on **raw-logit MSE**. It does not train against rankings or KL, and the offline consolidated-weight penalty is zero.

For each checkpoint, the primal loss is `norm_change + λ × logit_MSE`. The existing multiplicative dual increases `λ` above the declared MSE tolerance and decreases it below that tolerance. Each checkpoint owns its multiplier; checkpointing and resume retain those values alongside its learned probes. The loss still has a logit gradient below the tolerance whenever the multiplier is positive. This is an optimization method, not a guarantee that the constraint is satisfied after every edit.

The default tolerance of `1e-10` corresponds to raw-logit RMS error of `1e-5`. It is a pilot, not a bound on maximum individual logit displacement. Report that maximum separately. The initial multiplier, adaptation rate, tolerance and private fitting steps remain explicit YAML settings.

Archived recipes without `constraint_metric` retain their KL interpretation. Selecting `constraint_metric: logit_mse` routes the same measured MSE into both the primal gradient and the dual update. Disabled KL and rank measurements are omitted from logs.
