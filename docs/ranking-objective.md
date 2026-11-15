# Emphasizing rare ranking violations

**A mean ranking penalty dilutes errors when almost every token already preserves its ranking.** The optional `microbatch_max` reduction gives each microbatch's worst per-position ranking penalty its full gradient. The outer objective averages those maxima with weights proportional to microbatch sizes. KL and logit MSE retain their existing means.

**Microbatch size now affects the objective.** This is a weighted average of microbatch maxima, not the maximum over the whole batch. The `ranking_loss` metric reports the penalty actually optimized. Token-ranking change counts and KL audits remain unchanged. The default is still the original mean reduction.

Use [the example recipe](../recipes/worst_ranking_violation.yaml) to select the reduction. The existing per-position penalty averages adjacent top-k gap violations and adds the cutoff violation; only the reduction across positions changes here. This option is an experiment, not a guarantee of preservation.
