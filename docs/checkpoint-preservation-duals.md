# Preservation pressure per checkpoint

`preservation.dual_scope: checkpoint` gives each source checkpoint an independent KL constraint. The shared Chisel network and every checkpoint's private probes still train jointly. The default `shared` scope retains the previous constraint on the sampled cohort's mean KL.

For checkpoint `i`, the loss contribution is

`fractional_group_norm_change_i + lambda_i * mean_KL_i + configured_ranking_and_storage_terms_i`.

The checkpoint batch averages these contributions. The logged `kl_penalty` is the mean of `lambda_i * mean_KL_i`, including any preservation curriculum multiplier; multiplying the mean lambda by the mean KL would be incorrect. `preservation_multiplier` reports the mean multiplier used on that batch.

After the last private update's loss measurement, only the sampled checkpoint's multiplier updates. Multipliers use the configured additive or multiplicative rule, and inherit the current predeclared tolerance schedule. The measurement precedes the final private optimizer step, as in the existing loop. Repeated private updates on the same batch do not create additional dual updates.

The source checkpoint SHA identifies each multiplier. Rank checkpoints save their multiplier dictionaries, audits gather the owning ranks' dictionaries, and exact resume restores them. Validation adaptation copies a multiplier and never updates training state. New staged initialization intentionally starts new multipliers unless it is an exact resume.

This separates difficult checkpoints; it does not guarantee preservation on every image or on unseen data. Changing the number of checkpoints changes the frequency of each multiplier's updates. Compare both scopes with recorded exposure counts and learning rates.
