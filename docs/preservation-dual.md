# Adjusting preservation pressure near the constraint

**The additive dual can adjust slowly near a very small KL tolerance.** The optional multiplicative update uses the relative constraint violation, with the per-step exponent bounded:

`λ ← clip(λ exp(η clip(KL / tolerance − 1, −1, 1)), minimum, maximum)`

The primal loss remains `norm_change + λ KL` plus its existing terms. Above twice the tolerance, λ grows by at most `exp(η)` per update; at zero KL, λ decreases by that factor. Clipping the relative violation prevents a single large error from causing an unbounded jump. This is an optimization option, not a convergence guarantee. The additive update remains the default.

**A new fitting stage can retain already-fitted probes.** Use the hash-pinned `initialize_shared_weights` hook after model construction and `initialize_fitted_probes` after offline setup. The latter restores the private weights and private Adam state; new private versions are written into the new run directory. The shared optimizer, dual and sampler start from the new YAML. This is explicitly a new stage, not an exact resume. To change the restored private learning rate, declare it through the existing curriculum hook.
