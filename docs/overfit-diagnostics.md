# Diagnose shrinkage on a fixed fitting set

`chisel.overfit_diagnostics` uses the ordinary injected harness loop with one recorded checkpoint and its fixed input batch. It compares the learned Chisel parameterization with direct rank-limited and dense effective edits. Direct edits are optimization controls, not replacement learned models. All runs save their YAML, source, fitted parameters and optimizer state.

The ordinary diagnostic objective is mean fractional norm change across weight families plus a fixed coefficient times centered logit MSE. Ranking weights and KL weights are zero. Both raw and centered logit errors, their per-example maxima, and probability changes are measured on the fitting set. Removing a common logit shift preserves softmax probabilities; raw errors remain visible.

An optional feasibility phase holds a requested group-average shrinkage `s` using `mu * max(0, norm_change + s)^2` while minimizing logit MSE. This removes pressure to keep shrinking once `s` is attained. The finite penalty does not guarantee the shrinkage constraint: report measured shrinkage alongside prediction error. A fit returning to identity fails any positive target.

L-BFGS is a full-batch diagnostic optimizer. Its closures evaluate actual teacher/student predictions, use unclipped gradients, and do not advance private Adam. Report closure evaluations as well as outer steps. These offline experiments are intentionally outside the NanoGPT training-computation budget.
