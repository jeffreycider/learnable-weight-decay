# Learned attention identity diagnostic

The global Chisel transformer learns norm-reducing attention rewrites from zero output heads. This diagnostic tests whether unrestricted fitting struggles to coordinate the two sides of an attention rewrite. It does not replace independent raw/consolidated edits in production Chisel.

For each attention head, let `V` be its value weights, `O` its output-projection columns, and `b` its unchanged value bias. Chisel's matrix code and trainable checkpoint-private write map generate a symmetric matrix `S` perpendicular to `b`:

\[
H=I+\frac{S}{2\sqrt{1+\|S\|_F^2}},\qquad V'=HV,\qquad O'=OH^{-1},\qquad Hb=b.
\]

Consequently `O'H(Vx+b)=O(Vx+b)`. Head-local attention mixing commutes with the value-coordinate transform; q/k normalization, RoPE, and learned attention scales remain unchanged. The eigenvalues of `H` lie between one half and three halves. Algebra uses FP64 before casting edits back to the model's parameter precision. Actual GPT logit measurements check floating-point errors separately.

The injected `chisel.attention_identity_fit.setup` hook follows `chisel.overfit_diagnostics.setup`. It trains the ordinary global transformer and private probes with Adam against the same mean fractional effective-norm objective. No analytic norm minimizer or optimized starting factors are used. Existing diagnostic measurement/checkpoint hooks save actual logit errors and both shared/private learned state.

Scope: one recorded checkpoint with `C=0`; only value/output matrices change, with the entire effective edit assigned to `R`. The all-family norm metric still counts every editable hidden matrix. This restricted diagnostic does not demonstrate consolidation learning, general unrestricted rewrites, cross-checkpoint generalization, or improved future training. Its norm-only updates are not GPT evaluations; GPT measurements occur at the configured audit cadence.

`recipes/attention_identity_offline.yaml` applies the same restricted diagnostic to a mixed checkpoint catalog. The injected replay component supplies each checkpoint's unchanged value biases and attention-head dimensions. The existing checkpoint owner scheme, shared-gradient reduction, private Adam persistence, distributed audits, and durable checkpoint generations are reused. Each update fits the norm objective only; recorded-token audits measure numerical preservation and do not supply optimization gradients. An audit of a training checkpoint is therefore not labelled as evaluation on fitted tokens. This mode still assigns the whole edit to R and remains an attention-only diagnostic.
