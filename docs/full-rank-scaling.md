# Optional full-rank writes

`model.full_rank_gains: true` adds scaling edits to Chisel's existing learned low-rank edits. The default remains `false`.

For each editable matrix, let `W = R + C`, let `U,V` be its trainable checkpoint-private write maps, and let `rho = RMS(R)`. For each destination `s ∈ {raw, consolidated}`, the shared global transformer emits `K_s`, vectors `a_s,b_s`, and a scalar `mu_s`:

\[
\Delta W_s=\rho U K_s V^\top
+W\odot\left[\mu_s\mathbf1\mathbf1^\top
 +(Ua_s)\mathbf1^\top+\mathbf1(Vb_s)^\top\right].
\]

The added term can have full matrix rank. Its coefficients still come from whole-model transformer processing; dense weights enter the write algebra, not the transformer's input tokens. Raw and consolidated edits remain independent.

The added output head starts at zero. Existing proposals are therefore preserved when its zero weights are added to a saved model. All new coefficients are trainable; the unchanged prediction-preservation objective decides which edits to retain.

This option changes neither the editable parameter inventory nor NanoGPT, MuonH, data, or the training loop. It does not establish function preservation by construction. The reference action's old rank bound no longer applies to this optional branch; norms and gradients are measured using the current dense action and actual GPT evaluations.

Tests cover existing-proposal equality, full-rank representability, independent R/C edits, global observation dependence, trainable private maps, and CUDA gradient parity and compilation reuse. These are correctness checks, not evidence of better learned simplification.

Optional `normalize_gains: true` divides the uniform code by √(nm), row gains by √m, and column gains by √n for an n×m matrix. For RMS-ρ weights and unit-norm probe columns, this makes each gain coordinate's edit norm approximately ρ, matching a low-rank code coordinate. Without this normalization, the uniform branch's derivative norm grows as ρ√(nm), which can dominate optimization despite adding only one output coordinate. This rescales the parameterization without changing its representable edits; zero added heads retain exactly the same initial edit. The default remains false for reproducibility of existing experiments.
