# Attention transformations plus independent weight edits

The optional `attention_branch` retains Chisel's independent raw/consolidated edits and adds coordinated attention transformations. One shared NanoGPT transformer produces all three output banks. Every checkpoint's read/write probes remain trainable and checkpointed.

For the ordinary low-rank and optional full-rank gain decoder, denote the independently emitted edits by `D_R,D_C`. For each attention head:

\[
\Delta R_V=D_{R,V}+(H-I)(R_V+C_V),\qquad
\Delta R_O=D_{R,O}+(R_O+C_O)(H^{-1}-I),\qquad
\Delta C=D_C.
\]

Other matrices receive `D_R,D_C` directly. The [attention diagnostic](learned-attention-identity.md) defines the learned, invertible `H` that fixes the unchanged value bias. The extra free edits can change the function; the combined action has **no algebraic preservation guarantee**. The common raw-radius safety scaling can also disturb attention cancellation if it activates. Ordinary training and audits evaluate the actual committed effective matrices.

`recipes/attention_augmented_offline.yaml` uses the ordinary offline logit-preservation objective, the mean weight-family effective `R+C` norm reduction, and independent raw/consolidated writes. It does not use the diagnostic's norm-only training loop. The replay adapter supplies attention dimensions and value biases, including for nonzero C.

An explicit `chisel.attention_stage.initialize` hook can copy an attention-only fit into the attention output head, while zeroing the new independent output heads. This preserves its initial edit exactly with the same private probes. The source checkpoint is hash checked. This is a new optimization stage with fresh shared Adam; the YAML and initialization artifacts must separately declare the origin and optimizer treatment of private probes. `audit_initial` evaluates recorded batches before the first update. Later fitted-batch audits are labelled separately.

This offline option does not yet wire attention metadata into the live NanoGPT hook. It does not establish learning improvements, generalization to unseen checkpoints, or successful R-to-C consolidation.
