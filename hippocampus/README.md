# Chisel project memory

The current implementation and checkpoint contract are documented in [README](../README.md) and [docs/checkpoints.md](../docs/checkpoints.md). YAML recipes are the execution specification.

Chisel has two jobs: simplify effective weights offline, and learn online which computations to consolidate. It independently emits edits to raw R and consolidated C. Offline shrinkage is measured on R+C, with equal weighting of weight families; also audit the actual combined Frobenius norm. Moving weights from R to C without changing R+C is not simplification. Ordinary MuonH updates R; Chisel alone updates C. Online credit comes from the next ordinary task gradient.

Train prediction preservation against actual teacher/student outputs. Reuse prior training minibatches or a representative replay sample. Exact teacher ties are equivalent when auditing top-100 rankings; CIFAR has ten classes, so audit all ten. Keep per-checkpoint input/output probes and their optimizer state trainable and persistent. Record finite numerical precision and zero-edit controls.

The older conserving-transfer-only specification omitted offline simplification and is superseded by the user's correction of November 15, 2026. It must not be used as the complete Chisel action or offline objective.

Use the copied [Muoff writing guide](style/04_writing_guide.md) and [reader-specific guidance](style/09_style_for_jeffrey.md) for reports. Current user instructions supersede historical process prescriptions in these reference files. Reports should be short HTML, with plots and notation carrying the evidence, clear topic sentences, and no undefined referents.

Do not put empirical results in this memory folder. Keep measurements, configurations, checkpoints and reports with the experiments. Preserve useful historical research on `archive/research-20261115`.
