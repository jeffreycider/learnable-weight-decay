# Held-out checkpoint evaluation

**Fit private probes before measuring transfer to a held-out checkpoint.** `validation_adaptation` partitions the checkpoint's recorded minibatch into support and query examples using a declared seed. Each FineWeb example is an entire sequence; CIFAR examples are images. The shared Chisel backbone stays fixed. Private Adam fits only the support examples, with the same effective-norm reward and prediction-preservation objective used in training.

**Query examples supply measurements only.** The final edit is audited on disjoint query examples. The adaptation dual is copied from the training dual; neither its updates nor the fitted probes enter the training state. Every evaluation starts from the checkpoint-private initialization. This measures adaptation at an unseen checkpoint, not training without access to that checkpoint's data.

Set `validation_adaptation` as in [the example recipe](../recipes/validation_adaptation.yaml), plus a validation manifest disjoint from the training manifest. `steps`, `private_lr`, `support_examples`, `query_examples`, and `seed` are explicit. The requested examples must fit in the recorded minibatch. Per-checkpoint audit files include adaptation steps, example counts, and additional model-forward counts. Validation does not select the within-run schedule or stopping point.

The per-checkpoint adaptation trace records support-set shrinkage and preservation before updates at powers of two, plus an audit after the final private update. Training-checkpoint audits remain unchanged. Omitting `validation_adaptation` retains the audit of freshly initialized private probes.
