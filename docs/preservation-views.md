# Preservation on alternate training-image views

**Recorded CIFAR inputs can be MixUp/CutMix composites.** The optional `replay_data_options.cifar.mix: null` evaluates preservation on the unmixed images from the same deterministic minibatch. Model state, captured sensors, dataset identity, split and replay position remain unchanged. This is an explicit offline preservation experiment; it does not change the recording run.

The domain-specific override accepts only `mix` and `transform`. Dataset paths, hashes, split seeds and train/validation selection cannot be overridden through this interface. Query images remain separate. The selected view is part of the run YAML, so teacher caches and deterministic replay use a fixed view policy within each run.

Use a new fitting stage when changing the view policy. This feature is not evidence that unmixed views improve preservation; compare matched runs and audit held-out images.
