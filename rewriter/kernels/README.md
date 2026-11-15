# Literal persistent Chisel training

`PersistentOfflineTrainer.step()` launches one cooperative CUDA kernel. Its
device-resident stage interpreter executes the current TransformerChisel,
anchored edits, frozen student GPT, exact-target FP64 KL, grouped norm and trust
objective, reverse passes, and optionally clipping/Adam. CUDA graphs and nested
kernel launches are not used. The teacher transformer is unnecessary: labels
come from the fixed teacher cache.

This is an experimental engineering implementation. The qualification results
in `runs/megakernel-transformer-v2/` describe specific cases, not approval to
replace scientific training. The zero-edit numerical KL floor is material
relative to the deadband. Runtime excludes host plan construction, checkpoint
loading and reconstruction of resident teacher probabilities.

The optional CUTLASS path calls `MmaMultistage` as a device function inside the
interpreter. Set `CHISEL_CUTLASS_INCLUDE` to the `include` directory of NVIDIA
CUTLASS 3.9.2, commit `ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e`.
Obtain CUTLASS from its official repository and retain its BSD license. No
CUTLASS host API or child kernel is launched. Without this environment variable,
the built-in WMMA implementation remains available.

The H100 engineering configuration uses:

```sh
export CHISEL_STREAM_ATTENTION=3
export CHISEL_PERSISTENT_MAX_REGISTERS=128
export CHISEL_PERSISTENT_BLOCK_MULTIPLIER=4
export CHISEL_HEAD_CHUNK_TOKENS=1024
export CHISEL_FUSED_HEAD_KL=1
export CHISEL_PACKED_ATTENTION=1
export CHISEL_ATTENTION_SHARED_STRIDE=136
export CHISEL_CUTLASS_INCLUDE=/path/to/cutlass/include
```

`CHISEL_CUDA_CCCL_INCLUDE` can point to missing CUDA CCCL headers if the runtime
image does not supply them. The loader checks cooperative occupancy and caps
the grid accordingly. Compilation and buffer construction happen before the
timed `step()`.

`microbatch_sequences=16` can evaluate a complete 64-sequence panel with reused
GPT buffers. Its KL mean is formed over the full panel, then the deadband is
applied once. Vocabulary chunks bound student-head storage. Exact FP64 teacher
probabilities remain resident and unmodified; the kernel currently does not
reconstruct them directly from the factorized cache.

`load_checkpoint(model, snapshot)` refreshes the fixed weight, observation,
anchor, group-reference and radius buffers, plus the corresponding instruction
constants. It preserves the compiled kernel, activation storage, accumulated
actor gradients and optimizer history. It rejects changed observation shapes
or changed editable/zero-reference group structure. `load_panel(inputs,
teacher_log_probs)` copies a same-shaped panel into the existing input and
target buffers. The constructor adopts the target tensor's storage, which this
method overwrites. Target reconstruction/source staging remains separate and
can require substantial extra memory; these methods do not establish complete
state-bank throughput. See the dedicated checkpoint-reuse qualification.

For the scientific 200-checkpoint optimizer batch, each checkpoint plan must
use `state_weight=1/200`, `apply_optimizer=False`, and a shared actor gradient
accumulator. Clear that accumulator once before the batch. After all local
checkpoint contributions, perform the external distributed SUM, then call
`PersistentActorOptimizer.step()` exactly once. No clipping or Adam belongs
inside the checkpoint plans in this mode. Full distributed execution is not
yet qualified; tests cover accumulation of two checkpoint contributions and
the single subsequent optimizer operation. One checkpoint panel with local
Adam is a different qualification scope.

`scripts/qualify_persistent_offline_chisel.py` compares against the eager
numerical reference. `scripts/benchmark_current_chisel_panel.py` separately
times the actual production helpers with compiled Chisel, accelerated grouped
norm, eager Flash GPT and production chunked KL. Its optional fused-KL variant
is separate from the production configuration. Both comparisons begin with
the same resident exact teacher targets.
