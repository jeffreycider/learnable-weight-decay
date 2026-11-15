# PR323 eight-H100 baseline

`configs/baseline_8h100_seed1.json` restores the original execution shape: eight H100s, one 64-sequence microbatch per rank, 1024 tokens/sequence, and 524288 global tokens per optimizer step. Model, AdamW/Muon hyperparameters, 3250-step cooldown schedule, data order, and live-weight evaluation are unchanged. This configuration uses the project's deterministic Torch2.11 execution policy; the published PR323 log used Torch2.10+cu128, so it is not an identical software-stack reproduction.

```sh
PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=2 \
/opt/venv211/bin/torchrun --standalone --nproc_per_node=8 -m baseline.launch \
  --config configs/baseline_8h100_seed1.json --mode baseline --output /absolute/durable/run-directory
```

The full GPT is compiled without controller sensor hooks. The earlier full-compilation OOM occurred in the instrumented path after Dynamo recompilation-limit fallback materialized eager vocabulary-loss intermediates; that failure did not establish that the pure compiled baseline needs smaller microbatches. Compilation runs on ordinary prescribed forwards, without synthetic training warmup.

Every rank writes its own optimizer/RNG/data checkpoint because Muon momentum is sharded. At step1000 all eight rank checkpoints are additionally retained under the configured `preserved_checkpoint_dir/step1000`, with a manifest, barrier, and filesystem sync before training continues. A rank0 file alone contains complete GPT weights but cannot restore the full eight-rank optimizer state. The periodic rolling checkpoints and final3250 checkpoints remain in the run output directory.

Measure steady training throughput from consecutive `event=train` wall-clock timestamps after compilation (e.g. steps11–50); exclude intervals crossing validation or checkpoint events. Report complete wall time separately, including compilation, evaluation, and checkpoint I/O. No approximate 11-minute completion claim is warranted until this execution's actual timing is measured.

Observed first-run qualification (seed1, lease53480b53): steps11–50 median0.1627754s and mean0.1628170s; peakallocation30,137,002,496bytes (~28.07GiB). This is ~531s projected pure3250-step training, excluding compilation, validation, and checkpoints. The complete run remains the authoritative elapsed-time measurement. Source archive SHA256: `97c0c32e904439d5e9764fcdc7edfa16d72932a68427d79f68d429e189f04c59`.

Completed fixed endpoints: seed0 validation3.2782857418, wall686.421s; seed1 validation3.2787988186, wall689.642s. Both ran3250 prescribed iterations. Mean validation3.2785422802; two-run acceptance margin0.002062, below0.004, so these controls alone do not satisfy the stated public acceptance formula. Finalshards and compactrecord hashes are retained under `/scratch/checkpoints/learnable-weight-decay/pr323_fast8_seed{0,1}/final3250`; local summaries are in `runs/fast8_seed{0,1}`.
