# Checkpoint contents

**Each trajectory record describes the weights used by one training pass and the gradients from that pass.** `000053` means training iteration 53, before optimizer update 53. Resume checkpoints are separate: `checkpoint_000100_rank0.pt` resumes after 100 completed updates.

```text
run_name/
  hyperparameters.yaml       # resolved configuration; recipe.yaml is the same recipe
  sources.yaml               # implementation hashes
  source.tar.gz              # implementation and recipes
  packages.yaml              # installed dependency versions
  environment.yaml
  data_identity.json         # FineWeb manifest, when applicable
  trajectory/
    000053/
      state.pt
      state.ready.json       # checksum and byte count; written after state.pt
      activation_rank0000.pt
      activation_rank0000.ready.json
      ...                    # one activation shard per training rank
  checkpoint_000100_rank0.pt  # restart state, including RNG and Chisel state
  latest_rank0.pt
  metrics.jsonl
```

| Field | Saved information |
|---|---|
| Hyperparameters | Architecture, initialization seed, data order, augmentation/mixing seeds, optimizer groups, LR schedules, precision, capture frequency |
| `weights` | Every model parameter and buffer at the forward-pass weights |
| `gradients` | Every parameter's ordinary task gradient, after global reduction and before optimizer mutation |
| `optimizer` | Complete base optimizer state, including momentum, Adam moments and current learning rates |
| `raw_radii` | Actual norm of each MuonH matrix; MuonH uses these tensor-specific radii, not one global radius hyperparameter |
| `activity` | Global coordinate RMS of linear inputs and ordinary output adjoints |
| `current` | Data cursor and RNG state before this pass |
| `next_data` | Cursor for the immediately following prescribed batch |
| Activation shards | Unprojected input/output vectors and output adjoints for selected hidden linear operations and residual blocks; aligned sample IDs, rank and gradient units |

**Activation capture is explicit in YAML.** `chisel.activation_records.setup.samples: 64` stores 64 positions globally, distributed over ranks and spread over each rank's batch. A sample ID identifies `(rank, example within the rank's batch, token position)`. Set `samples: null` to save every position at the selected modules. These are actual vectors, not fixed random projections. Current Chisel inputs still use the RMS bank; the raw vectors support future encoders without redoing the capture.

**Earlier v1 records contain RMS statistics, not raw activation vectors.** They remain usable by the current Chisel model. The new v2 cohort stores raw activation shards. No converter can recover those vectors from RMS alone.

**Inputs, labels and logits are reconstructible rather than duplicated in each checkpoint.** CIFAR uses a content-hashed dataset, explicit sampling/augmentation/mixing seeds and cursor. FineWeb uses content-hashed shards and its cursor; its prescribed order has no shuffle seed. Recreating intermediates needs the recorded source/runtime and a model pass. The code does not claim that a summary stores every possible future sensor. All-position capture stores the selected modules' tensors, not every internal attention intermediate.

**Every rank retains its own replay state.** A v2 state file has the rank-0 cursor and full reduced gradients; activation shards carry each rank's cursor and RNG state. Adjoints retain the actual backward reduction: local-batch mean or sum, explicitly labeled. They are not individual output-token gradients.

**R2 is a verified backup of the shared-filesystem artifacts.** Immutable uploads check local SHA256, refuse destination collisions, read the full object back, then write a success receipt. They never delete local files. The metadata backup also saves YAML and source archives. `python -m chisel.catalog recipes/catalog.yaml` includes only nonterminal records with every declared activation shard present and verified.

**Offline replay verifies source identity.** `recipes/source_compatibility.yaml` permits only listed old-hash/new-hash pairs qualified for equivalent default behavior. It is not a blanket bypass. Offline restart pins immutable private-probe versions. Distributed restart additionally requires a generation manifest covering every rank; changing world size needs explicit resharding.

## Imported PR267 recordings

`python -m chisel.import_recording import.yaml` merges all eight losslessly compressed rank shards after checking their SHA256 receipts. The import configuration specifies the original run, output directory, shared YAML template, dataset pattern and step list. Import checks parameter coverage and reconstructs the exact saved global token batch before publishing a new receipt.

Imported records explicitly identify the original PR267 source and each shard hash. Their `source_hashes` pin the qualified replay implementation, **not** the original producer. Original residual samples and sampled head logits remain in `legacy_samples`; missing output-adjoint RMS, raw linear input/adjoint samples and Python/NumPy RNG are declared unavailable. Native v2 records retain the richer observation contract.

A100 qualification on seed 0, step 1000 reproduced the published forward loss, MuonH direction and sphere update exactly. The replayed batch matched all 512 × 1024 stored tokens. This establishes compatibility for the tested state, not equivalence of every imported state.
