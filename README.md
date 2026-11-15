# Archived research

Useful research retained separately from the supported Chisel harness.

| Area | Sources |
|---|---|
| SOAP hypergradient variants and pinned optimizers | `rewriter/soap*`, `scripts/*soap*`, `baseline/record46`, `baseline/pr395` |
| Persistent CUDA kernels and numerical experiments | `rewriter/kernels`, `rewriter/persistent*`, `rewriter/compensated*` |
| Storage, prefetch and distributed private-state prototypes | `rewriter/*prefetch*`, `rewriter/*stager*`, `rewriter/*probe_store*` |
| Reusable algebra, ranking losses and diagnostics | `rewriter/exact_gauges.py`, `rewriter/tiled*`, benchmark scripts |
| Published MuonH reference | `baseline/muonh` |

Local import dependencies and focused tests are retained. Some experiments still expect their original datasets or launch environment; this branch is a research archive, not an alternative production entrypoint. Obsolete trainers that are neither selected references nor dependencies were removed. Run artifacts and credentials are not included. The supported implementation lives on the repair PR ladder.
