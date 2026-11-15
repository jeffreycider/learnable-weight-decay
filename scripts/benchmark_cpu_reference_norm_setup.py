"""Measure the removable norm-setup cost, without running or allocating GPT.

Uses mmap checkpoint tensors, FP32 conversion, and the actual FP64 norm routines. This excludes
teacher-state copies, teacher checksums, Chisel and GPT computation; it is not
an end-to-end speedup measurement. Concurrent local training may affect timings.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from rewriter.edit_scope import editable_names, editable_weight_norm_sq
from rewriter.grouped_norm import GroupedNormObjective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    state = torch.load(args.state, map_location='cpu', mmap=True, weights_only=False)
    # Parameter inventory from the serialized observation excludes rotary buffers.
    observation = state['observation']
    from rewriter.prefix_cache import unpack_record
    snapshot = unpack_record(observation, device='cpu').snapshot
    names = set(snapshot.matrices) | set(snapshot.vectors)
    parameters = {n: v for n, v in state['model'].items() if n in names}
    assert parameters.keys() == names
    stored_dtypes = sorted({str(v.dtype) for v in parameters.values()})
    conversion_start = time.perf_counter()
    parameters = {n: v.float() for n, v in parameters.items()}
    conversion_seconds = time.perf_counter()-conversion_start
    rows = []
    for repeat in range(4):
        start = time.perf_counter()
        base = editable_weight_norm_sq(parameters, parameters.keys())
        after_base = time.perf_counter()
        grouped = GroupedNormObjective(parameters, zero_reference_policy='freeze')
        after_groups = time.perf_counter()
        editable = editable_names(parameters, 'grouped') - grouped.frozen_parameter_names
        trust = editable_weight_norm_sq(parameters, editable)
        end = time.perf_counter()
        rows.append(dict(repeat=repeat, base_seconds=after_base-start,
                         group_seconds=after_groups-after_base,
                         trust_seconds=end-after_groups, total_seconds=end-start,
                         base_squared=float(base), trust_squared=float(trust)))
    result = dict(scope=__doc__, state=str(args.state.resolve()),
                  parameter_elements=sum(v.numel() for v in parameters.values()),
                  threads=1, stored_dtypes=stored_dtypes,
                  one_time_fp32_conversion_seconds=conversion_seconds, samples=rows,
                  warm_median_seconds=statistics.median(r['total_seconds'] for r in rows[1:]),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
