"""CPU-only accessibility diagnostic for preselected binary gauges.

Measures an edit-space restriction, not the best attainable preservation loss.
No model execution or fitting is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from rewriter.algebra import Anchors, Factors
from rewriter.state_bank import StateBankReader
from scripts.evaluate_many_channel_gauge import reconstruct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-bank', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--selection-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if digest(args.selection) != args.selection_sha256:
        raise ValueError('selection changed')
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    start = time.perf_counter()
    selection = json.loads(args.selection.read_text())
    reader = StateBankReader(args.state_bank, verify_once=True, mmap_model=True)
    results = []
    for sid in sorted({row['state_id'] for row in selection['rows']}):
        rows = [row for row in selection['rows'] if row['state_id'] == sid]
        record = reader.read(sid, device='cpu', model_device='cpu')
        if any(row['state_sha256'] != record.assert_verified() for row in rows):
            raise ValueError('checkpoint differs from selection')
        edited = reconstruct(record.model, rows)
        matrices = []
        vector_sq = 0.
        for name, value in edited.items():
            if value is record.model[name]:
                continue
            delta = value.double() - record.model[name].double()
            total = float(delta.square().sum())
            if delta.ndim == 1:
                if name not in record.snapshot.vectors:
                    raise ValueError('gauge edits an unobserved vector')
                vector_sq += total
                continue
            anchors = record.snapshot.matrices[name].anchors
            # FP32 saved anchors have small orthogonality error. Re-QR in FP64
            # preserves their spans and makes this an orthogonal projection.
            a = torch.linalg.qr(anchors.A.double(), mode='reduced').Q
            b = torch.linalg.qr(anchors.B.double(), mode='reduced').Q
            factors = Factors.project_dense(delta, Anchors(a, b))
            projected = factors.dense()
            residual_sq = float((delta - projected).square().sum())
            accessible_sq = float(factors.norm_sq())
            if abs(total-accessible_sq-residual_sq) > 1e-10*max(total, 1.):
                raise AssertionError('orthogonal decomposition mismatch')
            matrices.append(dict(name=name, edit_squared_norm=total,
                                 accessible_squared_norm=accessible_sq,
                                 outside_squared_norm=residual_sq))
        total_matrix = sum(x['edit_squared_norm'] for x in matrices)
        outside = sum(x['outside_squared_norm'] for x in matrices)
        results.append(dict(state_id=sid, state_sha256=rows[0]['state_sha256'],
                            matrix_outside_energy_fraction=outside/total_matrix,
                            whole_edit_outside_energy_fraction=outside/(total_matrix+vector_sq),
                            vector_edit_squared_norm=vector_sq, matrices=matrices))
    payload = dict(results=results, selection_sha256=args.selection_sha256,
                   source_sha256=digest(Path(__file__)),
                   elapsed_seconds=time.perf_counter()-start,
                   gpt_forwards=0, gpt_backwards=0,
                   interpretation='Distance of these selected edits from fixed-anchor action spans; not a lower bound on achievable preservation or evidence that other anchored edits cannot work.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps({k:v for k,v in payload.items() if k != 'results'}))
    print(json.dumps([{k:v for k,v in r.items() if k != 'matrices'} for r in results]))


if __name__ == '__main__':
    main()
