"""CPU-only finite check of a proposed teaching edit on existing TRAIN inputs.

The teacher here is the unedited CPU FP32 model, not cached CUDA logits. This
checks this execution only; it is neither GPU qualification nor a learned edit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from baseline.model import GPT
from rewriter.gpt_precision import configure_precision, fp32_head_inputs
from rewriter.rank_preservation import topk_preservation_flags
from scripts.evaluate_adaptive_gauge_precision import reconstruct_cpu


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('state', 'panel', 'panel-metadata', 'selection', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--sequences', type=int, default=16)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    torch.set_num_threads(2)
    configure_precision('fp32')
    start = time.perf_counter()
    metadata = json.loads(a.panel_metadata.read_text())
    assert digest(a.state) == metadata['state_sha256']
    assert digest(a.panel) == metadata['sha256']
    state = torch.load(a.state, map_location='cpu', mmap=True, weights_only=False)
    panel = torch.load(a.panel, map_location='cpu', mmap=True, weights_only=False)
    assert panel['metadata']['role'] == 'train'
    assert panel['metadata']['state_sha256'] == metadata['state_sha256']
    assert panel['metadata']['sequence_ids'] == metadata['sequence_ids']
    inputs = panel['inputs'][:a.sequences]
    assert 0 < a.sequences <= len(panel['inputs'])
    selected = json.loads(a.selection.read_text())['rows']
    assert all(r['state_id'] == state['completed_steps'] and r['state_sha256'] == metadata['state_sha256'] for r in selected)
    base = state['model']
    dim = base['embed.weight'].shape[1]
    depth = len({n.split('.')[1] for n in base if n.startswith('blocks.')})
    model = GPT(base['embed.weight'].shape[0], depth, dim).float().eval().requires_grad_(False)
    model.load_state_dict(base, strict=True)
    edits = reconstruct_cpu(base, selected)
    assert all(torch.equal(v, edits['direct_binary'][n]) for n, v in edits['adaptive_fp64_cast'].items())
    # Equal stored tensors need no duplicate model forward.
    rows = {name: dict(different_logits=0, max_abs_logit_difference=0., strict_top100_changes=0,
                       literal_argmax_changes=0) for name in ('direct_binary', 'adaptive_fp32')}
    params = dict(model.named_parameters())
    original = {n: params[n].clone() for n in edits['direct_binary']}
    for completed, sequence in enumerate(inputs.split(1), 1):
        for n, value in original.items():
            params[n].copy_(value)
        logits = model.proj(fp32_head_inputs(model, sequence)).float()
        teacher = 15 * logits * (logits.square() + 15**2).rsqrt()
        for name, row in rows.items():
            for n, value in edits[name].items():
                params[n].copy_(value)
            logits = model.proj(fp32_head_inputs(model, sequence)).float()
            student = 15 * logits * (logits.square() + 15**2).rsqrt()
            row['different_logits'] += int((student != teacher).sum())
            row['max_abs_logit_difference'] = max(row['max_abs_logit_difference'], float((student-teacher).abs().max()))
            flags = topk_preservation_flags(teacher.reshape(-1, teacher.shape[-1]), student.reshape(-1, student.shape[-1]), k=100)
            row['strict_top100_changes'] += int(flags['strict_topk_changed'].sum())
            row['literal_argmax_changes'] += int((student.argmax(-1) != teacher.argmax(-1)).sum())
        print(json.dumps(dict(completed_sequences=completed, partial=rows)), flush=True)
    result = dict(scope=__doc__, state_sha256=metadata['state_sha256'], panel_sha256=metadata['sha256'],
                  selection_sha256=digest(a.selection), source_sha256=digest(__file__),
                  sequence_ids=metadata['sequence_ids'][:a.sequences], tokens=inputs.numel(),
                  cpu_threads=torch.get_num_threads(), torch_version=torch.__version__,
                  rows=rows, fp64_cast_stored_weights_equal_direct=True,
                  gpt_forwards=3*a.sequences, gpt_backwards=0, query_access=False,
                  learned_updates=0, elapsed_seconds=time.perf_counter()-start)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
