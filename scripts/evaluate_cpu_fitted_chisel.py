"""Finite CPU evaluation of a saved Chisel action, on an existing TRAIN panel.

This evaluates the saved action, not the analytic teaching edit. Analytically
fitted output-head controls are explicitly identified in the result. The reference
is a fresh unedited CPU FP32 GPT forward, not cached GPU logits. No fitting,
validation access, or optimizer update occurs. CPU results do not qualify CUDA.
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
from rewriter.adaptive_coordinate_chisel import AdaptiveCoordinateChisel
from rewriter.adaptive_sensing import query_weight_cores
from rewriter.edit_scope import editable_names, editable_weight_norm_sq, scoped_snapshot
from rewriter.gpt_precision import configure_precision, fp32_head_inputs
from rewriter.grouped_norm import GroupedNormObjective
from rewriter.prefix_cache import unpack_record
from rewriter.rank_preservation import topk_preservation_flags
from scripts.fit_cpu_edit_teacher import physical_action


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def source_digests():
    root = Path(__file__).resolve().parents[1]
    names = ('scripts/evaluate_cpu_fitted_chisel.py', 'scripts/fit_cpu_edit_teacher.py',
        'baseline/model.py', 'rewriter/adaptive_coordinate_chisel.py',
        'rewriter/grouped_coordinate_chisel.py',
        'rewriter/coordinate_chisel.py', 'rewriter/adaptive_sensing.py',
        'rewriter/adaptive_qr_reuse.py',
        'rewriter/adaptive_action.py', 'rewriter/adaptive_preservation.py',
        'rewriter/edit_scope.py', 'rewriter/grouped_norm.py',
        'rewriter/prefix_cache.py', 'rewriter/gpt_precision.py', 'rewriter/rank_preservation.py')
    return {name: digest(root/name) for name in names}


@torch.no_grad()
def load_action(state, checkpoint, state_sha256):
    """Recreate the exact diagnostic architecture; refuse silent conversions."""
    meta = checkpoint['metadata']
    if meta['schema'] != 'cpu_edit_teacher_fit_v1' or meta['state_sha256'] != state_sha256:
        raise ValueError('learned checkpoint/state identity mismatch')
    if not isinstance(checkpoint['step'], int) or checkpoint['step'] < 0:
        raise ValueError('invalid learned update counter')
    snapshot = unpack_record(state['observation'], device='cpu').snapshot
    names = set(snapshot.matrices) | set(snapshot.vectors)
    parameters = {n: v.detach().float() for n, v in state['model'].items() if n in names}
    grouped = GroupedNormObjective(parameters, zero_reference_policy='freeze')
    editable = editable_names(parameters, 'grouped') - grouped.frozen_parameter_names
    snapshot = scoped_snapshot(snapshot, editable_weight_norm_sq(parameters, editable),
                               meta['radius_fraction'], scope='grouped')
    if float(snapshot.radius) != meta['radius']:
        raise ValueError('action radius differs from saved training recipe')
    actor_class = AdaptiveCoordinateChisel
    if meta.get('diagnostic_control') in ('functional_group_head_lstsq','functional_group_head_gradient'):
        from rewriter.grouped_coordinate_chisel import GroupedCoordinateChisel
        actor_class = GroupedCoordinateChisel
    with torch.random.fork_rng(devices=[]):
        actor = actor_class(rank=8, trajectory_dim=400, model_dim=256,
            num_layers=2, latent_tokens=8, fresh_weight_core=True,
            factor_precision='fp64', residual_precision='fp32')
        if actor.configuration() != meta['architecture']:
            raise ValueError('unsupported learned checkpoint architecture')
        private = actor.make_private(snapshot,
            state_key=('cpu-teaching', 0, state['completed_steps'], state_sha256),
            seed=meta['args']['private_seed'])
    actor.load_state_dict(checkpoint['actor'], strict=True)
    private.load_state_dict(checkpoint['private'], strict=True)
    actor.eval()
    raw = actor(snapshot, private, weight_cores=query_weight_cores(parameters, private))
    action, norm, guard = physical_action(raw, grouped, snapshot)
    return parameters, grouped, action, dict(raw_action_norm=float(norm), guard=float(guard))


@torch.no_grad()
def stored_edits(parameters, action):
    """Match FP64 edit addition followed by a single original-storage cast."""
    result = {}
    for name, factor in action.matrices.items():
        delta = factor.dense()
        if name == 'embed.weight':
            delta = delta.T
        base = parameters[name]
        if delta.shape != base.shape:
            raise ValueError('stored edit orientation mismatch: ' + name)
        result[name] = (base.double() + delta.double()).to(base.dtype)
    for name, delta in action.vectors.items():
        base = parameters[name]
        if delta.shape != base.shape:
            raise ValueError('vector edit shape mismatch: ' + name)
        result[name] = (base.double() + delta.double()).to(base.dtype)
    if any(not x.isfinite().all() for x in result.values()):
        raise ValueError('nonfinite stored edit')
    return result


@torch.no_grad()
def logit_metrics(teacher, student, *, k=100, chunk_tokens=128):
    if teacher.shape != student.shape or teacher.ndim < 2:
        raise ValueError('matching token/vocabulary arrays required')
    teacher = teacher.reshape(-1, teacher.shape[-1])
    student = student.reshape_as(teacher)
    if not teacher.isfinite().all() or not student.isfinite().all():
        raise ValueError('nonfinite logits')
    flags = topk_preservation_flags(teacher, student, k=k)
    kl_sum, kl_max, raw_min = 0., 0., float('inf')
    for t, s in zip(teacher.split(chunk_tokens), student.split(chunk_tokens)):
        logp, logq = t.double().log_softmax(-1), s.double().log_softmax(-1)
        raw = (logp.exp() * (logp - logq)).sum(-1)
        raw_min = min(raw_min, float(raw.min()))
        kl = raw.clamp_min(0)
        kl_sum += float(kl.sum())
        kl_max = max(kl_max, float(kl.max()))
    return dict(tokens=len(teacher), kl_sum=kl_sum, max_token_kl=kl_max,
        raw_min_token_kl=raw_min, different_logits=int((teacher != student).sum()),
        max_abs_logit_difference=float((teacher-student).abs().max()),
        **{name: int(value.sum()) for name, value in flags.items()})


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('state', 'checkpoint', 'panel', 'panel-metadata', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--sequences', type=int, default=16)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    torch.set_num_threads(2)
    configure_precision('fp32')
    started = time.perf_counter()
    metadata = json.loads(a.panel_metadata.read_text())
    state_sha, panel_sha = digest(a.state), digest(a.panel)
    if state_sha != metadata['state_sha256'] or panel_sha != metadata['sha256']:
        raise ValueError('panel/state checksum mismatch')
    state = torch.load(a.state, map_location='cpu', mmap=True, weights_only=False)
    checkpoint = torch.load(a.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    panel = torch.load(a.panel, map_location='cpu', mmap=True, weights_only=False)
    if (panel['metadata']['role'] != 'train' or panel['metadata']['state_sha256'] != state_sha
            or panel['metadata']['sequence_ids'] != metadata['sequence_ids']):
        raise ValueError('requires matching existing TRAIN panel')
    if not 0 < a.sequences <= len(panel['inputs']):
        raise ValueError('invalid sequence count')
    parameters, grouped, action, action_info = load_action(state, checkpoint, state_sha)
    edited = stored_edits(parameters, action)
    norm_metrics = grouped.evaluate({**parameters, **edited}).metrics()
    base = state['model']
    depth = len({n.split('.')[1] for n in base if n.startswith('blocks.')})
    model = GPT(base['embed.weight'].shape[0], depth, base['embed.weight'].shape[1])
    model = model.float().eval().requires_grad_(False)
    model.load_state_dict(base, strict=True)
    live = dict(model.named_parameters())
    rows = []
    for i, sequence in enumerate(panel['inputs'][:a.sequences].split(1)):
        for name in edited:
            live[name].copy_(base[name])
        z = model.proj(fp32_head_inputs(model, sequence)).float()
        teacher = 15 * z * (z.square() + 15**2).rsqrt()
        for name, value in edited.items():
            live[name].copy_(value)
        z = model.proj(fp32_head_inputs(model, sequence)).float()
        student = 15 * z * (z.square() + 15**2).rsqrt()
        rows.append(dict(sequence_id=metadata['sequence_ids'][i], **logit_metrics(teacher, student)))
        print(json.dumps(rows[-1], allow_nan=False), flush=True)
    tokens = sum(r['tokens'] for r in rows)
    current_sources = source_digests()
    training_sources = checkpoint['metadata']['source_sha256']
    drift = {name: dict(training=expected, evaluation=current_sources[name])
             for name, expected in training_sources.items()
             if name in current_sources and expected != current_sources[name]}
    result = dict(scope=__doc__, state_sha256=state_sha, panel_sha256=panel_sha,
        checkpoint_sha256=digest(a.checkpoint), source_sha256=current_sources,
        training_source_sha256=training_sources, changed_common_source_files=drift,
        training_args=checkpoint['metadata']['args'],
        diagnostic_control=checkpoint['metadata'].get('diagnostic_control'),
        head_gradient_updates=checkpoint['metadata'].get('head_gradient_updates',0),
        parent_checkpoint_sha256=checkpoint['metadata'].get('parent_sha256'),
        learned_updates=checkpoint['step'], rows=rows, tokens=tokens,
        mean_token_kl=sum(r['kl_sum'] for r in rows)/tokens,
        max_token_kl=max(r['max_token_kl'] for r in rows),
        strict_top100_changed=sum(r['strict_topk_changed'] for r in rows),
        literal_argmax_changed=sum(r['literal_argmax_changed'] for r in rows),
        norm_metrics=norm_metrics, action=action_info,
        kl_convention='FP64 softmax KL(reference||edited), negative roundoff clamped to zero; raw minimum retained per sequence',
        torch_version=torch.__version__, cpu_threads=torch.get_num_threads(),
        gpt_forwards=2*a.sequences, gpt_backwards=0, query_access=False,
        elapsed_seconds=time.perf_counter()-started)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
