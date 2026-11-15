"""Matched local saved-state complete updates; prior fidelity gate is required."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from baseline.execution import enable_deterministic_execution
from baseline.model import GPT
from rewriter.state_bank import StateBankReader
from rewriter.teacher_cache import TeacherCacheReader
from rewriter.offline_systems import CachedBaseNormPreservation, VerifiedStateConstants, select_panel_blocks
from rewriter.accelerated_grouped_norm import AcceleratedGroupedNormObjective
from rewriter.chisel_factory import build_chisel
from rewriter.edit_scope import scoped_snapshot
from rewriter.tokenwise_reference import TokenwiseConfig, TokenwiseDual
from rewriter.meta_optimizer import build_meta_adam
from rewriter.graph_tokenwise import TokenwiseThinGradientGraph
from rewriter.graph_tokenwise_experimental import CompensatedDxTokenwiseGraph, require_compensated_dx_environment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('policy', 'qualification', 'output'): p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    proof = json.loads(a.qualification.read_text())
    assert all(proof['gates'].values()) and proof['experimental_dx']
    assert hashlib.sha256(a.policy.read_bytes()).hexdigest() == proof['policy_sha256']
    require_compensated_dx_environment()
    enable_deterministic_execution(); torch.set_num_threads(1)
    a.output.mkdir(parents=True, exist_ok=False)
    import wandb
    run = wandb.init(project='chisel', mode='offline', dir=str(a.output), config={'qualification': str(a.qualification)})
    device = torch.device('cuda:0')
    reader = StateBankReader('/scratch/checkpoints/learnable-weight-decay/chisel-state-bank-seed0-v1', verify_once=True, mmap_model=True)
    cache = TeacherCacheReader('/scratch/checkpoints/learnable-weight-decay/chisel-teacher-cache1000-fp32-eps23-v2', verify_once=True, mmap=True)
    engine = CachedBaseNormPreservation(GPT(**reader.manifest['provenance']['model_config']).to(device).requires_grad_(False),
        microbatch_sequences=16, teacher_cache_device=device, kl_dtype=torch.float64, gpt_precision='fp32')
    constants = VerifiedStateConstants(engine, AcceleratedGroupedNormObjective)
    saved = torch.load(a.policy, map_location='cpu', weights_only=False)
    actor = build_chisel(saved['metadata']['architecture']).to(device)
    forward = torch.compile(actor, fullgraph=True, dynamic=False)
    params = list(actor.parameters())
    optimizer = build_meta_adam(actor, lr=1e-6, representation_lr_multiplier=10)
    cfg = TokenwiseConfig(**saved['metadata']['tokenwise_objective'])
    ds = saved['primal_dual']; dual = TokenwiseDual(ds['multipliers'], config=cfg, dual_lr=ds['dual_lr']); dual.load_state_dict(ds)
    states = [6, 251, 499, 999]
    def read(state):
        return reader.read(state, model_device='cpu', device=device), select_panel_blocks(cache.read(state, 'train'), range(4))
    # All file validation/state-constant first visits are outside warm timings.
    for state in states:
        record, panel = read(state); constants.load(cache, record)
        del record, panel
    rows = []; reference_gradient = reference_delta = None
    for label, graph_type in [('strict', TokenwiseThinGradientGraph), ('hybrid', CompensatedDxTokenwiseGraph),
                              ('hybrid', CompensatedDxTokenwiseGraph), ('strict', TokenwiseThinGradientGraph)]:
        actor.load_state_dict(saved['controller']); optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
        actor.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
        record, panel = read(states[0]); constants.load(cache, record); grouped = constants.grouped()
        snapshot = scoped_snapshot(record.snapshot, constants.radius_squared(grouped), .01, scope='grouped')
        graph = graph_type(actor, forward, snapshot, engine, grouped, panel.inputs.to(device), None, dual, states[0],
            state_weight=.25, teacher_features=panel.features.to(device, copy=True), dynamic_zero_groups=True, token_config=cfg)
        setup = time.perf_counter(); graph.capture(warmups=1); graph.replay(); torch.cuda.synchronize()
        setup = time.perf_counter()-setup
        accum = [torch.zeros_like(p) for p in params]
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start = time.perf_counter()
        for state in states:
            record, panel = read(state); graph.load_checkpoint(cache, record, panel, dual, state_constants=constants)
            graph.accumulate_into(accum); graph.check_finite()
        for p, g in zip(params, accum): p.grad = g
        # Gradient export is excluded identically; clipping/Adam remain timed.
        torch.cuda.synchronize(); accumulated_seconds = time.perf_counter()-start
        gradient = torch.cat([p.grad.flatten() for p in params]).detach().cpu()
        start = time.perf_counter()
        norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True); optimizer.step(); torch.cuda.synchronize()
        elapsed = accumulated_seconds + time.perf_counter()-start
        policy = {name: value.detach().cpu().clone() for name, value in actor.state_dict().items()}
        delta = torch.cat([(policy[name]-saved['controller'][name]).flatten() for name in policy])
        if reference_gradient is None: reference_gradient, reference_delta = gradient, delta
        relative = lambda x,y: float((x.double()-y.double()).norm()/y.double().norm().clamp_min(1e-300))
        row = dict(backend=label, seconds=elapsed, capture_and_first_replay_seconds=setup,
                   gradient_relative=relative(gradient, reference_gradient), adam_delta_relative=relative(delta, reference_delta),
                   gradient_norm=float(norm), peak_bytes=torch.cuda.max_memory_allocated())
        assert row['gradient_relative'] <= .003 and row['adam_delta_relative'] <= .003
        rows.append(row); print(json.dumps(row), flush=True)
        graph.close(); del graph, accum, record, panel, gradient, delta, policy
        actor.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
    strict = statistics.median(x['seconds'] for x in rows if x['backend']=='strict')
    hybrid = statistics.median(x['seconds'] for x in rows if x['backend']=='hybrid')
    result = dict(scope='One local four-checkpoint x64 saved-Adam update, weight1/4; ABBA. Warm state read/teacher head/decomposition/clip/Adam included; capture, first-visit validation, gradient export excluded; no distributedSUM or dual ascent',
        policy_sha256=proof['policy_sha256'], qualification=str(a.qualification), rows=rows,
        strict_seconds=strict, hybrid_seconds=hybrid, time_reduction_percent=100*(1-hybrid/strict),
        dual_unchanged=dual.state_dict()==ds, default_torch_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        default_torch_cudnn_tf32=torch.backends.cudnn.allow_tf32)
    (a.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    run.log({'strict_seconds': strict, 'hybrid_seconds': hybrid}); run.finish()


if __name__ == '__main__': main()
