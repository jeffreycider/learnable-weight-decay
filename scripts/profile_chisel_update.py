"""Single-rank complete-step profiler for the offline Chisel trainer.

Replays exactly one rank's share of a production optimizer update: for each
listed fitting checkpoint, verified state load, cached-panel block selection
for the given epoch, compiled actor proposal, FP32 student / FP64 KL forward,
GPT backward, actor credit, then gradient clipping and one Adam step on the
accumulated gradient. The arithmetic, data, precision, microbatch and
per-checkpoint weight (1/200) are the production ones; the all-reduce across
ranks is absent (single process), and periodic audits/checkpoints are excluded.

Arms differ only in host behaviour:
  --timing sync|event|none     production synchronous phase timing, CUDA-event
                               timing resolved once per checkpoint, or none
  --deferred-kl                one FP64 readback per checkpoint (rewriter.deferred_kl)
  --kl-backend chunked|fused   reference chunked KL or the fused Triton kernel
                               (fused is NOT expected to be bit-identical; its
                               gradient difference is reported, never hidden)
Every arm reports the pre-clip flat actor gradient hash so that bit-exactness
against the first arm is checked, not assumed. ``--profile-states K`` wraps the
first K checkpoints of the last measured pass in torch.profiler and reports the
GPU-busy fraction (union of kernel/memcpy intervals over the window) and the top
kernels, which is what separates host stalls from slow kernels.

Staging: ``--stage-from BANK`` copies only the listed state files plus the bank
manifest/index/complete files into ``--state-bank`` when missing; the teacher
cache is read in place (mmap) exactly as production does. This never launches a
fit and never writes to production directories.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, shutil, statistics, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('baseline', 'rewriter', 'scripts')
            for p in sorted((root / folder).rglob('*.py'))}


def stage_states(source, target, state_ids):
    source, target = Path(source), Path(target)
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in ('manifest.json', 'index.jsonl', 'complete.json', 'available.json'):
        if (source / name).exists() and not (target / name).exists():
            shutil.copy2(source / name, target / name)
            copied.append(name)
    for state_id in state_ids:
        name = f'state_{state_id:04d}.pt'
        if not (target / name).exists():
            shutil.copy2(source / name, target / f'{name}.staging')
            os.replace(target / f'{name}.staging', target / name)
            copied.append(name)
    return copied


def select_training_blocks(panel, count, epoch, data_seed, state_id, stable_seed):
    """Mirror of train_multistate_chisel.panel_inputs block selection for role=train."""
    if count == 0 or len(panel.inputs) == count:
        return panel
    size = panel.metadata['microbatch_sequences']
    if count % size or len(panel.inputs) % count:
        raise ValueError('subset size must tile the fixed panel in complete generation microbatches')
    chunks = len(panel.inputs) // count
    cycle = max(epoch, 0) // chunks
    generator = torch.Generator().manual_seed(stable_seed('cached_panel_blocks', data_seed, state_id, cycle))
    blocks = torch.randperm(len(panel.inputs) // size, generator=generator).tolist()
    offset = (max(epoch, 0) % chunks) * (count // size)
    return panel.select_blocks(blocks[offset:offset + count // size])


def gpu_busy_from_trace(path):
    """Union of device-side intervals over the trace window, from a chrome trace."""
    trace = json.loads(Path(path).read_text())
    events = trace['traceEvents'] if isinstance(trace, dict) else trace
    device, window_start, window_end = [], None, None
    kernels = {}
    for event in events:
        if event.get('ph') != 'X':
            continue
        start, duration = event.get('ts'), event.get('dur', 0)
        if start is None:
            continue
        window_start = start if window_start is None else min(window_start, start)
        window_end = start + duration if window_end is None else max(window_end, start + duration)
        if event.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset'):
            device.append((start, start + duration))
            name = event.get('name', '?')
            kernels[name] = kernels.get(name, 0.) + duration
    device.sort()
    busy, current = 0., None
    for start, end in device:
        if current is None or start > current[1]:
            if current is not None:
                busy += current[1] - current[0]
            current = [start, end]
        else:
            current[1] = max(current[1], end)
    if current is not None:
        busy += current[1] - current[0]
    window = (window_end - window_start) if window_start is not None else 0.
    top = sorted(kernels.items(), key=lambda kv: -kv[1])[:25]
    return dict(window_seconds=window / 1e6, device_busy_seconds=busy / 1e6,
                device_busy_fraction=(busy / window) if window else None,
                top_kernels=[dict(name=n[:160], seconds=s / 1e6, fraction_of_busy=(s / busy) if busy else None) for n, s in top],
                kernel_launches=sum(1 for _ in device))


def flat_gradient(actor):
    return torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad.detach()).flatten() for p in actor.parameters()])


def tensor_digest(value):
    value = value.detach().cpu().contiguous().reshape(-1)
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _main(argv, cleanups):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--state-bank', type=Path, required=True)
    p.add_argument('--teacher-cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--policy', type=Path, help='trainer checkpoint supplying architecture, actor, Adam and options; omitted => fresh actor from --chisel-* flags')
    p.add_argument('--state-ids', type=int, nargs='+', help='explicit fitting checkpoints (one rank share); default: rank slice of the policy batch')
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--world-size', type=int, default=8)
    p.add_argument('--epoch', type=int, help='epoch for cached-panel block selection; default policy next_epoch or 0')
    p.add_argument('--checkpoint-batch', type=int, default=200, help='global checkpoint batch for the 1/N weight')
    p.add_argument('--sequences', type=int, default=64)
    p.add_argument('--microbatch', type=int, default=16)
    p.add_argument('--gpt-precision', choices=('baseline', 'fp32'), default='fp32')
    p.add_argument('--norm-backend', choices=('reference', 'fused', 'auto'), default='auto')
    p.add_argument('--timing', choices=('sync', 'event', 'none'), default='sync')
    p.add_argument('--deferred-kl', action='store_true')
    p.add_argument('--kl-backend', choices=('chunked', 'fused', 'tiled', 'tiled-softcap'), default='chunked',
                   help='tiled = rewriter.tiled_teacher_kl on FP32 softcapped teacher logits (no FP64 log-prob materialization; FP64 sum order differs); tiled-softcap = raw head outputs with the softcap and its derivative inside the kernels (APPROXIMATE backward rounding; qualification arm only)')
    p.add_argument('--prefetch', action='store_true', help='stage the next checkpoint and panel on a worker thread (rewriter.state_prefetch)')
    p.add_argument('--prefetch-panel-only', action='store_true', help='worker stages only the teacher panel (GIL-light); record load stays on the consumer')
    p.add_argument('--coalesced-observations', action='store_true', help='move the packed observation in one transfer per dtype (rewriter.coalesced_state_reader)')
    p.add_argument('--constants-cache', action='store_true', help="Astra's SHA-keyed base-norm / group-reference / radius scalar caches (rewriter.offline_systems)")
    p.add_argument('--repeats', type=int, default=2, help='measured passes after one warmup pass')
    p.add_argument('--profile-states', type=int, default=0, help='torch.profiler window over the first K checkpoints of the last measured pass')
    p.add_argument('--stage-from', type=Path, help='source bank to copy the listed state files from when absent locally')
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--eager-chisel', action='store_true')
    p.add_argument('--data-seed', type=int, default=400001)
    p.add_argument('--split-seed', type=int, default=500001)
    p.add_argument('--radius-fraction', type=float, default=.01)
    p.add_argument('--kl-tolerance', type=float, default=1e-6)
    p.add_argument('--augmented-rho', type=float, default=.01)
    p.add_argument('--dual-lr', type=float, default=.1)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--gradient-clip', type=float, default=1.)
    p.add_argument('--chisel-model-dim', type=int, default=512)
    p.add_argument('--chisel-layers', type=int, default=4)
    p.add_argument('--chisel-patch-size', type=int, default=64)
    p.add_argument('--shrinkage-coefficient', type=float, help='objective shrinkage coefficient; default = saved policy option (1.0 if absent). The repair stage uses 0.')
    p.add_argument('--label', default='')
    args = p.parse_args(argv)
    if args.repeats < 1 or args.sequences < 1 or args.microbatch < 1 or args.checkpoint_batch < 1:
        p.error('positive repeats, sequences, microbatch and checkpoint batch required')
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA requested but unavailable')
    if int(os.environ.get('RANK', '0')) == 0:
        args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'summary.json').exists():
        p.error('output already holds a finished measurement; use a fresh directory')
    before = source_hashes()

    import scripts.train_multistate_chisel as train
    from baseline.model import GPT
    from baseline.execution import enable_deterministic_execution
    from rewriter.gpt_precision import configure_precision
    from rewriter.state_bank import StateBankReader
    from rewriter.teacher_cache import TeacherCacheReader
    from rewriter.function_preservation import FunctionalPreservation
    from rewriter.grouped_norm import GroupedNormObjective
    from rewriter.chisel_factory import build_chisel, chisel_architecture
    from rewriter.meta_optimizer import build_meta_adam
    from rewriter.primal_dual import PrimalDual
    from rewriter.step_timing import install
    from rewriter.deferred_kl import mean_kl_and_backward_deferred

    import torch.distributed as dist
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    # Explicit index: torch 2.11 rejects a bare 'cuda' device in set_device.
    device = torch.device('cuda', local_rank) if args.device == 'cuda' else torch.device('cpu')
    if device.type == 'cuda':
        enable_deterministic_execution()
        torch.cuda.set_device(device)
    torch.set_num_threads(1)
    configure_precision(args.gpt_precision)
    if world > 1:
        dist.init_process_group(backend='nccl' if device.type == 'cuda' else 'gloo')
        if args.state_ids is None:
            args.rank, args.world_size = rank, world
    distributed_note = ('full checkpoint batch across %d ranks: production sharding, SUM all-reduce, shared Adam, gathered dual step' % world
                        if world > 1 else 'one rank share of one optimizer update; no all-reduce')

    saved = None
    if args.policy:
        saved = torch.load(args.policy, map_location='cpu', weights_only=False)
        options = saved['metadata']['options']
        if options.get('gpt_precision', 'baseline') != args.gpt_precision:
            p.error('policy precision differs from --gpt-precision')
        if args.shrinkage_coefficient is None:
            args.shrinkage_coefficient = float(options.get('shrinkage_coefficient', 1.))
        for key, value in (('kl_tolerance', 'kl_tolerance'), ('augmented_rho', 'augmented_rho'), ('dual_lr', 'dual_lr'),
                           ('radius_fraction', 'radius_fraction'), ('data_seed', 'data_seed'), ('split_seed', 'split_seed'),
                           ('lr', 'lr'), ('gradient_clip', 'gradient_clip')):
            if key in options:
                setattr(args, value, options[key])
    if args.shrinkage_coefficient is None:
        args.shrinkage_coefficient = 1.
    if not math.isfinite(args.shrinkage_coefficient) or args.shrinkage_coefficient < 0:
        p.error('nonnegative finite shrinkage coefficient required')
    reader_kwargs = dict(verify_once=True, mmap_model=True)
    if args.stage_from:
        ids_for_staging = args.state_ids
        if ids_for_staging is None:
            p.error('--stage-from requires explicit --state-ids')
        staged = stage_states(args.stage_from, args.state_bank, ids_for_staging)
    else:
        staged = []
    if args.coalesced_observations:
        from rewriter.coalesced_state_reader import CoalescedStateBankReader
        reader = CoalescedStateBankReader(args.state_bank, **reader_kwargs)
    else:
        reader = StateBankReader(args.state_bank, **reader_kwargs)
    cache = TeacherCacheReader(args.teacher_cache, verify_once=True, mmap=True, require_complete=(args.teacher_cache / 'manifest.json').exists())
    provenance = reader.manifest['provenance']
    model_config = provenance['model_config']
    train_states = reader.manifest['split']['train']
    epoch = args.epoch
    batch_states = None
    if saved is not None:
        saved_world = saved['metadata']['world_size']
        if epoch is None:
            epoch = saved['next_epoch']
        if args.state_ids is None:
            batches = (train.sharded_epoch_batches(train_states, epoch, args.data_seed, args.checkpoint_batch, saved_world)
                       if options.get('fixed_state_shards') else train.epoch_batches(train_states, epoch, args.data_seed, args.checkpoint_batch))
            batch_states = batches[saved['next_batch']]
            states = batch_states[args.rank::args.world_size]
        else:
            states = list(args.state_ids)[rank::world] if world > 1 else list(args.state_ids)
            batch_states = list(args.state_ids)
    else:
        epoch = 0 if epoch is None else epoch
        batch_states = list(args.state_ids) if args.state_ids else train_states
        states = batch_states[args.rank::args.world_size] if world == 1 and not args.state_ids else batch_states[rank::world]
    if any(s not in train_states for s in states):
        p.error('only fitting checkpoints may be profiled; never held-out states')
    if world > 1 and len(batch_states) % world:
        p.error('distributed run needs a checkpoint batch divisible by the world size')

    model = GPT(**model_config).to(device).requires_grad_(False)
    engine_class = FunctionalPreservation
    if args.constants_cache:
        from rewriter.offline_systems import CachedBaseNormPreservation, VerifiedStateConstants
        engine_class = CachedBaseNormPreservation
    engine = engine_class(model, microbatch_sequences=args.microbatch, teacher_cache_device=device,
                          kl_dtype=torch.float64, gpt_precision=args.gpt_precision)
    import rewriter.function_preservation as preservation
    original_kl = preservation.chunked_teacher_kl
    cleanups.append(lambda: setattr(preservation, 'chunked_teacher_kl', original_kl))
    if args.kl_backend == 'fused':
        from rewriter.accelerated_kl import fused_teacher_kl
        preservation.chunked_teacher_kl = lambda logits, log_probs, chunk_tokens=128: fused_teacher_kl(logits, log_probs)
    original_view_forward = preservation._LogitsView.forward
    cleanups.append(lambda: setattr(preservation._LogitsView, 'forward', original_view_forward))
    if args.kl_backend in ('tiled', 'tiled-softcap'):
        from rewriter.tiled_teacher_kl import tiled_teacher_kl
        softcap_inside = args.kl_backend == 'tiled-softcap'
        # The teacher payload carried in TeacherBatch.log_probs is FP32 head logits for these arms
        # (softcapped for 'tiled', raw for 'tiled-softcap'); arithmetic inside the kernel stays FP64.
        preservation.chunked_teacher_kl = lambda logits, teacher_logits, chunk_tokens=128: tiled_teacher_kl(logits, teacher_logits, softcap=softcap_inside)
        engine.kl_dtype = torch.float32
        if softcap_inside:
            def raw_forward(self, inputs):
                # Identical GPT body; only the final 15*z*(z^2+225)^-1/2 is deferred into the KL kernel.
                if self.gpt_precision == 'fp32':
                    from rewriter.gpt_precision import fp32_head_inputs
                    return self.gpt.proj(fp32_head_inputs(self.gpt, inputs))
                x = self.gpt.norm1(self.gpt.embed(inputs))
                for block in self.gpt.blocks:
                    x = block(x)
                return self.gpt.proj(self.gpt.norm2(x)).float()
            preservation._LogitsView.forward = raw_forward
    if saved is not None:
        architecture = saved['metadata']['architecture']
    else:
        probe_rank = provenance['sensor_config'].get('probe_rank', provenance['sensor_config'].get('rank', 8))
        architecture = chisel_architecture(args, rank=probe_rank, trajectory_dim=(2 * model_config['num_layers'] + 1) * 2 * probe_rank)
    with torch.random.fork_rng(devices=[device.index] if device.type == 'cuda' else []):
        torch.manual_seed(200001)
        actor = build_chisel(architecture).to(device)
    if saved is not None:
        actor.load_state_dict(saved['controller'])
    initial_actor = copy.deepcopy(actor.state_dict())
    forward = actor if args.eager_chisel or device.type == 'cpu' else torch.compile(actor, fullgraph=True, dynamic=False)
    if args.norm_backend == 'auto':
        args.norm_backend = 'fused' if device.type == 'cuda' else 'reference'
    if args.norm_backend == 'fused':
        from rewriter.accelerated_grouped_norm import AcceleratedGroupedNormObjective as norm_class
    else:
        norm_class = GroupedNormObjective
    dual = PrimalDual(train_states, tolerance=args.kl_tolerance, rho=args.augmented_rho, dual_lr=args.dual_lr)
    if saved is not None and saved.get('primal_dual'):
        dual.load_state_dict(saved['primal_dual'])
    kl_backward = mean_kl_and_backward_deferred if args.deferred_kl else train.mean_kl_and_backward
    constants = None
    if args.constants_cache:
        if args.norm_backend == 'auto':
            args.norm_backend = 'fused' if device.type == 'cuda' else 'reference'
        constants = VerifiedStateConstants(engine, __import__('rewriter.accelerated_grouped_norm', fromlist=['AcceleratedGroupedNormObjective']).AcceleratedGroupedNormObjective
                                           if args.norm_backend == 'fused' else GroupedNormObjective)
    timer, restore = install(train, args.timing)
    cleanups.append(restore)
    timing_holder = [None]
    original_forward = engine._forward

    def timed_gpt_forward(parameters, inputs):
        key = 'teacher_gpt_forward_seconds' if parameters is engine.parameters else 'student_gpt_forward_seconds'
        return train.timed(timing_holder[0], key, lambda: original_forward(parameters, inputs), device)
    engine._forward = timed_gpt_forward
    diagnostics = {}

    def select_panel(panel, state_id):
        return select_training_blocks(panel, args.sequences, epoch, args.data_seed, state_id, train.stable_seed)

    prefetcher = None
    if args.prefetch or args.prefetch_panel_only:
        from rewriter.state_prefetch import StatePrefetcher
        prefetcher = StatePrefetcher(reader, cache, device=device, select_panel=select_panel, stage_record=not args.prefetch_panel_only)
        cleanups.append(prefetcher.close)

    def run_state(state_id, phase, panel_cache):
        timing_holder[0] = phase
        staged = {}
        def load():
            if prefetcher is None:
                record = reader.read(state_id, device=device, model_device='cpu')
            else:
                record, staged['panel'], staged['worker'] = prefetcher.take(state_id)
                if record is None:  # panel-only prefetch: production record path
                    record = reader.read(state_id, device=device, model_device='cpu')
            if constants is not None:
                constants.load(cache, record)
                grouped = constants.grouped()
            else:
                cache.load_engine(engine, record)
                grouped = norm_class(engine.parameters, zero_reference_policy='freeze')
            if state_id not in diagnostics:
                diagnostics[state_id] = train.anchored_norm_ceiling(record.snapshot, grouped)
            grouped.state_diagnostics = diagnostics[state_id]
            return record, grouped
        record, grouped = train.timed(phase, 'state_load_seconds', load, device)
        selection_started = time.perf_counter()
        if prefetcher is None:
            panel = train.timed(phase, 'teacher_panel_load_seconds', lambda: cache.read(state_id, 'train'), device)
            panel = select_panel(panel, state_id)
        else:
            panel = staged['panel']
            if phase is not None:
                for key, value in staged['worker'].items():
                    phase['prefetch_worker_' + key] = value
        if any(train.heldout_sequence(i, args.split_seed) for i in panel.sequence_ids):
            raise ValueError('held-out sequences cannot enter a profiled update')
        if phase is not None:
            phase['block_selection_host_seconds'] = time.perf_counter() - selection_started
        if constants is not None:
            # production build_proposal now accepts the cached radius directly (norm_squared) and the coefficient
            parameters, regularizer, stats, _, credit = train.build_proposal(
                forward, record.snapshot, engine, grouped, radius_fraction=args.radius_fraction, timings=phase,
                norm_squared=constants.radius_squared(grouped), shrinkage_coefficient=args.shrinkage_coefficient)
        else:
            parameters, regularizer, stats, _, credit = train.build_proposal(
                forward, record.snapshot, engine, grouped, radius_fraction=args.radius_fraction, timings=phase,
                shrinkage_coefficient=args.shrinkage_coefficient)
        if args.kl_backend in ('tiled', 'tiled-softcap'):
            from rewriter.function_preservation import TeacherBatch
            from rewriter.tiled_teacher_kl import teacher_logits_from_features, raw_teacher_head, validate_panel_for_tiled
            bound_panel = panel
            validate_panel_for_tiled(bound_panel, engine)  # same precision/binding/head checks, once per state, outside the KL path
            head = raw_teacher_head if args.kl_backend == 'tiled-softcap' else teacher_logits_from_features
            def provider(a, b, bound_panel=bound_panel, head=head):
                # Same exact head reconstruction as TeacherPanel.teacher_batch (head_logits), minus the FP64 log-softmax
                # (and minus the softcap for tiled-softcap). Checks ran once per state above.
                logits = head(engine, bound_panel.features[a:b])
                return TeacherBatch(bound_panel.inputs[a:b].clone(), logits, engine.teacher_id, 'past_training', bound_panel.sequence_ids[a:b])
        else:
            provider = lambda a, b: panel.teacher_batch(engine, a, b)
        measured = kl_backward(engine, parameters, regularizer, panel.inputs, kl_budget=1e-4,
                               state_weight=1 / args.checkpoint_batch, actor_credit=credit, timings=phase,
                               teacher_provider=provider, primal_dual=dual, state_id=state_id)
        if timer is not None:
            timer.resolve(device if device.type == 'cuda' else None)
        row = dict(state_id=state_id, sequence_ids=list(panel.sequence_ids), mean_kl=measured['mean_kl'],
                   objective=measured['objective'], norm_term=stats['norm_term'], global_scale=stats['global_scale'],
                   component_seconds=phase)
        del record, grouped, parameters, regularizer, credit, panel
        return row

    passes = []
    reference_digest = None
    initial_dual = dual.state_dict()
    log = (args.output / f'states_rank{rank}.jsonl').open('w')
    profile_summary = None

    def all_equal_flag(value):
        if world == 1:
            return bool(value)
        flag = torch.tensor(int(bool(value)), device=device if device.type == 'cuda' else 'cpu')
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    def max_over_ranks(value):
        if world == 1:
            return float(value)
        tensor = torch.tensor(float(value), dtype=torch.float64, device=device if device.type == 'cuda' else 'cpu')
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return float(tensor.item())
    for repeat in range(args.repeats + 1):
        actor.load_state_dict(initial_actor)
        multiplier = saved['metadata']['options'].get('representation_lr_multiplier', 1.) if saved is not None else 1.
        optimizer = build_meta_adam(actor, lr=args.lr, representation_lr_multiplier=multiplier)
        if saved is not None and saved.get('optimizer') is not None:
            optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
        dual.load_state_dict(initial_dual)
        optimizer.zero_grad(set_to_none=True)
        if world > 1:
            dist.barrier()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        profiling = repeat == args.repeats and args.profile_states > 0 and device.type == 'cuda' and rank == 0
        rows = []
        if prefetcher is not None:
            prefetcher.wait_seconds = 0.
            prefetcher.schedule(states)
        started = time.perf_counter()
        profiler = None
        for index, state_id in enumerate(states):
            if profiling and index == 0:
                from torch.profiler import profile, ProfilerActivity
                profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False, with_stack=False)
                profiler.__enter__()
                profile_started = time.perf_counter()
            phase = None if args.timing == 'none' else {}
            rows.append(run_state(state_id, phase, None))
            if profiler is not None and index + 1 == min(args.profile_states, len(states)):
                torch.cuda.synchronize(device)
                profile_seconds = time.perf_counter() - profile_started
                profiler.__exit__(None, None, None)
                trace = args.output / 'trace.json'
                profiler.export_chrome_trace(str(trace))
                profile_summary = dict(states=min(args.profile_states, len(states)), host_wall_seconds=profile_seconds,
                                       **gpu_busy_from_trace(trace))
                profiler = None
        local_gradient_digest = tensor_digest(flat_gradient(actor))
        reduce_started = time.perf_counter()
        train.aggregate_controller_gradients(actor, world)   # SUM across ranks, exactly as production
        gathered_rows = [row for local_rows in train.all_rank_objects(
            [dict(state_id=r['state_id'], mean_kl=r['mean_kl']) for r in rows], world) for row in local_rows]
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        reduce_seconds = time.perf_counter() - reduce_started
        gradient_digest = tensor_digest(flat_gradient(actor))
        adam_started = time.perf_counter()
        grad = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), args.gradient_clip, error_if_nonfinite=True))
        optimizer.step()
        dual.step([(r['state_id'], r['mean_kl']) for r in gathered_rows], expected_ids=batch_states)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        adam_seconds = time.perf_counter() - adam_started
        wall = time.perf_counter() - started
        wall_max = max_over_ranks(wall)
        # True post-update equivalence objects: parameters, Adam moments/steps, dual multipliers.
        parameter_digest = tensor_digest(torch.cat([p.detach().flatten() for p in actor.parameters()]))
        optimizer_digest = hashlib.sha256(b''.join(
            (name + tensor_digest(v)).encode() for s in optimizer.state.values() for name, v in sorted(s.items()) if isinstance(v, torch.Tensor)
        ) + json.dumps([[g['lr'], g.get('lr_multiplier', 1.)] for g in optimizer.param_groups]).encode()).hexdigest()
        dual_digest = hashlib.sha256(json.dumps(dual.state_dict(), sort_keys=True).encode()).hexdigest()
        if reference_digest is None:
            reference_digest = gradient_digest
            reference_update = (parameter_digest, optimizer_digest, dual_digest, grad)
        same_update = (parameter_digest, optimizer_digest, dual_digest, grad) == reference_update
        same_update = all_equal_flag(same_update and gradient_digest == reference_digest)
        totals = {}
        for row in rows:
            for key, value in (row['component_seconds'] or {}).items():
                totals[key] = totals.get(key, 0.) + value
        nested = {'student_gpt_forward_seconds', 'teacher_gpt_forward_seconds'}
        timed_sum = sum(v for k, v in totals.items() if k.endswith('_seconds') and not k.endswith('_host_seconds')
                        and not k.startswith('prefetch_worker_') and k not in nested)
        result = dict(repeat=repeat, warmup=repeat == 0, wall_seconds=wall, wall_seconds_max_over_ranks=wall_max,
                      states=len(states), seconds_per_state=wall / len(states), adam_and_clip_seconds=adam_seconds,
                      reduce_and_gather_seconds=reduce_seconds,
                      timed_phase_sum_seconds=timed_sum, untimed_remainder_seconds=wall - adam_seconds - reduce_seconds - timed_sum,
                      component_totals=totals, gradient_norm=grad, gradient_sha256=gradient_digest,
                      local_gradient_sha256=local_gradient_digest, parameters_after_step_sha256=parameter_digest,
                      optimizer_state_sha256=optimizer_digest, dual_state_sha256=dual_digest,
                      update_bit_identical_to_first_pass_all_ranks=same_update,
                      gradient_bit_identical_to_first_pass=gradient_digest == reference_digest,
                      prefetch_wait_seconds=prefetcher.wait_seconds if prefetcher is not None else None,
                      mean_kl=[r['mean_kl'] for r in rows],
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0)
        passes.append(result)
        for row in rows:
            log.write(json.dumps(dict(repeat=repeat, **row), allow_nan=False) + '\n')
        log.flush()
        if rank == 0:
            print(json.dumps({k: v for k, v in result.items() if k not in ('component_totals', 'mean_kl')}), flush=True)
    log.close()
    # Monkeypatches, timing installer and the prefetch thread are undone in main()'s finally
    # (also on exceptions), so a failed arm can never leak into a later in-process reference run.
    measured = [x for x in passes if not x['warmup']]
    summary = dict(label=args.label, arm=dict(timing=args.timing, deferred_kl=args.deferred_kl, kl_backend=args.kl_backend, prefetch=args.prefetch, prefetch_panel_only=args.prefetch_panel_only, constants_cache=args.constants_cache, coalesced_observations=args.coalesced_observations, shrinkage_coefficient=args.shrinkage_coefficient,
                                               norm_backend=args.norm_backend, gpt_precision=args.gpt_precision),
                   scope=distributed_note + '; audits/checkpoints excluded; warmup pass reported separately (cold compile + first-touch IO)',
                   world_size=world, rank=rank, states=states, batch_states=batch_states, epoch=epoch, sequences=args.sequences, microbatch=args.microbatch,
                   checkpoint_batch=args.checkpoint_batch, device=str(device), staged_files=staged,
                   policy=str(args.policy) if args.policy else None,
                   policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest() if args.policy else None,
                   median_wall_seconds=statistics.median(x['wall_seconds'] for x in measured),
                   median_wall_seconds_max_over_ranks=statistics.median(x['wall_seconds_max_over_ranks'] for x in measured),
                   median_seconds_per_state=statistics.median(x['seconds_per_state'] for x in measured),
                   warmup_wall_seconds=passes[0]['wall_seconds'],
                   all_measured_passes_bit_identical=all(x['gradient_bit_identical_to_first_pass'] for x in passes),
                   all_passes_same_update_all_ranks=all(x['update_bit_identical_to_first_pass_all_ranks'] for x in passes),
                   gradient_sha256=reference_digest, parameters_after_step_sha256=passes[0]['parameters_after_step_sha256'],
                   optimizer_state_sha256=passes[0]['optimizer_state_sha256'], dual_state_sha256=passes[0]['dual_state_sha256'],
                   passes=passes, profile=profile_summary,
                   compute=vars(engine.compute), torch_version=torch.__version__, source_sha256=before)
    if before != source_hashes() and not os.environ.get('PROFILE_CHISEL_SKIP_SOURCE_GUARD'):
        # Other agents edit this shared tree; a real measurement must be discarded if any
        # repository source changed underneath it. Tests set the environment override.
        raise RuntimeError('source changed during measurement; discard')
    if rank == 0:
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    else:
        (args.output / f'summary_rank{rank}.json').write_text(json.dumps(summary, indent=2) + '\n')
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return summary


def main(argv=None):
    """Run one arm; every process-global patch is undone on normal AND exceptional exit."""
    cleanups = []
    try:
        return _main(argv, cleanups)
    finally:
        for undo in reversed(cleanups):
            try:
                undo()
            except Exception:  # never mask the original error with a cleanup failure
                pass


if __name__ == '__main__':
    main()
