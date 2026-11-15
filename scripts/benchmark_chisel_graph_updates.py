"""Bounded full200/8-rank qualification; never launch without owner handoff.

Reference is the deployed cached/untimed dense path. Both arms restore the same
saved actor, Adam, dual and RNG; use identical production state/panel selections.
Capture setup is reported separately. Any structural recapture DURING the batch
is timed. A common preclip diagnostic clone is included in both update timers;
CPU numerical comparisons and a full post-update fitting-panel audit are outside.
This does not include periodic production checkpoint saves or held-out audits.
"""
import argparse
import copy
from contextlib import contextmanager
from datetime import timedelta
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from rewriter.offline_systems import select_panel_blocks
from rewriter.qualification_metrics import (tensor_comparison, metric_within,
    parameter_delta, optimizer_snapshot, compare_adam, logit_response_comparison)
from scripts.benchmark_chisel_systems import production_batch, source_hashes


def benchmark_source_hashes():
    result=source_hashes()
    root=Path(__file__).resolve().parents[1]
    for folder in ('baseline','rewriter','scripts'):
        for path in (root/folder).rglob('*'):
            if path.suffix in ('.cu','.cuh','.cpp','.h'):
                result[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def load_pinned_policy(path, expected_sha256):
    """Load an immutable checkpoint, rejecting the wrong file or a rolling write."""
    path=Path(path)
    def stamp():
        st=path.stat()
        return (st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
    before=stamp()
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual!=expected_sha256:raise ValueError('policy SHA differs from owner-pinned checkpoint')
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if stamp()!=before:raise ValueError('policy changed during verified load')
    return saved,before


def check_shrinkage_policy(saved, expected_coefficient):
    coefficient=saved['metadata']['options'].get('shrinkage_coefficient',1.)
    if not math.isfinite(expected_coefficient) or expected_coefficient<0 or coefficient!=expected_coefficient:
        raise ValueError('saved shrinkage coefficient differs from the requested policy')
    declared=saved['metadata'].get('objective_policy',{}).get('shrinkage_coefficient',coefficient)
    if declared!=coefficient:raise ValueError('saved objective metadata contradicts options')
    return coefficient


def benchmark_objective(saved, expected_coefficient, override=None):
    """Declare a benchmark-only objective without rewriting checkpoint metadata.

    The saved-policy assertion remains mandatory even with an override. Both
    comparison arms and the matched prerequisite use the effective coefficient;
    actor parameters, Adam moments, dual state and data selection remain saved.
    """
    original = check_shrinkage_policy(saved, expected_coefficient)
    effective = original if override is None else override
    if not math.isfinite(effective) or effective < 0:
        raise ValueError('benchmark shrinkage override must be finite and nonnegative')
    return dict(saved_shrinkage_coefficient=original,
                effective_shrinkage_coefficient=effective,
                explicit_benchmark_override=override is not None,
                scope='restored single-step systems qualification; not a training run or checkpoint migration')


def selected_panel(panel, state_id, epoch, options):
    import scripts.train_multistate_chisel as train
    if len(panel.inputs) != 128 or panel.metadata['microbatch_sequences'] != 16:
        raise ValueError('production128/generationmicro16 required')
    generator = torch.Generator().manual_seed(train.stable_seed(
        'cached_panel_blocks', options['data_seed'], state_id, epoch//2))
    permutation = torch.randperm(8, generator=generator).tolist()
    offset = (epoch % 2)*4
    result = select_panel_blocks(panel, permutation[offset:offset+4])
    if any(train.heldout_sequence(i, options['split_seed']) for i in result.sequence_ids):
        raise ValueError('held-out data cannot enter benchmark updates')
    return result


def benchmark_tolerance(saved, override=None):
    """Explicit disposed tau change; preserve every saved multiplier and clock."""
    original = saved['metadata']['options']['kl_tolerance']
    if saved['primal_dual']['tolerance'] != original:
        raise ValueError('saved dual tolerance contradicts options')
    effective = original if override is None else override
    if not math.isfinite(effective) or effective <= 0:
        raise ValueError('positive finite benchmark tolerance required')
    state = copy.deepcopy(saved['primal_dual'])
    state['tolerance'] = effective
    return state, dict(saved_kl_tolerance=original, effective_kl_tolerance=effective,
        explicit_tolerance_override=override is not None,
        qualification='fresh objective qualification; old-tau prerequisites are structural/raw-KL evidence only')


def benchmark_initialization(saved, fresh_optimizer, restored_dual, fresh=False):
    """Choose explicit new-stage state without mutating a saved checkpoint."""
    optimizer = copy.deepcopy(fresh_optimizer if fresh else saved['optimizer'])
    dual = copy.deepcopy(restored_dual)
    if fresh:
        dual['multipliers'] = dict.fromkeys(dual['multipliers'], 0.)
        dual['updates'] = 0
    def digest(value):
        h = hashlib.sha256()
        def visit(v):
            if isinstance(v, torch.Tensor):
                t=v.detach().cpu().contiguous()
                h.update(str((str(t.dtype),tuple(t.shape))).encode())
                h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
            elif isinstance(v, dict):
                for k in sorted(v,key=str):h.update(repr(k).encode());visit(v[k])
            elif isinstance(v, (tuple,list)):
                for x in v:visit(x)
            else:h.update(repr(v).encode())
        visit(value)
        return h.hexdigest()
    return optimizer, dual, dict(optimizer_dual_initialization='fresh' if fresh else 'saved',
        initial_optimizer_sha256=digest(optimizer),initial_dual_sha256=digest(dual),
        initial_optimizer_state_count=len(optimizer['state']),initial_dual_updates=dual['updates'],
        initial_optimizer_steps=[float(v['step']) for v in optimizer['state'].values() if 'step' in v])


def paired_orders(repeats, graph_comparison=False):
    if graph_comparison:
        if repeats != 2:raise ValueError('graph comparison requires two warm timings per backend')
        return [('reference','graph','hybrid'),('hybrid','graph'),('graph','hybrid')]
    return [('reference', 'graph') if i % 2 == 0 else ('graph', 'reference')
            for i in range(repeats+1)]


@contextmanager
def benchmark_records(states, read, *, stager_factory=None, staging_rows=None):
    """Same ordered states in both arms; worker lifetime belongs inside timing.

    Capture and recapture must use the normal device reader separately. This
    stream's prefetched observations live on the host and only support rebind.
    """
    if stager_factory is None:
        yield ((state,*read(state)) for state in states)
        return
    stager=stager_factory()
    try:
        stager.schedule(states)
        def records():
            for state in states:
                record,panel,seconds=stager.take(state)
                if staging_rows is not None:
                    staging_rows.append(dict(state_id=state,**seconds))
                yield state,record,panel
        yield records()
    finally:
        stager.close()


def checked_local_stage(function, exchange_errors):
    """No collectives inside function; every rank exchanges errors before next stage."""
    result=error=None
    try:result=function()
    except Exception as exc:error=repr(exc)
    errors=exchange_errors(error)
    if any(errors):raise RuntimeError('rank local stage failed: '+repr(errors))
    return result


def detached_fitting_proposal(make_proposal, *, include_norm_readout=False):
    """Use fitting's compiled actor specialization even inside a no-grad audit."""
    with torch.enable_grad():
        edited,regularizer,stats,_,_=make_proposal()
        parameters={n:v.detach() for n,v in edited.items()}
        norm=float(regularizer.detach())
    if include_norm_readout:
        return parameters,norm,realized_norm_readout(stats)
    return parameters,norm


def realized_norm_readout(stats):
    """Validate and copy realized norm diagnostics, retaining percentage units."""
    from rewriter.grouped_norm import GROUP_NAMES
    result={}
    for key in ('group_l2_percentage_reduction','group_new_l2_norm'):
        values=stats[key]
        if set(values)!=set(GROUP_NAMES):raise ValueError('all fifteen realized norm groups required')
        result[key]={name:float(values[name]) for name in GROUP_NAMES}
        if not all(math.isfinite(v) for v in result[key].values()):
            raise ValueError('nonfinite realized group norm readout')
    for key in ('mean_group_l2_percentage_reduction','realized_edit_norm'):
        result[key]=float(stats[key])
        if not math.isfinite(result[key]):raise ValueError('nonfinite realized norm readout')
    if min(result['group_new_l2_norm'].values())<0 or result['realized_edit_norm']<0:
        raise ValueError('realized L2 norms cannot be negative')
    if not math.isclose(result['mean_group_l2_percentage_reduction'],
                        sum(result['group_l2_percentage_reduction'].values())/15,
                        rel_tol=1e-10,abs_tol=1e-10):
        raise ValueError('realized reduction must use fifteen equal groups')
    return result


def validate_prerequisite(summary, policy_sha, shrinkage, fused_softcap, dynamic_zero_groups=False, kl_backend='chunked', prefetch_checkpoints=False,
                          linear_backend='ieee', dx_backend='ieee', activation_backend='native'):
    if kl_backend not in ('chunked','tiled'):
        raise ValueError('unsupported KL backend')
    if (linear_backend not in ('ieee','tf32x3','packed3')
            or dx_backend not in ('ieee','tf32x3','cublas3_large','cublas3_cached')
            or activation_backend not in ('native','fused')):
        raise ValueError('unsupported graph backend')
    if (summary['policy_sha256'] != policy_sha or summary['sequences'] != 64
            or len(set(summary['states'])) < 3 or summary['shrinkage_coefficient'] != shrinkage
            or summary.get('fused_softcap', False) != fused_softcap
            or summary.get('dynamic_zero_groups', False) != dynamic_zero_groups
            or summary.get('kl_backend','chunked') != kl_backend
            or summary.get('linear_backend') != linear_backend
            or summary.get('dx_backend') != dx_backend
            or summary.get('activation_backend') != activation_backend):
        raise ValueError('matched policy/three-state64/shrinkage/backend prerequisite required')
    if prefetch_checkpoints and summary.get('prefetch_checkpoints') is not True:
        raise ValueError('prefetched rebind prerequisite required')
    for state in summary['states']:
        layout = summary['original_layout_action'][f'{state}:True']
        if not math.isfinite(layout['relative_error']) or layout['relative_error'] > 1e-7:
            raise ValueError('static snapshot changes the original actor proposal')
        for raw in (True,False):
            metric=summary['qualification'][f'{state}:{raw}']
            if not (math.isfinite(metric['relative_error']) and metric['relative_error'] <= .003
                    and math.isfinite(metric['cosine']) and metric['cosine'] >= .99999
                    and math.isfinite(metric['absolute_kl_difference']) and metric['absolute_kl_difference'] <= 1e-9):
                raise ValueError('checkpoint prerequisite numerical gate failed')
    for name in ('accum:True','accum:False','adam_update'):
        metric=summary['qualification'][name]
        if not (math.isfinite(metric['relative_error']) and metric['relative_error'] <= .003
                and math.isfinite(metric['cosine']) and metric['cosine'] >= .99999):
            raise ValueError('accumulated gradient/Adam prerequisite numerical gate failed')
    if dynamic_zero_groups:
        mask=summary['qualification']['early_zero_mask']
        if (mask['mode']!='dynamic' or mask['rejected_changed_mask']
                or not mask['zero_reference_groups'] or summary['states'][0]!=mask['state_id']):
            raise ValueError('initial frozen-state dynamic transition prerequisite required')


def row_errors(reference, candidate):
    a = {r['state_id']: r for r in reference}
    b = {r['state_id']: r for r in candidate}
    if len(a) != 200 or a.keys() != b.keys():
        raise ValueError('200 unique checkpoint rows required in each arm')
    result = []
    for n, row in a.items():
        if tuple(row['sequence_ids']) != tuple(b[n]['sequence_ids']):
            raise ValueError('checkpoint panel identities differ')
        values = [row[k] for k in ('mean_kl','objective')]+[b[n][k] for k in ('mean_kl','objective')]
        if not all(math.isfinite(v) for v in values):
            raise ValueError('nonfinite checkpoint metric')
        result.append(dict(state_id=n, kl_difference=b[n]['mean_kl']-row['mean_kl'],
                           objective_difference=b[n]['objective']-row['objective']))
        if 'realized_norms' in row or 'realized_norms' in b[n]:
            if 'realized_norms' not in row or 'realized_norms' not in b[n]:
                raise ValueError('realized norm readout missing in one arm')
            x,y=(realized_norm_readout(r['realized_norms']) for r in (row,b[n]))
            # Keep absolute outcomes: agreement alone does not establish shrinkage.
            result[-1]['realized_norms']=dict(reference=x,candidate=y)
            result[-1]['realized_norm_differences']=dict(
                mean_reduction_percentage_points=y['mean_group_l2_percentage_reduction']-x['mean_group_l2_percentage_reduction'],
                group_reduction_percentage_points={g:y['group_l2_percentage_reduction'][g]-v for g,v in x['group_l2_percentage_reduction'].items()},
                group_new_l2_norm={g:y['group_new_l2_norm'][g]-v for g,v in x['group_new_l2_norm'].items()},
                realized_edit_norm=y['realized_edit_norm']-x['realized_edit_norm'])
    return result


def compare_signatures(reference, candidate, initial, dual_lr, tolerance):
    metrics = {key: tensor_comparison(reference[key], candidate[key])
               for key in ('preclip','clipped')}
    metrics['adam_delta'] = tensor_comparison(parameter_delta(reference['parameters'], initial),
                                             parameter_delta(candidate['parameters'], initial))
    metrics['adam'] = compare_adam(reference['adam'], candidate['adam'])
    metrics['training_rows'] = row_errors(reference['rows'], candidate['rows'])
    metrics['post_update_rows'] = row_errors(reference['post'], candidate['post'])
    # Same production dual.step, fed the respective measured training KLs.
    da, db = reference['dual']['multipliers'], candidate['dual']['multipliers']
    if any(reference['dual'][k] != candidate['dual'][k] for k in ('schema','updates','rho','dual_lr','tolerance')):
        raise ValueError('dual configuration/update counter differs')
    if da.keys() != db.keys():raise ValueError('dual state inventories differ')
    metrics['dual_max_difference'] = max(abs(da[n]-db[n]) for n in da)
    metrics['dual_finite'] = all(math.isfinite(v) for v in (*da.values(), *db.values()))
    okay = all(metric_within(metrics[k], .003, .99999) for k in ('preclip','clipped'))
    okay &= metric_within(metrics['adam_delta'], .003)
    for key in ('exp_avg','exp_avg_sq'):
        entry = metrics['adam'][key]
        okay &= all(metric_within(m, .003) for m in [entry['global_metric'], *entry['groups']])
    for key in ('training_rows','post_update_rows'):
        okay &= all(abs(r['kl_difference']) <= 1e-9 and abs(r['objective_difference']) <= 1e-4
                    for r in metrics[key])
    okay &= metrics['dual_finite'] and metrics['dual_max_difference'] <= dual_lr*1e-9/tolerance+1e-12
    metrics['within_declared_global_gates'] = bool(okay)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('policy','state-bank','teacher-cache','output','prerequisite-summary'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--repeats', type=int, choices=(1,2), default=2)
    parser.add_argument('--fused-softcap', action='store_true')
    parser.add_argument('--dynamic-zero-groups', action='store_true')
    parser.add_argument('--prefetch-checkpoints', action='store_true',
                        help='Graph arm only: verified pinned host staging; requires matching prerequisite')
    parser.add_argument('--kl-backend', choices=('chunked','tiled'), default='chunked')
    parser.add_argument('--collective-timeout-seconds', type=int, default=360)
    parser.add_argument('--policy-sha256', required=True)
    parser.add_argument('--expected-shrinkage-coefficient', type=float, required=True)
    parser.add_argument('--benchmark-shrinkage-override', type=float,
                        help='Benchmark only: change BOTH arms objective, keep saved policy/Adam/dual; requires matching prerequisite')
    parser.add_argument('--benchmark-kl-tolerance-override', type=float)
    parser.add_argument('--fresh-optimizer-dual',action='store_true',help='Explicit new-stage fresh Adam and zero dual; original policy unchanged')
    parser.add_argument('--hybrid-prerequisite-summary', type=Path,
                        help='Enable dense oracle once then strict graph versus hybrid ABBA; fresh qualification')
    parser.add_argument('--linear-backend', choices=('ieee','tf32x3','packed3'), default='ieee')
    parser.add_argument('--dx-backend', choices=('ieee','tf32x3','cublas3_large','cublas3_cached'), default='ieee')
    parser.add_argument('--activation-backend', choices=('native','fused'), default='native')
    args = parser.parse_args()
    graph_comparison = args.hybrid_prerequisite_summary is not None
    if not graph_comparison and (args.linear_backend,args.dx_backend,args.activation_backend)!=('ieee','ieee','native'):
        parser.error('nondefault backends require explicit hybrid prerequisite and graph comparison')
    if graph_comparison and (args.repeats!=2 or not args.prefetch_checkpoints):
        parser.error('graph comparison requires repeats2 and prefetch in both graph arms')
    graph_backends = {'graph':dict(linear_backend='ieee',dx_backend='ieee',activation_backend='native')}
    if graph_comparison:
        graph_backends['hybrid']=dict(linear_backend=args.linear_backend,dx_backend=args.dx_backend,activation_backend=args.activation_backend)
    orders = paired_orders(args.repeats,graph_comparison)

    if not 60 <= args.collective_timeout_seconds <= 600:
        parser.error('collective timeout must be between60 and600seconds; retain external19min job deadline')
    rank, world, local = (int(os.environ.get(k, d)) for k,d in
                          (('RANK',0),('WORLD_SIZE',1),('LOCAL_RANK',0)))
    if world != 8:parser.error('full production qualification requires exactly8 ranks')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'summary.json').exists():parser.error('choose an unused output directory')
    pinned = benchmark_source_hashes()
    from baseline.execution import enable_deterministic_execution, execution_environment
    from baseline.model import GPT
    from rewriter.accelerated_grouped_norm import AcceleratedGroupedNormObjective
    from rewriter.chisel_factory import build_chisel
    from rewriter.edit_scope import scoped_snapshot
    from rewriter.graph_offline import NativeThinGradientGraph
    from rewriter.meta_optimizer import build_meta_adam
    from rewriter.offline_systems import CachedBaseNormPreservation, VerifiedStateConstants
    from rewriter.primal_dual import PrimalDual
    from rewriter.resident_teacher import ResidentTeacherFeatures
    from rewriter.state_bank import StateBankReader
    from rewriter.teacher_cache import TeacherCacheReader
    import scripts.train_multistate_chisel as train
    torch.set_num_threads(1);enable_deterministic_execution();torch.cuda.set_device(local)
    device = torch.device('cuda',local)
    # Observed first-visit updates took224s and cold compilation66s. A fast
    # rank must not time out after120s while a peer is still making progress.
    # The external19min benchmark deadline remains the absolute job bound.
    dist.init_process_group('nccl',timeout=timedelta(seconds=args.collective_timeout_seconds))
    saved,policy_stamp = load_pinned_policy(args.policy,args.policy_sha256)
    opts = dict(saved['metadata']['options'])
    restored_dual,tolerance_policy = benchmark_tolerance(saved,args.benchmark_kl_tolerance_override)
    opts['kl_tolerance'] = tolerance_policy['effective_kl_tolerance']
    if (opts['checkpoint_batch'],opts['sequences_per_checkpoint'],opts['gpt_microbatch']) != (200,64,16):
        raise ValueError('saved workload is not200x64micro16')
    if opts['gpt_precision']!='fp32' or opts['objective']!='primal-dual' or opts['norm_backend']!='fused':
        raise ValueError('strictFP32/primal-dual/fused norm required')
    if opts.get('gradient_diagnostics') or opts.get('compile_gpt') or opts.get('eager_chisel'):
        raise ValueError('this driver requires the qualified native compiled actor/eager GPT execution')
    reader = StateBankReader(args.state_bank,verify_once=True,mmap_model=True)
    cache = TeacherCacheReader(args.teacher_cache,verify_once=True,mmap=True)
    identity = saved['metadata']['state_bank_identity']
    for path,key in ((args.state_bank,'manifest_sha256'),(args.teacher_cache,'teacher_cache_manifest_sha256')):
        if hashlib.sha256((path/'manifest.json').read_bytes()).hexdigest()!=identity[key]:
            raise ValueError('saved policy and input manifest differ')
    states,epoch = production_batch(saved,reader.manifest['split']['train'],world)
    local_states = states[rank::world]
    if len(local_states)!=25:raise ValueError('25 checkpoints/rank required')
    engine = CachedBaseNormPreservation(GPT(**reader.manifest['provenance']['model_config']).to(device).requires_grad_(False),
        microbatch_sequences=16,teacher_cache_device=device,kl_dtype=torch.float64,gpt_precision='fp32')
    constants = VerifiedStateConstants(engine,AcceleratedGroupedNormObjective)
    actor = build_chisel(saved['metadata']['architecture']).to(device)
    forward = torch.compile(actor,fullgraph=True,dynamic=False)
    named = dict(actor.named_parameters()); parameters = list(named.values())
    dual = PrimalDual(reader.manifest['split']['train'],tolerance=opts['kl_tolerance'],rho=opts['augmented_rho'],dual_lr=opts['dual_lr'])
    objective = benchmark_objective(saved,args.expected_shrinkage_coefficient,args.benchmark_shrinkage_override)
    initial_optimizer,restored_dual,initialization = benchmark_initialization(saved,
        build_meta_adam(actor,lr=opts['lr'],representation_lr_multiplier=opts.get('representation_lr_multiplier',1.)).state_dict(),
        restored_dual,args.fresh_optimizer_dual)
    objective.update(tolerance_policy)
    objective.update(initialization)
    shrinkage = objective['effective_shrinkage_coefficient']
    prerequisite=json.loads(args.prerequisite_summary.read_text())
    validate_prerequisite(prerequisite,args.policy_sha256,shrinkage,args.fused_softcap,args.dynamic_zero_groups,args.kl_backend,args.prefetch_checkpoints)
    hybrid_prerequisite=None
    if graph_comparison:
        hybrid_prerequisite=json.loads(args.hybrid_prerequisite_summary.read_text())
        validate_prerequisite(hybrid_prerequisite,args.policy_sha256,shrinkage,args.fused_softcap,
            args.dynamic_zero_groups,args.kl_backend,args.prefetch_checkpoints,**graph_backends['hybrid'])

    prerequisite_context = {}
    for label,prereq in [('graph',prerequisite),('hybrid',hybrid_prerequisite)]:
        if prereq is None:continue
        prerequisite_context[label] = dict(kl_tolerance=prereq.get('kl_tolerance'),
            optimizer_dual_initialization=prereq.get('optimizer_dual_initialization','saved'))
        if args.benchmark_kl_tolerance_override is not None and prereq.get('kl_tolerance') != opts['kl_tolerance']:
            raise ValueError('explicit tolerance override requires matched effective-tau prerequisite')
        if prerequisite_context[label]['optimizer_dual_initialization'] != initialization['optimizer_dual_initialization']:
            raise ValueError('prerequisite optimizer/dual initialization differs')
    objective['prerequisite_context']=prerequisite_context

    stager_factory=None
    if args.prefetch_checkpoints:
        from rewriter.graph_stager import GraphCheckpointStager
        # Never share mutable verification caches with the synchronous reader.
        worker_reader=StateBankReader(args.state_bank,verify_once=True,mmap_model=True)
        worker_cache=TeacherCacheReader(args.teacher_cache,verify_once=True,mmap=True)
        def stager_factory():
            return GraphCheckpointStager(worker_reader,worker_cache,pin=True,
                select_panel=lambda panel,state:selected_panel(panel,state,epoch,opts))
    accum = [torch.zeros_like(p) for p in parameters]
    initial = {n:saved['controller'][n] for n in named}
    rng_cpu,rng_gpu = torch.get_rng_state(),torch.cuda.get_rng_state(device)
    graph = None
    def read(state):
        record = reader.read(state,device=device,model_device='cpu')
        panel = selected_panel(cache.read(state,'train'),state,epoch,opts)
        return record,panel
    def close_graph():
        nonlocal graph
        if graph is not None:graph.close();graph=None
        gc.collect();torch.cuda.empty_cache()
    def make_graph(record,panel,backend):
        constants.load(cache,record);grouped=constants.grouped()
        snapshot=scoped_snapshot(record.snapshot,constants.radius_squared(grouped),opts['radius_fraction'],scope='grouped')
        resident=ResidentTeacherFeatures(panel,engine)
        result=NativeThinGradientGraph(actor,forward,snapshot,engine,grouped,resident.inputs,None,dual,record.completed_steps,
            state_weight=1/200,teacher_features=resident.features,shrinkage_coefficient=shrinkage,
            fused_softcap=args.fused_softcap,dynamic_zero_groups=args.dynamic_zero_groups,kl_backend=args.kl_backend,
            **graph_backends[backend])
        result.capture();return result
    def proposal(record):
        constants.load(cache,record);grouped=constants.grouped()
        return train.build_proposal(forward,record.snapshot,engine,grouped,radius_fraction=opts['radius_fraction'],
            norm_squared=constants.radius_squared(grouped),shrinkage_coefficient=shrinkage)
    def collective_error(error):
        errors=train.all_rank_objects(error,world)
        if any(errors):raise RuntimeError('rank qualification failure: '+repr(errors))
    def checked(function):
        return checked_local_stage(function,lambda error:train.all_rank_objects(error,world))
    reference=None;last_graph_signatures={};trials=[];wb=None
    def heartbeat(phase, **fields):
        (args.output/f'heartbeat_rank{rank}.json').write_text(json.dumps(dict(
            phase=phase,unix_time=time.time(),rank=rank,**fields))+'\n')
    if rank==0:
        import wandb
        wb=wandb.init(project='chisel-systems',mode='offline',dir=str(args.output),
            config=dict(workload='full200x64micro16',world_size=world,production_changed=False,
                        benchmark_objective=objective,prefetch_checkpoints=args.prefetch_checkpoints,graph_backends=graph_backends,
                        backend='native-thin-graph',fused_softcap=args.fused_softcap,dynamic_zero_groups=args.dynamic_zero_groups,kl_backend=args.kl_backend))
        (args.output/'startup.json').write_text(json.dumps(dict(states=states,local_states=[states[r::world] for r in range(world)],
            epoch=epoch,options=opts,benchmark_objective=objective,prefetch_checkpoints=args.prefetch_checkpoints,source_sha256=pinned,policy_sha256=args.policy_sha256,
            environment=execution_environment(),fused_softcap=args.fused_softcap,dynamic_zero_groups=args.dynamic_zero_groups,kl_backend=args.kl_backend,
            collective_timeout_seconds=args.collective_timeout_seconds,
            graph_backends=graph_backends,trial_orders=orders,hybrid_prerequisite_summary=hybrid_prerequisite,
            hybrid_prerequisite_sha256=None if not graph_comparison else hashlib.sha256(args.hybrid_prerequisite_summary.read_bytes()).hexdigest(),
            prerequisite_summary=prerequisite,prerequisite_sha256=hashlib.sha256(args.prerequisite_summary.read_bytes()).hexdigest()),indent=2)+'\n')
    for repeat,order in enumerate(orders):
        for backend in order:
            close_graph();actor.load_state_dict(saved['controller'])
            optimizer=build_meta_adam(actor,lr=opts['lr'],representation_lr_multiplier=opts.get('representation_lr_multiplier',1.))
            optimizer.load_state_dict(copy.deepcopy(initial_optimizer));dual.load_state_dict(restored_dual)
            torch.set_rng_state(rng_cpu);torch.cuda.set_rng_state(rng_gpu,device)
            setup=time.perf_counter();error=None
            try:
                heartbeat('setup_begin',repeat=repeat,backend=backend,state_id=local_states[0])
                if backend in graph_backends:
                    record,panel=read(local_states[0])
                    heartbeat('capture_begin',repeat=repeat,backend=backend,state_id=local_states[0])
                    graph=make_graph(record,panel,backend)
                    del record,panel
                heartbeat('setup_complete',repeat=repeat,backend=backend,seconds=time.perf_counter()-setup)
            except Exception as exc:error=repr(exc)
            collective_error(error);torch.cuda.synchronize();setup=time.perf_counter()-setup
            optimizer.zero_grad(set_to_none=True);torch._foreach_zero_(accum)
            dist.barrier();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            started=time.perf_counter();rows=[];staging_rows=[];recaptures=0;error=None
            try:
                with benchmark_records(local_states,read,
                        stager_factory=stager_factory if backend in graph_backends else None,
                        staging_rows=staging_rows) as records:
                    for state,record,panel in records:
                        if backend=='reference':
                            edited,regularizer,stats,_,credit=proposal(record)
                            measured=train.mean_kl_and_backward(engine,edited,regularizer,panel.inputs,kl_budget=opts['kl_budget'],
                                state_weight=1/200,actor_credit=credit,teacher_provider=lambda a,b:panel.teacher_batch(engine,a,b),
                                primal_dual=dual,state_id=state)
                            row=dict(state_id=state,sequence_ids=panel.sequence_ids,**measured)
                            del edited,regularizer,credit,stats
                        else:
                            try:graph.load_checkpoint(cache,record,panel,dual,state_constants=constants,radius_fraction=opts['radius_fraction'])
                            except ValueError as exc:
                                if 'zero-reference' not in str(exc):raise
                                close_graph()
                                # Host-prefetched observations cannot initialize a CUDA actor.
                                capture_record,capture_panel=read(state) if args.prefetch_checkpoints else (record,panel)
                                graph=make_graph(capture_record,capture_panel,backend);recaptures+=1
                                del capture_record,capture_panel
                            out=graph.accumulate_into(accum);graph.check_finite()
                            row=dict(state_id=state,sequence_ids=panel.sequence_ids,
                                     mean_kl=float(out['mean_kl']),objective=float(out['objective']))
                            del out
                        rows.append(row);del record,panel
                        if len(rows)%5==0:
                            heartbeat('accumulation',repeat=repeat,backend=backend,
                                completed=len(rows),total=25,elapsed=time.perf_counter()-started)
                if backend in graph_backends:
                    for p,g in zip(parameters,accum):p.grad=g
            except Exception as exc:error=repr(exc)
            collective_error(error)
            checked(torch.cuda.synchronize);accumulation_seconds=time.perf_counter()-started
            train.aggregate_controller_gradients(actor,world)
            preclip=checked(lambda:{n:p.grad.detach().clone() for n,p in named.items()})
            rows=sorted([r for part in train.all_rank_objects(rows,world) for r in part],key=lambda r:r['state_id'])
            def finish_update():
                norm=float(torch.nn.utils.clip_grad_norm_(parameters,opts['gradient_clip'],error_if_nonfinite=True))
                optimizer.step();dual.step([(r['state_id'],r['mean_kl']) for r in rows],expected_ids=states)
                torch.cuda.synchronize();return norm,time.perf_counter()-started
            grad_norm,seconds=checked(finish_update)
            peak=torch.cuda.max_memory_allocated();compiles=train.counters()
            signature=checked(lambda:dict(parameters={n:p.detach().cpu().clone() for n,p in named.items()},
                preclip={n:v.cpu() for n,v in preclip.items()},clipped={n:p.grad.detach().cpu().clone() for n,p in named.items()},
                adam=optimizer_snapshot(optimizer,named),dual=copy.deepcopy(dual.state_dict()),rows=rows))
            del preclip;close_graph()
            # Same dense evaluation path after BOTH updates; current (post-step)
            # dual is restored to the saved multiplier for comparable objectives.
            dual.load_state_dict(restored_dual);post=[];error=None
            try:
                with torch.no_grad():
                    for state in local_states:
                        record,panel=read(state)
                        edited,regularizer,norm_readout=detached_fitting_proposal(lambda:proposal(record),include_norm_readout=True)
                        kl=train.evaluate_sequences(engine,edited,panel.inputs,teacher_provider=lambda a,b:panel.teacher_batch(engine,a,b))
                        post.append(dict(state_id=state,sequence_ids=panel.sequence_ids,mean_kl=kl,
                                         realized_norms=norm_readout,
                                         objective=float(regularizer)+dual.terms(kl,state)['kl_penalty']))
                        del record,panel,edited,regularizer
                        if len(post)%5==0:heartbeat('post_update_audit',repeat=repeat,backend=backend,completed=len(post),total=25)
            except Exception as exc:error=repr(exc)
            collective_error(error)
            signature['post']=sorted([r for part in train.all_rank_objects(post,world) for r in part],key=lambda r:r['state_id'])
            if reference is None:
                if backend!='reference':raise AssertionError('first trial must establish reference')
                reference=signature
            error=None
            try:
                metrics=compare_signatures(reference,signature,initial,opts['dual_lr'],opts['kl_tolerance'])
                exact=(all(metrics[k]['bit_identical'] for k in ('preclip','clipped','adam_delta'))
                       and all(metrics['adam'][k]['global_metric']['bit_identical'] for k in ('exp_avg','exp_avg_sq'))
                       and all(r['kl_difference']==r['objective_difference']==0 for k in ('training_rows','post_update_rows') for r in metrics[k])
                       and metrics['dual_max_difference']==0)
                metrics['reference_repeat_bit_identical']=exact if backend=='reference' else None
                if backend=='reference':metrics['within_declared_global_gates'] &= exact
            except Exception as exc:error=repr(exc)
            collective_error(error)
            rank_rows=train.all_rank_objects(dict(rank=rank,seconds=seconds,accumulation_seconds=accumulation_seconds,
                setup_seconds=setup,peak_bytes=peak,recaptures=recaptures,grad_norm=grad_norm,
                staging_rows=staging_rows,
                clip_coefficient=min(1.,opts['gradient_clip']/(grad_norm+1e-6)),within_gates=metrics['within_declared_global_gates']),world)
            okay=all(r['within_gates'] for r in rank_rows)
            result=dict(repeat=repeat,backend=backend,warmup=repeat==0,seconds=max(r['seconds'] for r in rank_rows),
                rank_rows=rank_rows,all_rank_gates=okay,compilation_counters=compiles,
                prefetch_checkpoints=args.prefetch_checkpoints and backend in graph_backends,
                arithmetic_class='approximate' if backend=='hybrid' else 'strict',graph_backend=graph_backends.get(backend))
            (args.output/f'qualification_{repeat}_{backend}_rank{rank}.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')
            if not okay:torch.save(dict(reference=reference,candidate=signature),args.output/f'failure_rank{rank}.pt')
            trials.append(result)
            if rank==0:
                (args.output/'trials.json').write_text(json.dumps(trials,indent=2)+'\n');print(json.dumps(result),flush=True)
                wb.log(dict(objective=3600/result['seconds'],seconds=result['seconds'],backend=backend,
                            repeat=repeat,warmup=repeat==0,all_rank_gates=okay,
                            gradient_relative_error=metrics['preclip']['relative_error'],
                            adam_delta_relative_error=metrics['adam_delta']['relative_error']))
            if not okay:raise RuntimeError('global numerical gate failed; no adoption')
            if backend in graph_backends:last_graph_signatures[backend]=signature
    # Mandatory finite-response diagnostics of the actual GLOBAL gradient at
    # nominal and0.1LR. Eight fitting checkpoints (one/rank), first16 selected
    # sequences each; same dense forward for both updated actors. Not timed.
    from rewriter.function_preservation import _ChunkedTeacherKL
    actor.load_state_dict(saved['controller']);dual.load_state_dict(restored_dual)
    record,panel=read(local_states[0]);diagnostics=[];error=None
    try:
        with torch.no_grad():
            edited,regularizer=detached_fitting_proposal(lambda:proposal(record))
            inputs=panel.inputs[:16].to(device)
            before=engine._forward(edited,inputs)
            teacher=panel.teacher_batch(engine,0,16).log_probs
            before_kl=float(_ChunkedTeacherKL.apply(before,teacher,engine.chunk_tokens))
            del edited,regularizer
            for scale in (1.,.1):
                after_reference=None;reference_update=None;reference_kl=None
                for backend,source in [('reference',reference),*last_graph_signatures.items()]:
                    actor.load_state_dict(saved['controller'])
                    optimizer=build_meta_adam(actor,lr=opts['lr'],representation_lr_multiplier=opts.get('representation_lr_multiplier',1.))
                    optimizer.load_state_dict(copy.deepcopy(initial_optimizer))
                    for group in optimizer.param_groups:group['lr']*=scale
                    for n,p in named.items():p.grad=source['preclip'][n].to(device)
                    torch.nn.utils.clip_grad_norm_(parameters,opts['gradient_clip'],error_if_nonfinite=True);optimizer.step()
                    edited,regularizer=detached_fitting_proposal(lambda:proposal(record))
                    after=engine._forward(edited,inputs)
                    kl=float(_ChunkedTeacherKL.apply(after,teacher,engine.chunk_tokens))
                    update=parameter_delta(named,initial)
                    if backend=='reference':after_reference=after;reference_update=update;reference_kl=kl
                    else:
                        response_metric=logit_response_comparison(before,after_reference,after)
                        if not response_metric['finite'] or not all(math.isfinite(v) for v in (before_kl,reference_kl,kl)):
                            raise FloatingPointError('nonfinite global-gradient finite-response diagnostic')
                        diagnostics.append(dict(backend=backend,state_id=local_states[0],sequence_ids=panel.sequence_ids[:16],lr_scale=scale,
                            response=response_metric,
                            adam_delta=tensor_comparison(reference_update,update),kl_before=before_kl,
                            kl_after_reference=reference_kl,kl_after_candidate=kl))
                    del edited,regularizer,after
                del after_reference,reference_update
            del before,teacher
    except Exception as exc:error=repr(exc)
    collective_error(error)
    responses=[r for part in train.all_rank_objects(diagnostics,world) for r in part]
    st=args.policy.stat()
    collective_error(None if (st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)==policy_stamp
                     else 'pinned policy file changed during benchmark')
    collective_error(None if pinned==benchmark_source_hashes() else 'source changed during measurement')
    if rank==0:
        (args.output/'summary.json').write_text(json.dumps(dict(scope=__doc__,world_size=world,checkpoint_batch=200,sequences=64,microbatch=16,
            benchmark_objective=objective,prefetch_checkpoints=args.prefetch_checkpoints,policy_sha256=args.policy_sha256,
            states=states,trials=trials,global_gradient_response_diagnostics=responses,
            source_sha256=pinned,median_seconds={b:statistics.median(t['seconds'] for t in trials if t['backend']==b and not t['warmup']) for b in (tuple(graph_backends) if graph_comparison else ('reference','graph'))},
            graph_backends=graph_backends,graph_comparison=graph_comparison,production_changed=False),indent=2,allow_nan=False)+'\n')
        wb.finish()
    dist.destroy_process_group()


if __name__=='__main__':main()
