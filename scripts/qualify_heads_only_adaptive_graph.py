"""Prepare/compare a native heads-only graph gate; never allocates a GPU.

Requires an explicitly closed four-state, single-rank parent. All three scratch
forks retain its actual shared/private Adam, RNG, data clocks and duals. Only the
shared trainability policy, execution backend and bounded stopping/readout flags
change. A future externally authorized run uses the existing native trainer.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def qualification_recipe(parent, *, start, graph):
    from scripts.train_coordinate_chisel import validate_recipe
    r = copy.deepcopy(parent)
    if (r.get('action_family') != 'adaptive_ucv_v1'
            or not r['architecture'].get('fresh_weight_core')
            or r['architecture'].get('factor_precision', 'fp32') != 'fp32'
            or r.get('private_inner_steps') != 8):
        raise ValueError('gate requires adaptive FP32 fresh cores and eight private steps')
    r.pop('shared_optimizer_step', None)
    r.pop('experimental_heads_only_graph', None)
    r.update(shared_update_policy='heads_only', qualification_only=True,
             actor_backend='adaptive_actor_graph' if graph else 'eager',
             private_actor_backend='adaptive_actor_graph' if graph else 'eager')
    if graph:
        r['experimental_heads_only_graph'] = True
    r['stopping'].update(max_updates=start+4, full_sweep_every=start+4,
                         review_after_full_sweeps=[], stop_when_all_fit=False)
    r['readout']['heldout']['enabled'] = False
    if 'review_protocol' in r:
        r['qualification_parent_review_protocol']=r.pop('review_protocol')
    r['scope'] = 'Scratch heads-only adaptive actor graph parity/restart gate; no science continuation.'
    validate_recipe(r)
    return r


def prepare(parent, output, source, *, abba=False):
    from scripts.prepare_coordinate_backend_fork import metadata_from_archive, identity
    from rewriter.private_probe_store import PrivateProbeStore
    parent, output, source = Path(parent), Path(output), Path(source)
    status = json.loads((parent/'status.json').read_text())
    if status.get('status') != 'completed' or status.get('exit_code') != 0:
        raise ValueError('parent must have authoritatively closed successfully')
    old = json.loads((parent/'recipe.json').read_text())
    from scripts.train_coordinate_chisel import shared_update_policy
    parent_policy=shared_update_policy(old)
    meta = json.loads((parent/'results/metadata.json').read_text())
    if meta['recipe'] != old:
        raise ValueError('parent recipe differs from committed metadata')
    if metadata_from_archive(parent/'source.tar.gz',old,world_size=1,expected_sha=status['source_sha256'])!=meta:
        raise ValueError('parent source archive differs from committed metadata')
    with PrivateProbeStore(parent/'results/joint_store/global', read_only=True) as store:
        _, ref = store._manifest_and_reference()
        saved = store.restore(manifest_ref=ref)
    header = saved['sampler']
    if header['run_identity'] != identity(meta) or header['world_size'] != 1:
        raise ValueError('gate requires bound single-rank parent identity')
    state = header['ranks'][0]['sampler']
    start = state['next_global_batch']
    if (state['last_full_sweep_update'] != start or state['stop_reason'] != 'review_boundary'
            or len(state['visits']) != 4):
        raise ValueError('closed four-state fitting review required')
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    names = ('reference','candidate','candidate_repeat','reference_repeat','resumed') if abba else ('reference','candidate','resumed')
    for name in names:
        root = output/name
        (root/'results').mkdir(parents=True)
        recipe = qualification_recipe(old, start=start, graph=name in ('candidate','candidate_repeat','resumed'))
        recipe['qualification_gradient_trace_until']=0 if name.endswith('_repeat') else start+2
        newmeta = metadata_from_archive(source, recipe, world_size=1, expected_sha=sha)
        for filename, value in [('recipe.json', recipe), ('results/metadata.json', newmeta)]:
            (root/filename).write_text(json.dumps(value, indent=2)+'\n')
        plan = dict(schema='chisel_distributed_checkpoint_fork_plan_v1', world_size=1,
                    source=str(parent/'results/joint_store'), destination=str(root/'results/joint_store'),
                    parent_manifest_ref=ref, source_run_identity=identity(meta),
                    destination_run_identity=identity(newmeta),
                    provenance=dict(source_code_id=status['source_sha256'], destination_code_id=sha,
                        reason='Bounded heads-only execution comparison; same numeric owners, no mode conversion.'))
        (root/'fork_plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    (output/'parent_binding.json').write_text(json.dumps(dict(parent=str(parent),
        manifest_ref=ref, start=start, interrupt=start+2, end=start+4,
        shared_optimizer_steps=state['shared_optimizer_steps'], source_sha256=sha, abba=abba, parent_shared_update_policy=parent_policy), indent=2)+'\n')


def frozen_owners(shared, actor):
    """Select exact body tensors and their existing Adam state by named groups."""
    head_ids = {id(p) for head in actor.output_heads for p in head.parameters()}
    names = {n for n, p in actor.named_parameters() if id(p) not in head_ids}
    groups = shared['optimizer']['param_groups']
    body = [g for g in groups if g.get('role') == 'representation']
    if len(body) != 1:
        raise ValueError('explicit representation Adam group required')
    ids = body[0]['params']
    return dict(parameters={n: shared['model'][n] for n in sorted(names)},
                optimizer={i: shared['optimizer']['state'].get(i) for i in ids},
                groups=groups)



def validate_execution_order(executions, order):
    uuids={row.get('device_uuid') for row in executions.values()}
    if len(uuids)!=1 or None in uuids or '' in uuids:
        raise ValueError('ABBA must use the same recorded physical GPU UUID')
    prior_end=None
    for name in order:
        row=executions[name]
        if row.get('status')!='completed' or row.get('exit_code')!=0:
            raise ValueError('ABBA arm is not successfully completed')
        if row['ended']<row['started'] or (prior_end is not None and row['started']<prior_end):
            raise ValueError('ABBA processes overlap or ran out of declared order')
        prior_end=row['ended']


def validate_trace_restart(path,current):
    if not path.exists():return
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    keys=[(row['update'],row['ordinal']) for row in rows]
    if len(set(keys))!=len(keys) or any(row['update']>current for row in rows):
        raise ValueError('gradient trace contains duplicate or uncommitted rows; preserve failed arm and use a new fork')

def compare(parent, output):
    from scripts.train_coordinate_chisel import build_actor
    from scripts.qualify_coordinate_fullbank import read_joint
    from scripts.compare_coordinate_inner_program import numeric_visit
    from rewriter.distributed_probe_store import _digest
    root = Path(output)
    binding = json.loads((root/'parent_binding.json').read_text())
    start, end = binding['start'], binding['end']
    before = read_joint(Path(parent)/'results')
    recipe = json.loads((root/'reference/recipe.json').read_text())
    actor = build_actor(recipe)
    baseline_frozen = frozen_owners(before[0], actor)
    states = {n: read_joint(root/n/'results') for n in ('reference','candidate','resumed')}
    all_rows = {n: list(map(json.loads, (root/n/'results/metrics.jsonl').read_text().splitlines())) for n in states}
    rows = {n: [r for r in values if r['event']=='joint_update' and start<r['update']<=end] for n,values in all_rows.items()}
    comparisons = {}
    for a,b in [('reference','candidate'), ('candidate','resumed')]:
        comparisons[a+'_vs_'+b] = dict(
            shared_parameters_Adam_RNG_exact=_digest(states[a][0])==_digest(states[b][0]),
            sampler_duals_compute_exact=_digest(states[a][1])==_digest(states[b][1]),
            private_parameters_Adam_ticks_exact=_digest(states[a][2])==_digest(states[b][2]),
            gradient_loss_strict_rows_exact=_digest([[numeric_visit(v) for v in r['rows']] for r in rows[a]])==_digest([[numeric_visit(v) for v in r['rows']] for r in rows[b]]),
            shared_clipping_exact=_digest([r['shared_preclip_gradient_norms'] for r in rows[a]])==_digest([r['shared_preclip_gradient_norms'] for r in rows[b]]))
    traces={n:[json.loads(line) for line in (root/n/'results/qualification-gradients.jsonl').read_text().splitlines()] for n in states}
    for a,b in [('reference','candidate'),('candidate','resumed')]:
        comparisons[a+'_vs_'+b]['raw_private_and_head_gradient_hashes_exact']=bool(traces[a]) and traces[a]==traces[b]
    arms = {}
    for n,state in states.items():
        visits = [v for r in rows[n] for v in r['rows']]
        touched = {(v['domain'],v['local_id']) for v in visits}
        initial = before[1][0]['visits']; final = state[1][0]['visits']
        starts = [r for r in all_rows[n] if r['event']=='start']
        checks = dict(body_and_Adam_frozen=_digest(frozen_owners(state[0],actor))==_digest(baseline_frozen),
            heads_changed=_digest(state[0]['model'])!=_digest(before[0]['model']),
            private_parameters_changed=any(_digest(p['private'])!=_digest(before[2][token]['private']) for token,p in state[2].items()),
            four_updates=[r['update'] for r in rows[n]]==list(range(start+1,end+1)),
            shared_Adam_clock=state[1][0]['shared_optimizer_steps']==binding['shared_optimizer_steps']+4,
            changed_checkpoint_revisits=len(touched)==4 and len(visits)==8,
            private_tick_deltas=final.keys()==initial.keys() and all(final[k]-initial[k]==18 for k in initial),
            heads_only_steps=all(r['shared_update_policy']=='heads_only' and r['shared_optimizer_step_performed'] for r in rows[n]),
            fixed_capture_counts=bool(starts) and all(r['actor_graph_layouts']==(0 if n=='reference' else 2) and r['private_actor_graph_layouts']==(0 if n=='reference' else 2) for r in starts))
        if n!='reference':
            checks['no_scope_fallback']=all(r['eager_mask_fallbacks']==0 and r['private_eager_mask_fallbacks']==0 for r in rows[n])
        arms[n] = dict(gates=checks, wall_seconds=[r['wall_seconds'] for r in rows[n]],
                       strict_counts=[sum(v['strict_topk_changes'] for v in r['rows']) for r in rows[n]])
    result = dict(scope='Native single-rank heads-only FP32 adaptive8-inner+joint; changed states, actual owners, exact restart; no GPU result until executed.',
                  comparisons=comparisons, arms=arms,
                  all_gates=all(all(v.values()) for v in comparisons.values()) and all(all(v['gates'].values()) for v in arms.values()))
    if binding.get('abba'):
        import statistics
        order=['reference','candidate','candidate_repeat','reference_repeat']
        executions={name:json.loads((root/name/'results/qualification-execution.json').read_text()) for name in order}
        validate_execution_order(executions,order)
        samples={}
        for name in order:
            arm_rows=[json.loads(line) for line in (root/name/'results/metrics.jsonl').read_text().splitlines()]
            measured=[r for r in arm_rows if r['event']=='joint_update' and start+2<r['update']<=end]
            if [r['update'] for r in measured]!=[end-1,end]:
                raise ValueError('missing warmed changed-state timing rows')
            samples[name]=[r['wall_seconds'] for r in measured]
        result['timing_abba']=dict(order=order,executions=executions,samples_seconds=samples,
            reference_median_seconds=statistics.median(samples['reference']+samples['reference_repeat']),
            graph_median_seconds=statistics.median(samples['candidate']+samples['candidate_repeat']),
            scope='Last two revisited updates per native four-update process, same physical GPU for all arms; includes state/store/commit; capture/setup/readouts excluded. One sequential ABBA, no distribution-wide claim.')
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['all_gates']:
        raise RuntimeError('heads-only graph gate failed; preserve all results')


def run(recipe, output, *, interrupt=None, stop_after=None):
    import scripts.train_coordinate_chisel as native
    config=json.loads(Path(recipe).read_text())
    if config.get('qualification_only') is not True:
        raise ValueError('native wrapper only accepts qualification recipes')
    if interrupt is not None and stop_after is not None:
        raise ValueError('choose either interrupted replay or timing-only stop')
    requested=interrupt if interrupt is not None else stop_after
    end=config['stopping']['max_updates']
    if requested is not None and (type(requested) is not int or not 0<requested<=end):
        raise ValueError('invalid qualification stop boundary')
    class QualificationStop(Exception): pass
    from rewriter.private_probe_store import PrivateProbeStore
    from rewriter.distributed_probe_store import _digest
    with PrivateProbeStore(Path(output)/'joint_store/global',read_only=True) as store:
        current=store.restore()['sampler']['next_global_batch']
    trace_until=config.get('qualification_gradient_trace_until',0)
    trace_path=Path(output)/'qualification-gradients.jsonl'
    validate_trace_restart(trace_path,current)
    execution=dict(started=time.time(),status='running',recipe_sha256=hashlib.sha256(Path(recipe).read_bytes()).hexdigest(),timing_only=stop_after is not None)
    original_private_clip=native.clip_private_gradients
    original_shared_clip=native.clip_shared_continuation
    ordinal=0
    def record(kind, parameters):
        nonlocal ordinal
        if current>=trace_until:return
        parameters=tuple(parameters)
        row=dict(update=current+1,kind=kind,ordinal=ordinal,
                 none_mask=[p.grad is None for p in parameters],
                 sha256=_digest([p.grad for p in parameters]))
        with trace_path.open('a') as stream:stream.write(json.dumps(row,sort_keys=True)+'\n')
        ordinal+=1
    def private_clip(parameters,*args,**kwargs):
        parameters=tuple(parameters);record('private',parameters)
        return original_private_clip(parameters,*args,**kwargs)
    def shared_clip(optimizer,*args,**kwargs):
        record('shared',tuple(p for g in optimizer.param_groups for p in g['params']))
        return original_shared_clip(optimizer,*args,**kwargs)
    native.clip_private_gradients=private_clip;native.clip_shared_continuation=shared_clip
    original=native.run_training
    def controlled(*args,**kwargs):
        import torch
        props=torch.cuda.get_device_properties(torch.cuda.current_device())
        execution['device_uuid']=str(props.uuid)
        execution['device_name']=props.name
        def after_commit(batch):
            nonlocal current,ordinal
            current=batch;ordinal=0
            if batch==requested:
                Path(output,'qualification-stop.json').write_text(json.dumps(dict(update=batch,after_atomic_joint_commit=True,timing_only=stop_after is not None)))
                raise QualificationStop()
        kwargs['after_commit']=after_commit
        return original(*args,**kwargs)
    native.run_training=controlled
    oldargv=sys.argv
    sys.argv=['scripts/train_coordinate_chisel.py','--recipe',recipe,'--output',output,'--resume']
    try:
        try: native.main()
        except QualificationStop:
            if interrupt is not None: raise SystemExit(75)
        execution.update(status='completed',exit_code=0)
    except BaseException as error:
        execution.update(status='interrupted' if isinstance(error,SystemExit) and error.code==75 else 'failed',exit_code=error.code if isinstance(error,SystemExit) else 1,error=repr(error))
        raise
    finally:
        native.run_training=original;sys.argv=oldargv
        native.clip_private_gradients=original_private_clip;native.clip_shared_continuation=original_shared_clip
        execution['ended']=time.time()
        Path(output,'qualification-execution.json').write_text(json.dumps(execution,indent=2)+'\n')
        if 'wandb' in sys.modules and getattr(sys.modules['wandb'],'run',None) is not None:
            sys.modules['wandb'].finish(exit_code=execution.get('exit_code',1))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['prepare','run','compare'])
    p.add_argument('--parent');p.add_argument('--output',required=True)
    p.add_argument('--source');p.add_argument('--recipe');p.add_argument('--interrupt',type=int)
    p.add_argument('--abba',action='store_true');p.add_argument('--stop-after',type=int)
    a=p.parse_args()
    if a.mode=='prepare': prepare(a.parent,a.output,a.source,abba=a.abba)
    elif a.mode=='compare': compare(a.parent,a.output)
    else: run(a.recipe,a.output,interrupt=a.interrupt,stop_after=a.stop_after)
