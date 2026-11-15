"""Explicit experimental fork: world1 baseline prefix -> world8 one-shot continuations.

Preparation only; never launches training or evaluates GPT. The learned arms apply
ONE frozen-checkpoint edit, retain original optimizer state, then resume ordinary
PR323. This is additional-compute, non-Track3 research, not a continuously online
controller. baseline.checkpoint's strict same-run resume API remains unchanged.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Fully recorded historical all-parameter frozen panel. Its action arithmetic is
# unchanged by the explicit scope helpers; other source mismatches still fail.
LEGACY_ALL_TRAINER_SHA256 = 'b9b18d3637c47fad588c5ee7fd6ac8555cf7a7fd2c0374418c5f793c7e2ca00d'
LEGACY_SCOPED_SHA256 = {
    'scripts/train_frozen_preservation.py':'5980e0247c0e700ccf593b8292291737dd670b3c8b232b0affb31dab99c8e819',
    'rewriter/edit_scope.py':'27b265b7c780b25f3dabcb285800411e6578355caf6c8e2f1759e11f87018ecb',
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def continuation_config(source_config, *, label, provenance, world_size=8):
    """Only execution and explicit fork metadata change; task knobs are inherited."""
    result = copy.deepcopy(source_config)
    result.update(compile=True, compile_loss=False, deterministic=True,
                  mode='baseline', world_size=world_size, benchmark_compliant=False,
                  experiment_kind='one_shot_frozen_rewrite', fork_label=label,
                  fork_provenance=copy.deepcopy(provenance))
    return result


def reshard_checkpoint(source, target_config, rank, *, model_state=None):
    """Pure CPU, nonmutating, independent output for one target rank.

    The first optimizer is replicated Adam, the second is PR323 Muon. Muon's
    sorted param-group order is preserved exactly; position modulo world decides
    ownership separately in each group. No numerical state transport is done.
    Counters are logical equivalent history, not fabricated physical execution;
    both original counters and aggregate prefix work are preserved in metadata.
    """
    cfg = source['config']; world = target_config['world_size']; step = source['step']
    if source.get('format_version') != 1 or source.get('controller') is not None:
        raise ValueError('expected format1 fixed-baseline checkpoint with no controller')
    if cfg.get('mode') != 'baseline' or cfg.get('world_size') != 1:
        raise ValueError('only a complete world1 baseline source can be resharded')
    if world not in (1, 2, 4, 8) or not 0 <= rank < world:
        raise ValueError('invalid target world/rank')
    if not (target_config.get('mode') == 'baseline' and target_config.get('compile') is True
            and target_config.get('compile_loss') is False and target_config.get('deterministic') is True
            and target_config.get('benchmark_compliant') is False
            and target_config.get('experiment_kind') == 'one_shot_frozen_rewrite'):
        raise ValueError('target must explicitly identify the compiled experimental fork')
    allowed = {'compile','compile_loss','deterministic','mode','world_size','benchmark_compliant',
               'experiment_kind','fork_label','fork_provenance'}
    changed = {k for k in cfg.keys() | target_config.keys() if k not in allowed and cfg.get(k) != target_config.get(k)}
    if changed:
        raise ValueError(f'task/optimizer/data settings changed: {sorted(changed)}')
    denominator = cfg['seq_len']*cfg['microbatch_sequences']
    if cfg['batch_tokens'] % (denominator*world) or not 0 < step < cfg['max_steps']:
        raise ValueError('invalid complete-step batch partition or fork clock')
    original_count = step*(cfg['batch_tokens']//denominator)
    expected = {'forward': original_count, 'backward': original_count}
    if source['counters'] != expected:
        raise ValueError('source counters do not describe prescribed world1 prefix')
    cursor = source['data_cursor']
    if (cursor['batches'] != step or cursor['world_size'] != 1 or cursor['rank'] != 0
            or cursor['batch_size'] != cfg['batch_tokens'] or cursor['seq_len'] != cfg['seq_len']
            or cursor['pos'] < 0 or cursor['pos'] % cfg['batch_tokens']):
        raise ValueError('source cursor does not describe the prescribed global prefix')
    if len(source['optimizers']) != 2:
        raise ValueError('expected exactly replicated Adam followed by Muon')
    adam, muon = source['optimizers']
    if not all('betas' in g for g in adam['param_groups']) or not all('mu' in g for g in muon['param_groups']):
        raise ValueError('optimizer order/type does not match PR323')
    owned = []; all_ids = []
    for group in muon['param_groups']:
        ids = group['params']; all_ids.extend(ids)
        owned.extend(pid for i, pid in enumerate(ids) if i % world == rank)
    if len(set(all_ids)) != len(all_ids) or set(muon['state']) != set(all_ids):
        raise ValueError('world1 Muon state must cover every unique parameter exactly once')
    if any(set(muon['state'][pid]) != {'momentum'} for pid in all_ids):
        raise ValueError('unexpected Muon state; no speculative state conversion')
    for value in source['model'].values():
        if value.device.type != 'cpu':
            raise ValueError('pure resharding requires CPU state tensors')
    selected_model = source['model'] if model_state is None else model_state
    if set(selected_model) != set(source['model']):
        raise ValueError('replacement model must include identical parameters and buffers')
    for name, old in source['model'].items():
        new = selected_model[name]
        if new.shape != old.shape or new.dtype != old.dtype or new.device.type != 'cpu':
            raise ValueError('replacement model shape/dtype/device mismatch: '+name)
    result = {k: copy.deepcopy(v) for k,v in source.items() if k not in ('model','optimizers','config')}
    result['model'] = copy.deepcopy(selected_model)
    result['optimizers'] = [copy.deepcopy(adam), dict(
        param_groups=copy.deepcopy(muon['param_groups']),
        state={pid:copy.deepcopy(muon['state'][pid]) for pid in owned})]
    result['config'] = copy.deepcopy(target_config)
    result['data_cursor'].update(rank=rank, world_size=world)
    result['counters'] = {k:original_count//world for k in expected}
    result['elapsed_seconds'] = 0.
    result['fork_history'] = dict(source_world_size=1, target_world_size=world,
        source_counters=copy.deepcopy(source['counters']),
        logical_target_counters_per_rank=copy.deepcopy(result['counters']),
        actual_prefix_global_forward=original_count, actual_prefix_global_backward=original_count,
        source_elapsed_seconds=source['elapsed_seconds'], source_data_cursor=copy.deepcopy(cursor),
        rng_policy='source CPU/CUDA/python/numpy RNG copied unchanged to every rank; GPT has no dropout',
        optimizer_state_policy='Adam replicated; Muon same momentum values reassigned by per-group position modulo world',
        one_shot_edit_only=True, preparation_gpt_forwards=0, preparation_gpt_backwards=0)
    return result


def validate_completed_run(run, source_config, checkpoint_sha, step, *, expected_updates=1000):
    """Read-only qualification before CUDA allocation or output creation."""
    run = Path(run)
    required = ['controller.pt','observation.pt','config.json','metrics.jsonl','source_record.txt','replay_provenance.json']
    if any(not (run/n).is_file() for n in required):
        raise ValueError('missing completed frozen-run artifact: '+str(run))
    cfg = json.loads((run/'config.json').read_text())
    if (cfg.get('checkpoint_sha256') != checkpoint_sha or cfg.get('checkpoint_step') != step
            or cfg.get('base_config') != source_config or cfg.get('steps') != expected_updates
            or cfg.get('controller_state_transferred') is not False):
        raise ValueError('frozen-run source checkpoint/config/update clock mismatch')
    recipe = dict(action_units='relative', optimizer_mode='adam', preservation_penalty='deadband',
                  kl_arithmetic='float64 from float32 GPT logits', source_readout='live')
    if any(cfg.get(k) != v for k,v in recipe.items()):
        raise ValueError('unsupported frozen-run reconstruction recipe')
    saved_controller = torch.load(run/'controller.pt',map_location='cpu',weights_only=False,mmap=True)
    if saved_controller.get('step') != expected_updates or saved_controller.get('config') != cfg:
        raise ValueError('controller checkpoint is not the completed declared endpoint')
    del saved_controller
    replay = json.loads((run/'replay_provenance.json').read_text())
    if replay.get('metadata',{}).get('completed_steps') != step:
        raise ValueError('replay provenance differs from source prefix')
    for split in ('train','audit'):
        if any(not 0 <= row['step'] < step for row in replay[split]):
            raise ValueError('replay contains future sequence IDs')
    if source_config.get('rewriter',{}).get('probe_rank',8) != 8:
        raise ValueError('this reconstruction is pinned to rank8')
    rows = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines() if line]
    finished = [r for r in rows if r.get('event') == 'finished']
    train = [r for r in rows if r.get('event') == 'train' and r.get('step') == expected_updates]
    audit = [r for r in rows if r.get('event') == 'evaluation' and r.get('label') == 'learned'
             and r.get('step') == expected_updates]
    if (len(finished) != 1 or finished[0].get('controller_steps') != expected_updates
            or len(train) != 1 or len(audit) != 1 or finished[0].get('benchmark_compliant') is not False):
        raise ValueError('frozen run not complete with a unique final learned audit')
    # Replaying actor/commit algebra must use the original source definitions.
    record = (run/'source_record.txt').read_text()
    core = ['rewriter/algebra.py','rewriter/controller.py','rewriter/function_preservation.py',
            'scripts/train_frozen_preservation.py']
    scope=cfg.get('editable_scope','all')
    if scope not in ('all','hidden','grouped'):raise ValueError('unknown recorded editable scope')
    objective=cfg.get('norm_objective','global_squared')
    if objective not in ('global_squared','grouped_l2') or ((scope=='grouped')!=(objective=='grouped_l2')):
        raise ValueError('recorded edit scope and norm objective do not match')
    if 'editable_scope' in cfg:core.append('rewriter/edit_scope.py')
    if objective=='grouped_l2':
        if cfg.get('norm_target_fraction')!=.01:raise ValueError('grouped loss must use percentage units')
        core.append('rewriter/grouped_norm.py')
    for name in core:
        found = re.findall(r'^===== '+re.escape(name)+r' SHA256 ([0-9a-f]{64}) =====$', record, re.M)
        legacy=(name=='scripts/train_frozen_preservation.py' and 'editable_scope' not in cfg
                and found==[LEGACY_ALL_TRAINER_SHA256])
        legacy_scoped=('norm_objective' not in cfg and scope in ('all','hidden')
            and name in LEGACY_SCOPED_SHA256 and found==[LEGACY_SCOPED_SHA256[name]])
        if len(found) != 1 or (found[0] != sha256(ROOT/name) and not legacy and not legacy_scoped):
            raise ValueError('local reconstruction source differs from frozen run: '+name)
    return dict(config=cfg, train=train[0], audit=audit[0], finished=finished[0],
                files={name:sha256(run/name) for name in required})


def check_endpoint_statistics(actual, record, *, rtol=1e-6):
    """Compare algebraic reconstruction to reported endpoint, not bit identity."""
    expected = {k:record['audit'][k] for k in ('norm_relative_change','realized_edit_norm')}
    expected.update({k:record['train'][k] for k in ('raw_action_norm','global_scale')})
    for key, old in expected.items():
        new = actual[key]
        # Absolute tolerance only protects true zero near machine precision.
        if not math.isfinite(new) or not math.isclose(new, old, rel_tol=rtol, abs_tol=1e-12):
            raise ValueError(f'endpoint reconstruction mismatch {key}: actual={new}, reported={old}')
    result=dict(reported=expected, reconstructed=actual, relative_tolerance=rtol, absolute_tolerance=1e-12)
    if (record.get('config',{}).get('norm_objective')=='grouped_l2'
            or record['audit'].get('norm_objective')=='grouped_l2'):
        from rewriter.grouped_norm import GROUP_NAMES
        key='group_l2_percentage_reduction'
        old_groups=record['audit'].get(key)
        new_groups=actual.get(key)
        if (not isinstance(old_groups,dict) or not isinstance(new_groups,dict)
                or set(old_groups)!=set(GROUP_NAMES) or new_groups.keys()!=old_groups.keys()):
            raise ValueError('endpoint reconstruction functional group names differ or are incomplete')
        # Percentage points: 1e-8 corresponds to 1e-10 fractional norm change.
        # This absolute tolerance handles groups whose measured change is zero.
        percentage_atol=1e-8
        for group in GROUP_NAMES:
            old,new=old_groups[group],new_groups[group]
            if (not math.isfinite(old) or not math.isfinite(new)
                    or not math.isclose(new,old,rel_tol=rtol,abs_tol=percentage_atol)):
                raise ValueError(f'endpoint reconstruction group percentage mismatch {group}: actual={new}, reported={old}')
        expected[key]=dict(old_groups)
        result['group_percentage_absolute_tolerance']=percentage_atol
    return result


@torch.no_grad()
def family_edit_statistics(base, edited, *, chunk_rows=512):
    """Stored-weight changes only; head means head matrix, all vectors are 1D."""
    result = {}
    for name, old in base.items():
        new = edited[name]
        family = ('embedding' if name == 'embed.weight' else
                  'one_dimensional' if old.ndim == 1 else
                  'head' if name == 'proj.weight' else 'hidden_matrices')
        row = result.setdefault(family, dict(base_norm_sq=0., edit_norm_sq=0., norm_sq_change=0.,
                                            changed_elements=0, elements=0))
        for start in range(0,len(old),chunk_rows):
            a=old[start:start+chunk_rows].double();b=new[start:start+chunk_rows].double();d=b-a
            squared=float(d.square().sum())
            row['base_norm_sq']+=float(a.square().sum());row['edit_norm_sq']+=squared
            row['norm_sq_change']+=float(2*(a*d).sum())+squared
            row['changed_elements']+=int((d != 0).sum());row['elements']+=d.numel()
    for row in result.values():
        row['weight_edit_inner_product']=(row['norm_sq_change']-row['edit_norm_sq'])/2
        denominator=math.sqrt(row['base_norm_sq']*row['edit_norm_sq'])
        row['cosine_toward_negative_weight']=(-row['weight_edit_inner_product']/denominator if denominator else None)
    return result


@torch.no_grad()
def reconstruct_frozen_edit(source, run, record, *, device):
    """Controller and dense optimizer algebra only; zero GPT evaluations."""
    from baseline.model import GPT
    from rewriter.controller import GlobalRewriter
    from rewriter.function_preservation import FunctionalPreservation
    from rewriter.algebra import RawAction, Factors
    from rewriter.edit_scope import (editable_names,editable_weight_norm_sq,scoped_action,
        keep_readonly_parameters,readonly_hashes,verify_readonly)
    from rewriter.grouped_norm import GroupedNormObjective
    model = GPT(**source['config']['model']).to(device)
    model.load_state_dict(source['model']); model.requires_grad_(False)
    engine = FunctionalPreservation(model, teacher_cache_device=device, kl_dtype=torch.float64)
    saved = torch.load(Path(run)/'controller.pt', map_location=device, weights_only=False)
    if saved['step'] != 1000 or saved['config'] != record['config']:
        raise ValueError('controller checkpoint differs from completed frozen-run metadata')
    observed = torch.load(Path(run)/'observation.pt', map_location=device, weights_only=False)
    snapshot = observed['snapshot']
    scope=record['config'].get('editable_scope','all')
    objective=record['config'].get('norm_objective','global_squared')
    if objective not in ('global_squared','grouped_l2') or ((scope=='grouped')!=(objective=='grouped_l2')):
        raise ValueError('recorded edit scope and norm objective do not match')
    editable=editable_names(engine.parameters,scope)
    grouped_objective=GroupedNormObjective(engine.parameters) if objective=='grouped_l2' else None
    editable_norm_sq=(engine.base_weight_norm_sq if scope=='all' else editable_weight_norm_sq(engine.parameters,editable))
    if scope!='all' and not math.isclose(float(snapshot.weight_norm),float(editable_norm_sq.sqrt()),rel_tol=1e-6):
        raise ValueError('scoped observation weight norm includes excluded parameters or wrong weights')
    readonly=readonly_hashes(engine.parameters,editable)
    expected_radius = float(snapshot.weight_norm)*record['config']['radius_fraction']
    if not math.isclose(float(snapshot.radius), expected_radius, rel_tol=1e-6):
        raise ValueError('observation radius differs from declared source recipe')
    actor = GlobalRewriter(rank=8, trajectory_dim=(2*source['config']['model']['num_layers']+1)*16).to(device)
    # The producer did not call eval(); preserve its mode for exact replay.
    actor.load_state_dict(saved['controller'])
    action = scoped_action(actor(snapshot),editable)
    scales = {}
    for name, base in engine.parameters.items():
        rms = base.float().square().mean().sqrt().clamp_min(1e-3)
        dim = 8*(sum(base.shape)-8) if base.ndim == 2 else base.numel()
        scales[name] = rms*(base.numel()/dim)**.5
    action = RawAction({n:Factors(f.P*scales[n],f.Q*scales[n],f.anchors)
                        for n,f in action.matrices.items()},
                       {n:v*scales[n] for n,v in action.vectors.items()})
    norm = action.norm_sq().clamp_min(1e-24).sqrt()
    scale = torch.minimum(norm.new_ones(()), snapshot.radius/(norm+1e-12))
    edited = keep_readonly_parameters(engine.edited_parameters(action, scale),engine.parameters,editable)
    grouped=grouped_objective.evaluate(edited) if grouped_objective is not None else None
    if grouped is None:
        change, edit_sq = engine.realized_parameter_norm_change(edited)
        relative_change=change/editable_norm_sq
    else:
        change,edit_sq=grouped.squared_norm_change,grouped.edit_norm_sq
        relative_change=grouped.mean_relative_change
    stats = dict(norm_relative_change=float(relative_change),
                 realized_edit_norm=float(edit_sq.sqrt()), raw_action_norm=float(norm),
                 global_scale=float(scale),**(grouped.metrics() if grouped is not None else {}))
    checks = check_endpoint_statistics(stats, record)
    family_stats = family_edit_statistics(engine.parameters, edited)
    if not math.isclose(sum(r['edit_norm_sq'] for r in family_stats.values()), float(edit_sq), rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError('family edit accounting differs from whole-model norm')
    if any(engine.compute.as_dict().values()):
        raise RuntimeError('preparation unexpectedly evaluated GPT')
    result = dict(source['model'])
    result.update({n:v.detach().cpu() for n,v in edited.items()})
    return result, dict(endpoint_checks=checks, preparation_compute=engine.compute.as_dict(),
        source_extra_compute=record['finished']['compute'], calibration=observed['calibration'],
        family_edit_statistics=family_stats,
        parameter_scale_formula='max(parameter_RMS,1e-3)*sqrt(numel/accessible_dimension); dimension=r*(n+m-r) or vector.numel',
        radius=float(snapshot.radius), base_weight_norm=float(engine.base_weight_norm_sq.sqrt()),
        editable_scope=scope,norm_objective=objective,editable_weight_norm=float(editable_norm_sq.sqrt()),
        **(grouped.metrics() if grouped is not None else {}),
        read_only_integrity=verify_readonly(edited,editable,readonly))


def write_arm(source, model_state, config, directory, *, provenance):
    """Serialize independent rank states and hash every emitted file."""
    from baseline.checkpoint import save_atomic
    directory.mkdir()
    (directory/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    files = {}
    for rank in range(config['world_size']):
        state = reshard_checkpoint(source, config, rank, model_state=model_state)
        path = directory/f'latest_rank{rank}.pt'
        save_atomic(state,path)
        files[path.name] = dict(bytes=path.stat().st_size,sha256=sha256(path))
        del state
    files['config.json'] = dict(bytes=(directory/'config.json').stat().st_size,
                               sha256=sha256(directory/'config.json'))
    manifest = dict(arm=config['fork_label'],files=files,provenance=provenance,
                    continuation_start_step=source['step'],continuation_end_step=config['max_steps'],
                    training_launched=False,benchmark_compliant=False)
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--checkpoint-sha256',required=True)
    p.add_argument('--low-lr-run');p.add_argument('--high-lr-run')
    p.add_argument('--learned-run',action='append',default=[],metavar='LABEL=DIRECTORY',
                   help='Repeat for any predeclared completed arm, including a different recorded KL band')
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0')
    args = p.parse_args()
    if sha256(args.checkpoint) != args.checkpoint_sha256:
        p.error('source checkpoint SHA256 mismatch')
    source = torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    cfg = source['config']
    from baseline.train import validate_config
    validate_config(cfg)
    if (source['step'] != 1750 or cfg['seed'] != 0 or cfg['max_steps'] != 3250
            or cfg.get('schedule_steps') != 3250 or cfg.get('unscored',False)
            or cfg['microbatch_sequences'] != 64):
        p.error('requires the scored-shape seed0 PR323 checkpoint1750 on schedule3250')
    runs = {}
    for label,path in (('low_lr',args.low_lr_run),('high_lr',args.high_lr_run)):
        if path:runs[label]=Path(path)
    for item in args.learned_run:
        label,separator,path = item.partition('=')
        if not separator or not path or not re.fullmatch(r'[a-z][a-z0-9_]*',label) or label in runs or label == 'identity':
            p.error('learned-run requires unique nonidentity LABEL=DIRECTORY')
        runs[label]=Path(path)
    if not runs:p.error('provide at least one completed learned run')
    records = {label:validate_completed_run(run,cfg,args.checkpoint_sha256,1750) for label,run in runs.items()}
    for record in records.values():
        if not all(math.isfinite(record['config'][k]) and record['config'][k] > 0
                   for k in ('lr','kl_budget','norm_target_fraction','radius_fraction')):
            p.error('invalid recorded objective or optimizer scales')
    device = torch.device(args.device)
    if device.type == 'cuda' and device.index is None:device=torch.device('cuda:0')
    if device.type != 'cuda':
        p.error('learned reconstruction requires the producer CUDA arithmetic; CPU is for reshard tests only')
    target = Path(args.output).resolve()
    if target.exists():p.error('refusing to overwrite an existing fork output')
    # Each arm x eight ranks replicates model/Adam; allow a conservative source-size bound.
    target.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(target.parent).free < 8*(len(runs)+1)*Path(args.checkpoint).stat().st_size + 1024**3:
        p.error('insufficient free disk for requested eight-rank forks (conservative bound)')
    # Validate structural conversion before CUDA initialization/output staging.
    common = dict(source_checkpoint=str(Path(args.checkpoint).resolve()), source_checkpoint_sha256=args.checkpoint_sha256,
        source_step=1750, source_world_size=1, target_world_size=8,
        source_elapsed_seconds=source['elapsed_seconds'],source_counters=source['counters'],
        source_config=cfg, logical_prefix_steps=1750, remaining_gpt_steps=1500,
        additional_compute_nonbenchmark=True,continuously_online_controller=False,
        same_seed_prefix_only=True, source_data_cursor=source['data_cursor'])
    probe = reshard_checkpoint(source, continuation_config(cfg,label='identity',provenance=common),0)
    del probe
    from baseline.execution import enable_deterministic_execution, execution_environment
    torch.set_num_interop_threads(20);enable_deterministic_execution();torch.cuda.set_device(device)
    common.update(preparation_environment=execution_environment(),torch=torch.__version__,cuda=torch.version.cuda,
                  source_files={str(path.relative_to(ROOT)):sha256(path) for path in
                      [Path(__file__).resolve(),ROOT/'baseline/checkpoint.py',ROOT/'baseline/optim.py',ROOT/'rewriter/controller.py',
                       ROOT/'rewriter/algebra.py',ROOT/'rewriter/function_preservation.py',ROOT/'rewriter/edit_scope.py',ROOT/'rewriter/grouped_norm.py',
                       ROOT/'scripts/train_frozen_preservation.py']})
    stage = Path(tempfile.mkdtemp(prefix='.'+target.name+'.incomplete-',dir=target.parent))
    manifests = {}
    try:
        for label in ('identity',*runs):
            metadata = copy.deepcopy(common)
            if label == 'identity':
                model_state = source['model']
                metadata.update(edit='identity',preparation_gpt_forwards=0,preparation_gpt_backwards=0)
            else:
                model_state, measurements = reconstruct_frozen_edit(source,runs[label],records[label],device=device)
                metadata.update(edit='one_shot_learned_anchored',frozen_run=str(runs[label].resolve()),
                    frozen_run_files=records[label]['files'],frozen_run_config=records[label]['config'],**measurements)
            config = continuation_config(cfg,label=label,provenance=metadata)
            manifests[label] = write_arm(source,model_state,config,stage/label,provenance=metadata)
            del model_state
            torch.cuda.empty_cache()
        (stage/'manifest.json').write_text(json.dumps(dict(arms=manifests,training_launched=False,
            benchmark_compliant=False,preparation_gpt_forwards=0,preparation_gpt_backwards=0),indent=2)+'\n')
        os.replace(stage,target)
    except Exception:
        # Keep potentially useful partial files visibly marked incomplete; never resume them.
        print('FAILED: incomplete artifacts preserved at '+str(stage),file=sys.stderr)
        raise
    print(json.dumps(dict(output=str(target),arms=list(manifests),training_launched=False)))


if __name__ == '__main__':
    main()
