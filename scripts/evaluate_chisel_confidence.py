"""Frozen-policy preservation confidence audit; no gradients, selection, or updates.

Reads one saved Chisel checkpoint, the verified GPT state bank, and the exact
corrected teacher panel cache the policy was fitted against, then reports
per-sequence and per-token KL/CE distributions for explicitly named populations
(see rewriter.confidence_audit.POPULATIONS), explicit early checkpoints, the
complete fresh next batch of a declared checkpoint, and an optional per-group
edit comparison against a reference policy (no extra GPT passes). The saved
action is used exactly as saved. Extra GPT forwards are counted and are not
reference-budget compliant. Nothing here feeds LR, stopping, or policy choice.
"""
from __future__ import annotations
import argparse,glob,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baseline.model import GPT
from baseline.execution import enable_deterministic_execution
from baseline.train import source_record
from rewriter.state_bank import StateBankReader
from rewriter.teacher_cache import TeacherCacheReader
from rewriter.chisel_factory import build_chisel,validate_checkpoint_architecture
from rewriter.function_preservation import FunctionalPreservation
from rewriter.gpt_precision import configure_precision,precision_metadata
from rewriter.grouped_norm import GroupedNormObjective
from rewriter.replay_buffer import next_training_sequences
from rewriter.confidence_audit import POPULATIONS,token_kl_and_ce,sequence_summary,population_summary,compare_edits
from rewriter.edit_scope import readonly_hashes,verify_readonly
from scripts.train_multistate_chisel import build_proposal,all_rank_objects,heldout_sequence,choose_audit_states
from scripts.evaluate_chisel_fresh import checkpoint_gpt_precision

EARLY_STATES=tuple(range(5,17))
PARITY_TOLERANCE=1e-9  # nats/token; FP64 sums in two orders over identical FP32 logits


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--state-bank',required=True,type=Path)
    parser.add_argument('--teacher-cache',required=True,type=Path,help='exact corrected teacher panels used for fitting')
    parser.add_argument('--train-data',required=True)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--fitting-states',nargs='*',type=int,help='fitting checkpoints; default: the trainer\'s 32 fixed panels plus early fitting states 5-16')
    parser.add_argument('--all-fitting-states',action='store_true',help='audit every fitting (train-split) checkpoint; overrides --fitting-states')
    parser.add_argument('--heldout-states',nargs='*',type=int,help='checkpoint-held-out states; default: all; pass with no values to audit none')
    parser.add_argument('--no-fitting-seen',action='store_true',help='skip the 128-sequence seen panels')
    parser.add_argument('--no-fitting-withheld',action='store_true',help='skip the 64-sequence withheld panels of the fitting checkpoints (phase-1 seen-only audits)')
    parser.add_argument('--checkpoint-sha256',help='required SHA256 of --checkpoint; mismatch aborts before loading')
    parser.add_argument('--reference-sha256',help='required SHA256 of --reference-checkpoint; mismatch aborts before loading')
    parser.add_argument('--fresh-state',type=int,help='must be the final bank checkpoint; its complete next ordinary batch is unseen by every fitting panel')
    parser.add_argument('--fresh-sequences',type=int,default=0,help='0 = the complete next batch (per batch when --fresh-batches > 1)')
    parser.add_argument('--fresh-batches',type=int,default=1,help='number of consecutive ordinary batches after the final checkpoint to audit (all unseen by every bank checkpoint); each batch is its own audit item')
    parser.add_argument('--reference-checkpoint',type=Path,help='second frozen policy of the same bank (e.g. the pre-repair policy); per-group edit cosine/amplitude against it, no extra GPT passes')
    parser.add_argument('--gpt-microbatch',type=int,help='default: the cached panel generation microbatch')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--eager-chisel',action='store_true')
    parser.add_argument('--verify-state-once',action='store_true');parser.add_argument('--mmap-state',action='store_true')
    parser.add_argument('--bootstrap-seed',type=int,default=0)
    parser.add_argument('--plots',action='store_true')
    args=parser.parse_args(argv)
    if args.fresh_sequences<0:parser.error('nonnegative fresh-sequences required')
    if args.fresh_batches<1:parser.error('fresh-batches must be positive')
    if args.reference_sha256 and args.reference_checkpoint is None:parser.error('reference-sha256 needs reference-checkpoint')
    return args


def main(argv=None):
    args=parse_args(argv)
    world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'))
    local=int(os.environ.get('LOCAL_RANK','0'));device=torch.device(f'cuda:{local}' if args.device=='cuda' else 'cpu')
    if device.type=='cuda':enable_deterministic_execution();torch.cuda.set_device(device)
    else:torch.set_num_threads(1)
    if world>1:dist.init_process_group('nccl' if device.type=='cuda' else 'gloo')
    def pinned(path,expected,label):
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if expected and digest!=expected.lower():raise ValueError(f'{label} SHA256 {digest} differs from the pinned {expected}')
        return digest
    checkpoint_digest=pinned(args.checkpoint,args.checkpoint_sha256,'checkpoint')
    if args.reference_checkpoint is not None:reference_digest=pinned(args.reference_checkpoint,args.reference_sha256,'reference checkpoint')
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    gpt_precision=checkpoint_gpt_precision(saved);configure_precision(gpt_precision)
    metadata=saved['metadata'];options=metadata['options'];architecture=metadata['architecture']
    validate_checkpoint_architecture(saved,architecture)
    identity=metadata['state_bank_identity']
    if hashlib.sha256((args.state_bank/'manifest.json').read_bytes()).hexdigest()!=identity['manifest_sha256']:
        raise ValueError('checkpoint and GPT state bank differ')
    cache_sha=hashlib.sha256((args.teacher_cache/'manifest.json').read_bytes()).hexdigest()
    if identity.get('teacher_cache_manifest_sha256') not in (None,cache_sha):
        raise ValueError('checkpoint was fitted against a different teacher cache')
    reader=StateBankReader(args.state_bank,verify_once=args.verify_state_once,mmap_model=args.mmap_state)
    cache=TeacherCacheReader(args.teacher_cache,verify_once=args.verify_state_once,mmap=args.mmap_state)
    if cache.manifest.get('execution',{}).get('gpt_precision','baseline')!=gpt_precision:
        raise ValueError('teacher cache and checkpoint GPT precision differ')
    provenance=reader.manifest['provenance'];training=provenance['training_config']
    split=reader.manifest['split'];train_ids=list(split['train']);audit_ids=list(split['audit'])
    panel_microbatch=next(iter(cache.expected.values()))['microbatch_sequences']
    microbatch=args.gpt_microbatch or panel_microbatch
    if microbatch!=panel_microbatch:raise ValueError('cached panels require their generation microbatch')
    if args.all_fitting_states:fitting=list(train_ids)
    else:fitting=sorted(set(args.fitting_states) if args.fitting_states is not None else
                   set(choose_audit_states(train_ids,options.get('training_audit_states') or 32))|{s for s in EARLY_STATES if s in train_ids})
    heldout=sorted(set(args.heldout_states)) if args.heldout_states is not None else audit_ids
    if not set(fitting)<=set(train_ids) or not set(heldout)<=set(audit_ids):
        raise ValueError('fitting states must be train-split and held-out states audit-split checkpoints')
    final_state=max(set(train_ids)|set(audit_ids))
    if args.fresh_state is not None and args.fresh_state!=final_state:
        raise ValueError(f'fresh next batch is globally unseen only after the final bank checkpoint {final_state}; later fitting panels sample earlier next batches')
    actor=build_chisel(architecture).to(device);actor.load_state_dict(saved['controller'])  # parameters keep requires_grad as fitted; no backward is ever called
    policy=actor if args.eager_chisel else torch.compile(actor,fullgraph=True,dynamic=False)
    reference_policy=None;reference_info=None
    if args.reference_checkpoint is not None:
        reference=torch.load(args.reference_checkpoint,map_location='cpu',weights_only=False)
        validate_checkpoint_architecture(reference,architecture)
        if reference['metadata']['state_bank_identity']!=identity or reference['metadata']['architecture']!=architecture:
            raise ValueError('reference policy belongs to another bank or architecture')
        reference_actor=build_chisel(architecture).to(device);reference_actor.load_state_dict(reference['controller'])
        reference_policy=reference_actor if args.eager_chisel else torch.compile(reference_actor,fullgraph=True,dynamic=False)
        reference_info=dict(path=str(args.reference_checkpoint),sha256=reference_digest,sha256_pinned=bool(args.reference_sha256),
                            update=reference['updates'],epoch=reference['next_epoch'],
                            readout='per-group cosine/amplitude of realized edits; attenuation has cosine near one and ratio below one')
    engine=FunctionalPreservation(GPT(**provenance['model_config']).to(device).requires_grad_(False),
        microbatch_sequences=microbatch,teacher_cache_device=device,kl_dtype=torch.float64,gpt_precision=gpt_precision)
    files=sorted(glob.glob(args.train_data))
    if not files:raise ValueError('no training shards')
    tolerance=float(options.get('kl_tolerance',0) or 0) or None
    if tolerance is None:raise ValueError('checkpoint lacks a KL tolerance; the audit reports against the trainer tolerance')
    proposal_kwargs={'shrinkage_coefficient':options['shrinkage_coefficient']} if 'shrinkage_coefficient' in options else {}
    items=[]
    for state_id in fitting:
        if not args.no_fitting_seen:items.append((state_id,'fitting_seen'))
        if not args.no_fitting_withheld:items.append((state_id,'fitting_withheld'))
    items+=[(state_id,'checkpoint_heldout_development') for state_id in heldout]
    if args.fresh_state is not None:items+=[(args.fresh_state,'fresh_next_batch',k) for k in range(args.fresh_batches)]
    items=[(i if len(i)==3 else (*i,0)) for i in items]
    if not items:raise ValueError('no audit items selected')
    config=dict(checkpoint=str(args.checkpoint),checkpoint_sha256=checkpoint_digest,checkpoint_sha256_pinned=bool(args.checkpoint_sha256),
        checkpoint_update=saved['updates'],checkpoint_epoch=saved['next_epoch'],state_bank_manifest_sha256=identity['manifest_sha256'],
        teacher_cache_manifest_sha256=cache_sha,gpt_precision=precision_metadata(gpt_precision),kl_tolerance=tolerance,
        populations=POPULATIONS,fitting_states=fitting,all_fitting_states=args.all_fitting_states,fitting_seen=not args.no_fitting_seen,
        fitting_withheld=not args.no_fitting_withheld,heldout_states=heldout,fresh_state=args.fresh_state,
        fresh_sequences=args.fresh_sequences or 'complete next batch',fresh_batches=args.fresh_batches,gpt_microbatch=microbatch,
        reference_checkpoint=reference_info,grouped_norm_backend='reference GroupedNormObjective (shrink metrics only; fitted run used '+str(options.get('norm_backend'))+')',
        items=len(items),world_size=world,gradient_or_optimizer_updates=False,policy_selection=False,
        official_gpt_validation=False,reference_budget_compliant=False,
        early_states_requested=[s for s in EARLY_STATES if s in set(fitting)|set(heldout)],
        heldout_panel_role='development validation: trainer LR/stopping consumed heldout_states feasibility scores',
        bootstrap_note='percentile bootstrap over audited units only; not a certificate for unaudited inputs')
    if rank==0:
        args.output.mkdir(parents=True,exist_ok=False)
        (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n');source_record(config,args.output)
    if world>1:dist.barrier()
    names=set(engine.parameters);rows=[];started=time.perf_counter();comparisons={};token_pools={p:[] for p in POPULATIONS}
    # load_engine -> load_teacher_state resets engine.compute for every checkpoint, so counters
    # are accumulated per item (deltas against a baseline taken right after the load) before the
    # next rebind. Cached teacher-head reconstructions are counted separately from full teacher
    # transformer forwards, which only the fresh population performs.
    COUNTERS=('student_forwards','student_tokens','student_backwards','student_backward_tokens','teacher_forwards','teacher_tokens')
    HEAD=('cached_teacher_head_forwards','cached_teacher_head_tokens')
    local_compute={k:0 for k in COUNTERS+HEAD};local_compute_by_population={}
    def snapshot():
        return dict(**{k:engine.compute.as_dict().get(k,0) for k in COUNTERS},**{k:getattr(engine,k,0) for k in HEAD})
    for state_id,population,batch_offset in items[rank::world]:
        record=reader.read(state_id,device=device,model_device='cpu')
        cache.load_engine(engine,record)
        baseline=snapshot()
        grouped=GroupedNormObjective(engine.parameters,zero_reference_policy='freeze')
        if population=='fresh_next_batch':
            available=training['batch_tokens']//training['seq_len']
            # batch_offset k audits the (k+1)-th ordinary batch after the final checkpoint; every
            # such batch lies after all bank checkpoints' past-data pools.
            batch=next_training_sequences(files,state_id+batch_offset,batch_tokens=training['batch_tokens'],seq_len=training['seq_len'],
                                          count=available,seed=600001)
            order=sorted(range(available),key=lambda i:batch.provenance[i]['sequence'])
            if args.fresh_sequences:order=order[:args.fresh_sequences]
            inputs=batch.inputs[order];ids=[batch.provenance[i]['id'] for i in order]
            roles=['audit' if heldout_sequence(i,options['split_seed']) else 'train' for i in ids]
            provider=lambda start,stop:engine.cache_teacher(inputs[start:stop],data_role='fresh_next_training_audit',sequence_ids=ids[start:stop])[0]
        else:
            panel=cache.read(state_id,'train' if population=='fitting_seen' else 'audit')
            inputs=panel.inputs;ids=list(panel.sequence_ids)
            expected_role='train' if population=='fitting_seen' else 'audit'
            if any(heldout_sequence(i,options['split_seed'])!=(expected_role=='audit') for i in ids):
                raise ValueError('cached panel violates the global sequence split')
            roles=[expected_role]*len(ids)
            provider=lambda start,stop:panel.teacher_batch(engine,start,stop)
        if True:  # one frozen policy per item; the action is used exactly as saved
            with torch.enable_grad():  # match fitting-time actor arithmetic; GPT stays no-grad
                parameters,regularizer,stats,editable,credit=build_proposal(policy,record.snapshot,engine,grouped,
                    radius_fraction=options['radius_fraction'],**proposal_kwargs)
            parameters={n:v.detach() for n,v in parameters.items()}
            del regularizer,credit
            comparison=None
            if reference_policy is not None:
                if state_id not in comparisons:
                    with torch.enable_grad():
                        reference_parameters,_r,_s,reference_editable,reference_credit=build_proposal(reference_policy,record.snapshot,engine,grouped,
                            radius_fraction=options['radius_fraction'],**proposal_kwargs)
                    if reference_editable!=editable:raise ValueError('reference policy edits a different parameter scope')
                    comparisons[state_id]=compare_edits(parameters,{n:v.detach() for n,v in reference_parameters.items()},engine.parameters,grouped.groups,editable)
                    del reference_parameters,_r,_s,reference_credit
                comparison=comparisons[state_id]
            kls=[];student=[];teacher=[];production=0.;parity=0.
            with torch.no_grad():
                for start in range(0,len(inputs),microbatch):
                    stop=min(start+microbatch,len(inputs))
                    measured=token_kl_and_ce(engine,parameters,provider(start,stop))
                    kls.append(measured['kl']);student.append(measured['student_nll']);teacher.append(measured['teacher_nll'])
                    production+=measured['production_mean_kl']*(stop-start)/len(inputs)
                    parity=max(parity,measured['parity_abs_difference'])
                integrity=verify_readonly(parameters,editable,readonly_hashes(engine.parameters,editable))
            if parity>PARITY_TOLERANCE:raise AssertionError(f'per-token KL mean departs from production chunked KL by {parity:.3e}')
            kl_tokens=torch.cat(kls)
            token_pools[population].append(kl_tokens.flatten().float())  # exact pooled quantiles need every token; FP32 storage of FP64 values
            summary=sequence_summary(kl_tokens,torch.cat(student),torch.cat(teacher),tolerance=tolerance,
                                     bootstrap_seed=args.bootstrap_seed)
            rows.append(dict(state_id=state_id,population=population,population_definition=POPULATIONS[population],
                fresh_batch_offset=batch_offset if population=='fresh_next_batch' else None,
                sequence_ids=ids,sequence_roles=roles,production_mean_kl=production,parity_abs_difference=parity,
                summary=summary,read_only_unchanged=integrity['unchanged'],reference_comparison=comparison,
                **{k:stats[k] for k in ('mean_group_l2_percentage_reduction','group_l2_percentage_reduction','realized_edit_norm',
                                         'raw_action_norm','global_scale','hard_radius','radius_saturated','proposed_edit_norm')}))
            del parameters
        after=snapshot();delta={k:after[k]-baseline[k] for k in after}
        if delta['student_tokens']!=rows[-1]['summary']['tokens']:raise AssertionError('student token count differs from the audited tokens of this item')
        for k,v in delta.items():local_compute[k]+=v
        bucket=local_compute_by_population.setdefault(population,{k:0 for k in after});bucket.update({k:bucket[k]+v for k,v in delta.items()});bucket['items']=bucket.get('items',0)+1
        if rank==0:print(json.dumps(dict(event='progress',completed_local_items=len(rows),elapsed_seconds=time.perf_counter()-started)),flush=True)
        del record,grouped,inputs,provider
    rows=sorted([r for group in all_rank_objects(rows,world) for r in group],key=lambda r:(r['population'],r['state_id'],r.get('fresh_batch_offset') or 0))
    # Per-token pools: each rank writes one compact float32 file per population to the
    # shared output directory; rank zero concatenates them for exact NumPy quantiles.
    # Nothing per-token crosses the collective layer (the production pool is ~1e8 tokens).
    # The output directory is created fresh by this run (exist_ok=False above), so no earlier
    # run's files can be present; each rank additionally refuses to overwrite and records its
    # token count, and rank zero verifies file set, world size and counts before pooling.
    pool_dir=args.output/'token_pools';pool_dir.mkdir(exist_ok=True)
    local_counts={}
    for population in POPULATIONS:
        if token_pools[population]:
            path=pool_dir/f'{population}_rank{rank}.npy'
            if path.exists():raise FileExistsError(f'stale token pool present: {path}')
            values=torch.cat(token_pools[population]).numpy()
            np.save(path,values);local_counts[population]=int(values.size)
    (pool_dir/f'manifest_rank{rank}.json').write_text(json.dumps(dict(rank=rank,world_size=world,counts=local_counts,
        dtype='float32 downcast of per-token FP64 KL; pooled quantiles are exact for these stored values, not bit-identical to an FP64 pool'))+'\n')
    token_pools.clear()
    if world>1:dist.barrier()
    pooled={}
    if rank==0:
        manifests=[json.loads((pool_dir/f'manifest_rank{r}.json').read_text()) for r in range(world)]
        if any(m['rank']!=r or m['world_size']!=world for r,m in enumerate(manifests)):raise AssertionError('token pool manifests do not match this run')
        for population in POPULATIONS:
            parts=[]
            for r,m in enumerate(manifests):
                if population in m['counts']:
                    part=np.load(pool_dir/f'{population}_rank{r}.npy',mmap_mode='r')
                    if part.dtype!=np.float32 or part.size!=m['counts'][population]:raise AssertionError('token pool file differs from its manifest')
                    parts.append(part)
            if parts:pooled[population]=np.concatenate(parts).astype(np.float64)
    compute={}
    for group in all_rank_objects(local_compute,world):
        for key,value in group.items():compute[key]=compute.get(key,0)+value
    compute_by_population={}
    for group in all_rank_objects(local_compute_by_population,world):
        for population,counts in group.items():
            bucket=compute_by_population.setdefault(population,{})
            for key,value in counts.items():bucket[key]=bucket.get(key,0)+value
    if compute.get('student_backwards',0):raise AssertionError('confidence audit performed a backward')
    if compute['student_tokens']!=sum(r['summary']['tokens'] for r in rows):raise AssertionError('accumulated student tokens differ from the audited rows')
    compute['accounting']='accumulated per item across checkpoint loads; cached_teacher_head_* are head-only reconstructions from the exact cache, teacher_forwards are full teacher transformer forwards (fresh population only)'
    if len(rows)!=len(items):raise AssertionError('missing or duplicated audit rows')
    if rank==0:
        populations={}
        for population in POPULATIONS:
            selected=[r for r in rows if r['population']==population]
            if selected:populations[population]=population_summary(selected,tolerance=tolerance,bootstrap_seed=args.bootstrap_seed,pooled_tokens=pooled[population])
        del pooled
        summary=dict(config=config,populations=populations,compute=compute,compute_by_population=compute_by_population,
                     token_pools='token_pools/<population>_rank<r>.npy with manifest_rank<r>.json: per-token KL computed in FP64 and stored as float32 (relative rounding ~6e-8); pooled quantiles are exact over these stored values, not bit-identical to quantiles of the unstored FP64 values. Per-sequence means and all per-state statistics remain FP64.',
                     elapsed_seconds=time.perf_counter()-started,
                     fresh_sequence_roles={role:sum(r['sequence_roles'].count(role) for r in rows if r['population']=='fresh_next_batch')
                                           for role in ('train','audit')} if args.fresh_state is not None else None)
        (args.output/'states.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        (args.output/'summary.md').write_text(markdown_table(summary))
        if args.plots:plot(rows,args.output,tolerance)
        print(json.dumps(dict(event='finished',compute=compute,elapsed_seconds=summary['elapsed_seconds'])),flush=True)
    if world>1:dist.destroy_process_group()


def markdown_table(summary):
    lines=['| population | states | seq | mean of state means KL | state p90 | state max | states <= tol | pooled seq p99 | pooled token p99.9 | mean CE change | 15-group shrink % |','|---|---|---|---|---|---|---|---|---|---|---|']
    for key,p in summary['populations'].items():
        lines.append(f"| {key} | {p['states']} | {p['sequences']} | {p['mean_of_state_means_kl']:.3e} | {p['state_mean_kl_quantiles']['p90']:.3e} | {p['state_mean_kl_quantiles']['p100']:.3e} | {p['states_with_mean_kl_at_or_below_tolerance']}/{p['states']} | {p['pooled_sequence_kl_quantiles']['p99']:.3e} | {p['pooled_token_kl_quantiles']['p99.9']:.3e} | {p['mean_ce_change']:+.2e} | {p['mean_group_l2_percentage_reduction']:.4f} |")
    lines.append('');lines.append(f"KL in nats/token, teacher||student, FP64; tolerance {summary['config']['kl_tolerance']:g} applies to checkpoint-panel means. "
                 'checkpoint_heldout_development panels were consumed by LR/stopping decisions. Bootstrap intervals are in summary.json.')
    return '\n'.join(lines)+'\n'


def plot(rows,output,tolerance):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4.5))
    for population in POPULATIONS:
        values=sorted(v for r in rows if r['population']==population for v in r['summary']['per_sequence_kl'])
        if not values:continue
        ax.plot(values,[(i+1)/len(values) for i in range(len(values))],label=f'{population} (n={len(values)} sequences)')
    ax.axvline(tolerance,color='k',ls='--',lw=.8,label='checkpoint-mean tolerance')
    ax.set_xscale('log');ax.set_xlabel('per-sequence mean KL(teacher||student), nats/token');ax.set_ylabel('fraction of sequences');ax.legend(fontsize=7)
    ax.set_title('Frozen policy: per-sequence KL by population')
    fig.tight_layout();fig.savefig(output/'confidence_audit.png',dpi=130);plt.close(fig)


if __name__=='__main__':main()
