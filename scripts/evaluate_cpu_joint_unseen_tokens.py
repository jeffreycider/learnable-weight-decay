"""Reserved TRAIN sequences unseen by Chisel at already-known GPT checkpoints.

Read-only evaluation of the joint training_state schema. This is not a heldout
checkpoint, validation-set, full-bank, or generalization qualification. Teacher
probabilities come from fresh original-state CPU forwards, one microbatch at a
time; neither GPU logits nor old CPU teacher caches are reused.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baseline.model import GPT
from rewriter.function_preservation import FunctionalPreservation
from rewriter.grouped_coordinate_chisel import GroupedCoordinateChisel
from rewriter.grouped_norm import GroupedNormObjective
from rewriter.prefix_cache import unpack_record
from rewriter.edit_scope import editable_names,editable_weight_norm_sq
from rewriter.tokenwise_reference import TokenwiseConfig,TokenwiseDual
from rewriter.ranking_dual import RankingDual
from scripts.fit_cpu_edit_teacher import sha,exact_state_equal
from scripts.prepare_cpu_teacher_state import validate_train_panel
from scripts.diagnose_cpu_norm_preservation import LocalCPUContexts,load_bound_state,tensor_sha,pin_sources


def validate_endpoint(checkpoint,result):
    """Require the reported final, forward-only fitted-panel acceptance point."""
    step=checkpoint['completed_updates'];final=result['final'];cases=checkpoint['metadata']['cases']
    if type(step) is not int or step<1 or result['completed_updates']!=step or final['completed_updates_before_measure']!=step or final['updates_after']!=step or final['learning'] or final['gradient_profile']:
        raise ValueError('training endpoint/counter mismatch')
    if checkpoint['dual']['updates']!=step:raise ValueError('training dual counter mismatch')
    rows={r['local_id']:r for r in final['rows']}
    if len(rows)!=2 or len(final['rows'])!=2 or set(rows)!={c['state_id'] for c in cases}:raise ValueError('exactly two matching endpoint cases required')
    for c in cases:
        r=rows[c['state_id']]
        if r['context_ids']!=c['sequence_ids']:raise ValueError('fitted endpoint token IDs mismatch')
        if r['strict_topk_changes']!=0 or not math.isfinite(r['norm_objective']) or r['norm_objective']>=0 or not math.isfinite(r['mean_group_l2_percentage_reduction']) or r['mean_group_l2_percentage_reduction']<=0:
            raise ValueError('training endpoint must preserve strict top100 and shrink both states')
        if r.get('ranking_k')!=100:raise ValueError('training endpoint must use top100')
    return rows


def select_unseen_inputs(panel,case,start,count):
    ids=panel['metadata']['sequence_ids'];trained=case['sequence_ids']
    if len(ids)!=len(set(ids)) or len(trained)!=len(set(trained)):raise ValueError('ambiguous sequence identities')
    if start<0 or count<1 or start+count>len(panel['inputs']) or start+count>len(ids):raise ValueError('invalid unseen panel slice')
    positions={uid:i for i,uid in enumerate(ids)}
    if any(uid not in positions for uid in trained):raise ValueError('fitted sequence missing from panel')
    old=panel['inputs'][[positions[uid] for uid in trained]]
    if tensor_sha(old)!=case['input_sha256']:raise ValueError('fitted input hash differs from panel')
    chosen=ids[start:start+count]
    if set(chosen)&set(trained):raise ValueError('unseen IDs overlap Chisel fitting IDs')
    return panel['inputs'][start:start+count].clone(),chosen


def validate_seed_identity(case,seed_metadata,actor,private):
    if seed_metadata['state_sha256']!=case['sha256']['state'] or actor.configuration()!=seed_metadata['architecture']:
        raise ValueError('shared architecture/state identity mismatch')
    if private.get_extra_state()!=case['private_identity']:raise ValueError('saved private identity mismatch')


def restore_evaluation_dual(ids,config,metadata,state):
    spec=metadata.get('ranking_constraint')
    if spec is None:
        dual=TokenwiseDual(ids,config=config,dual_lr=metadata['dual_lr'])
    else:
        if spec['tolerance']!=0. or spec['measurement']!='mean_nonnegative_relative_gap_margin_loss':raise ValueError('unsupported ranking constraint')
        dual=RankingDual(ids,config=config,dual_lr=metadata['dual_lr'],rank_scale=spec['scale'],rank_dual_lr=spec['dual_lr'])
    dual.load_state_dict(state)
    return dual


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--training-result',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--start',type=int,default=2);parser.add_argument('--sequences',type=int,default=2)
    a=parser.parse_args()
    if not 1<=a.sequences<=16:raise ValueError('bounded evaluation requires 1 to 16 new sequences per checkpoint')
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');started=time.perf_counter()
    sources=pin_sources(a.output);checkpoint_sha=sha(a.checkpoint);result_sha=sha(a.training_result)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False,mmap=True);training=json.loads(a.training_result.read_text());accepted=validate_endpoint(saved,training)
    expected=a.training_result.parent/f"checkpoint_{saved['completed_updates']:04d}.pt"
    if a.checkpoint.resolve()!=expected.resolve():raise ValueError('checkpoint path is not corresponding recorded endpoint')
    metadata_path=a.training_result.parent/'metadata.json'
    if json.loads(json.dumps(saved['metadata']))!=json.loads(metadata_path.read_text()):raise ValueError('checkpoint metadata differs from completed run')
    meta=saved['metadata'];config=TokenwiseConfig(**meta['objective']);ids=[c['state_id'] for c in meta['cases']]
    if config.ranking_k!=100 or config.ranking_tie_policy!='exact_teacher_ties':raise ValueError('requires original strict top100 objective')
    dual=restore_evaluation_dual(ids,config,meta,saved['dual']);dual_before=copy.deepcopy(dual.state_dict())
    actor=GroupedCoordinateChisel(rank=8,trajectory_dim=400,model_dim=256,num_layers=2,latent_tokens=8,fresh_weight_core=True,factor_precision='fp64',residual_precision='fp32')
    actor.load_state_dict(saved['actor'],strict=True);actor.eval().requires_grad_(False)
    engine=None;contexts=LocalCPUContexts();rows=[];case_records=[]
    for case in meta['cases']:
        tick=time.perf_counter();paths=case['paths']
        for name,path in paths.items():
            if sha(path)!=case['sha256'][name]:raise ValueError('original case source hash changed: '+name)
        state=torch.load(paths['state'],map_location='cpu',weights_only=False,mmap=True);panel=torch.load(paths['panel'],map_location='cpu',weights_only=False,mmap=True)
        panel_meta=json.loads(Path(paths['panel_metadata']).read_text());validate_train_panel(panel,panel_meta,case['sha256']['state'],case['sha256']['panel'])
        if state['completed_steps']!=case['state_id']:raise ValueError('state counter identity mismatch')
        inputs,newids=select_unseen_inputs(panel,case,a.start,a.sequences);del panel
        seed=torch.load(paths['checkpoint'],map_location='cpu',weights_only=False,mmap=True)['metadata'];snapshot=unpack_record(state['observation'],device='cpu').snapshot
        private=actor.make_private(snapshot,state_key=('cpu-teaching',0,case['state_id'],case['sha256']['state']),seed=seed['args']['private_seed'])
        private.load_state_dict(saved['private'][case['state_id']],strict=True);private.requires_grad_(False);validate_seed_identity(case,seed,actor,private)
        if engine is None:
            base=state['model'];depth=len({n.split('.')[1] for n in base if n.startswith('blocks.')})
            model=GPT(base['embed.weight'].shape[0],depth,base['embed.weight'].shape[1]).float().eval().requires_grad_(False)
            engine=FunctionalPreservation(model,microbatch_sequences=1,chunk_tokens=128,teacher_cache_device='cpu',kl_dtype=torch.float64,gpt_precision='fp32')
        load_bound_state(engine,state['model'],case['sha256']['state']);load_id=engine.teacher_id
        def teacher(start,stop):
            if engine.teacher_id!=load_id:raise ValueError('teacher was reloaded during evaluation')
            return engine.cache_teacher(inputs[start:stop],data_role='past_training',sequence_ids=newids[start:stop])[0]
        grouped=GroupedNormObjective(engine.parameters,zero_reference_policy='freeze');editable=editable_names(engine.parameters,'grouped')-grouped.frozen_parameter_names
        context=dict(domain='gpt',local_id=case['state_id'],engine=engine,snapshot=snapshot,grouped=grouped,inputs=inputs,ids=newids,teacher=teacher,norm_squared=editable_weight_norm_sq(engine.parameters,editable))
        case_config=dual.config_for_state(case['state_id']) if isinstance(dual,RankingDual) else config
        row=contexts.measure(context,actor,private,case_config,dual,learning=False,weight=.5)
        if isinstance(dual,RankingDual):row.update(dual.describe_measurement(case['state_id'],row['ranking_penalty']))
        row['elapsed_seconds']=time.perf_counter()-tick;rows.append(row)
        if not exact_state_equal(private.state_dict(),saved['private'][case['state_id']]):raise ValueError('private state changed during evaluation')
        if engine.compute.teacher_forwards!=a.sequences or engine.compute.student_forwards!=a.sequences or engine.compute.student_backwards!=0:raise ValueError('unexpected GPT evaluation counts')
        case_records.append(dict(state_id=case['state_id'],source_paths=paths,source_sha256=case['sha256'],fitted_ids=case['sequence_ids'],evaluated_ids=newids,panel_slice=[a.start,a.start+a.sequences],input_sha256=tensor_sha(inputs),seed_provenance=case.get('seed_provenance'),training_endpoint=accepted[case['state_id']]))
        print(json.dumps(row,allow_nan=False),flush=True);del context,grouped,private,state
    if not exact_state_equal(actor.state_dict(),saved['actor']) or not exact_state_equal(dual.state_dict(),dual_before):raise ValueError('actor or dual changed during evaluation')
    if any(sha(name)!=digest for name,digest in sources.items()):raise ValueError('evaluation sources changed')
    result=dict(scope=__doc__,checkpoint_sha256=checkpoint_sha,training_result_sha256=result_sha,training_metadata_sha256=sha(metadata_path),completed_training_updates=saved['completed_updates'],source_sha256=sources,training_source_sha256=meta['source_sha256'],cases=case_records,rows=rows,compute=contexts.compute,gpt_teacher_forwards=contexts.compute['teacher_forwards'],gpt_student_forwards=contexts.compute['student_forwards'],gpt_forwards=contexts.compute['teacher_forwards']+contexts.compute['student_forwards'],gpt_backwards=0,optimizer_updates=0,dual_updates=0,actor_private_dual_unchanged=True,teacher_origin='fresh CPU original-checkpoint forwards; no probability/cache reuse',unseen_by_chisel_tokens=True,heldout_checkpoint_claim=False,full_bank_claim=False,elapsed_seconds=time.perf_counter()-started)
    (a.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
