"""Post-fit fresh-sequence test; no gradients, LR selection, or policy updates."""
from __future__ import annotations
import argparse,glob,hashlib,json,math,os,sys,time
from pathlib import Path
import torch
import torch.distributed as dist
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baseline.model import GPT
from baseline.execution import enable_deterministic_execution
from baseline.train import source_record
from rewriter.state_bank import StateBankReader
from rewriter.chisel_factory import build_chisel,validate_checkpoint_architecture
from rewriter.function_preservation import FunctionalPreservation
from rewriter.gpt_precision import precision_metadata
from rewriter.grouped_norm import GroupedNormObjective
from rewriter.primal_dual import PrimalDual
from scripts.train_multistate_chisel import (globally_heldout_next_sequences,stable_seed,
    build_proposal,evaluate_sequences,all_rank_objects,deadband)


def select_edit_component(parameters, original, editable, component):
    """Counterfactual frozen edit, never an update to Chisel or its objective.

    Ordinary biases are the editable one-dimensional tensors. Gains and scalar
    parameters remain original even if a caller supplied changed replacements.
    """
    if component not in ('matrices', 'biases'):
        raise ValueError('component must be matrices or biases')
    if set(parameters) != set(original) or not set(editable) <= set(original):
        raise ValueError('parameter names differ')
    dimension = 2 if component == 'matrices' else 1
    return {name: parameters[name] if name in editable and value.ndim == dimension else value
            for name, value in original.items()}


def checkpoint_objective(saved, state_id, kl, stats):
    """Versioned readout only; never mutate multipliers or select a policy."""
    metadata=saved['metadata'];options=metadata['options']
    policy=metadata.get('objective_policy',{})
    kind=policy.get('kind',options.get('objective','legacy-deadband'))
    if kind=='primal-dual':
        state=saved.get('primal_dual')
        if state is None:raise ValueError('primal-dual checkpoint lacks multipliers')
        dual=PrimalDual(state['multipliers'],tolerance=options['kl_tolerance'],
                       rho=options['augmented_rho'],dual_lr=options['dual_lr'])
        dual.load_state_dict(state)
        if state_id in dual.multipliers:raise ValueError('fresh evaluator requires held-out checkpoint IDs')
        terms=dual.terms(kl,state_id)
        coefficient=policy.get('shrinkage_coefficient',options.get('shrinkage_coefficient',1.))
        if not math.isfinite(coefficient) or coefficient<0:raise ValueError('invalid checkpoint shrinkage coefficient')
        unweighted=stats.get('unweighted_norm_term',stats['norm_term'])
        norm_term=coefficient*unweighted
        return dict(**terms,norm_term=norm_term,unweighted_norm_term=unweighted,shrinkage_coefficient=coefficient,objective=norm_term+terms['kl_penalty'],
                    trust_term=0.,edit_size_penalty_enabled=False,objective_kind=kind,
                    objective_readout='diagnostic augmented objective; held-out multiplier is zero',
                    objective_used_for_selection=False)
    if kind!='legacy-deadband':raise ValueError('unknown checkpoint objective version')
    # Historical 77eaccd checkpoints include the .01 edit penalty. New explicit
    # legacy-deadband runs can omit it; metadata decides, not the current code.
    trust=(.01*(stats['realized_edit_norm']/stats['hard_radius'])**2
           if policy.get('edit_size_penalty',True) else 0.)
    penalty,_=deadband(kl,options['kl_budget'])
    return dict(kl_penalty=penalty,objective=stats['norm_term']+trust+penalty,
                trust_term=trust,edit_size_penalty_enabled=policy.get('edit_size_penalty',True),objective_kind=kind,objective_used_for_selection=False)


def checkpoint_gpt_precision(saved):
    """Fresh teacher and edited GPT must use the arithmetic used for fitting."""
    metadata=saved['metadata']
    mode=metadata['options'].get('gpt_precision','baseline')
    expected=precision_metadata(mode)
    recorded=metadata.get('gpt_precision')
    if recorded is None and mode=='baseline':
        return mode  # Historical mixed-precision checkpoint before this field.
    if recorded!=expected:
        raise ValueError('checkpoint GPT precision/normalization metadata differs')
    return mode


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--state-bank',required=True,type=Path)
    parser.add_argument('--train-data',required=True)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--sequences',type=int,default=64)
    parser.add_argument('--gpt-microbatch',type=int,default=16)
    parser.add_argument('--state-ids',nargs='+',type=int,help='predeclared held-out checkpoint subset; default is every held-out state')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--eager-chisel',action='store_true')
    parser.add_argument('--separate-edit-types',action='store_true',help='also evaluate frozen matrix-only and bias-only edits; never select or train on these tests')
    args=parser.parse_args(argv)
    world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'))
    local=int(os.environ.get('LOCAL_RANK','0'));device=torch.device(f'cuda:{local}' if args.device=='cuda' else 'cpu')
    if device.type=='cuda':enable_deterministic_execution();torch.cuda.set_device(device)
    else:torch.set_num_threads(1)
    if world>1:dist.init_process_group('nccl' if device.type=='cuda' else 'gloo')
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    gpt_precision=checkpoint_gpt_precision(saved)
    metadata=saved['metadata'];architecture=metadata['architecture'];validate_checkpoint_architecture(saved,architecture)
    expected=metadata['state_bank_identity']['manifest_sha256']
    if hashlib.sha256((args.state_bank/'manifest.json').read_bytes()).hexdigest()!=expected:
        raise ValueError('evaluation checkpoint and GPT state bank differ')
    reader=StateBankReader(args.state_bank,verify_once=True,mmap_model=True)
    if metadata['state_bank_identity']['selected_states']!=reader.manifest['expected_steps']:
        raise ValueError('post-fit whole-bank test rejects prefix-only Chisel fits')
    provenance=reader.manifest['provenance'];training=provenance['training_config']
    actor=build_chisel(architecture).to(device);actor.load_state_dict(saved['controller'])
    actor_forward=actor if args.eager_chisel else torch.compile(actor,fullgraph=True,dynamic=False)
    engine=FunctionalPreservation(GPT(**provenance['model_config']).to(device).requires_grad_(False),
        microbatch_sequences=args.gpt_microbatch,teacher_cache_device=device,kl_dtype=torch.float64,
        gpt_precision=gpt_precision)
    names=set(engine.parameters);files=sorted(glob.glob(args.train_data))
    if not files:raise ValueError('no training shards')
    heldout=list(reader.manifest['split']['audit'])
    state_ids=sorted(args.state_ids) if args.state_ids else heldout
    if len(state_ids)!=len(set(state_ids)) or not set(state_ids)<=set(heldout):
        raise ValueError('test state IDs must be unique held-out GPT checkpoints')
    config=dict(checkpoint=str(args.checkpoint),checkpoint_update=saved['updates'],
        checkpoint_epoch=saved['next_epoch'],state_bank_manifest_sha256=expected,state_ids=state_ids,
        architecture=architecture,sequences_per_state=args.sequences,gpt_microbatch=args.gpt_microbatch,
        gpt_precision=precision_metadata(gpt_precision),
        data_role='fresh next ordinary batch, globally held-out Chisel sequence IDs',
        policy_selection='checkpoint path fixed before any fresh test',gradient_or_optimizer_updates=False,
        official_gpt_validation=False,world_size=world)
    config['separate_edit_types']=args.separate_edit_types
    if rank==0:
        args.output.mkdir(parents=True,exist_ok=False)
        (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n');source_record(config,args.output)
    if world>1:dist.barrier()
    rows=[];started=time.perf_counter();compute={}
    for state_id in state_ids[rank::world]:
        record=reader.read(state_id,device=device,model_device='cpu')
        engine.load_teacher_state({n:v for n,v in record.model.items() if n in names},
                                  {n:v for n,v in record.model.items() if n not in names})
        grouped=GroupedNormObjective(engine.parameters,zero_reference_policy='freeze')
        inputs,ids=globally_heldout_next_sequences(files,state_id,training,args.sequences,
            stable_seed('fresh_next_audit',metadata['options']['data_seed'],state_id),metadata['options']['split_seed'])
        # Match fitting/audit arithmetic for the Chisel graph; GPT is no-grad.
        with torch.enable_grad():
            parameters,regularizer,stats,editable,credit=build_proposal(actor_forward,record.snapshot,engine,grouped,
                radius_fraction=metadata['options']['radius_fraction'])
        with torch.no_grad():
            def teacher(start,stop):
                return engine.cache_teacher(inputs[start:stop],data_role='fresh_next_training_audit',sequence_ids=ids[start:stop])[0]
            kl,ce=evaluate_sequences(engine,parameters,inputs,teacher_provider=teacher,return_ce=True)
            objective=checkpoint_objective(saved,state_id,kl,stats)
            stats.update(objective)
            components={}
            if args.separate_edit_types:
                for component in ('matrices','biases'):
                    selected=select_edit_component(parameters,engine.parameters,editable,component)
                    value,measurement=evaluate_sequences(engine,selected,inputs,teacher_provider=teacher,return_ce=True)
                    components[component]=dict(mean_kl=value,**measurement)
            group_edits={};relative_edits={}
            for group,group_names in grouped.groups.items():
                squared=grouped.reference_squared[group].new_zeros(())
                for name in group_names:
                    squared+=(parameters[name].float()-engine.parameters[name].float()).double().square().sum()
                edit_norm=float(squared.sqrt());reference=float(grouped.reference_norms[group])
                if reference==0 and edit_norm!=0:raise AssertionError('zero reference group was edited')
                group_edits[group]=edit_norm;relative_edits[group]=edit_norm/reference if reference else 0.
            rows.append(dict(state_id=state_id,sequence_ids=ids,mean_kl=kl,group_edit_frobenius_norm=group_edits,
                group_relative_edit_norm=relative_edits,component_preservation=components,**stats,**ce))
        for key,value in engine.compute.as_dict().items():compute[key]=compute.get(key,0)+value
        if rank==0:print(json.dumps(dict(event='progress',completed_local_states=len(rows),elapsed_seconds=time.perf_counter()-started)),flush=True)
        del record,parameters,regularizer,credit,inputs,grouped
    rows=sorted([r for group in all_rank_objects(rows,world) for r in group],key=lambda r:r['state_id'])
    counts={}
    for group in all_rank_objects(compute,world):
        for key,value in group.items():counts[key]=counts.get(key,0)+value
    if counts.get('student_backwards',0):raise AssertionError('post-fit evaluation performed a backward')
    if [r['state_id'] for r in rows]!=state_ids:raise AssertionError('missing or duplicated held-out checkpoint')
    if rank==0:
        keys=('mean_kl','kl_penalty','objective','mean_group_l2_percentage_reduction','ce_change','teacher_ce','student_ce')
        summary={k:sum(r[k] for r in rows)/len(rows) for k in keys}
        for key in ('logprob_rms_shift','centered_logit_rms_shift'):
            summary[key]=(sum(r[key]**2 for r in rows)/len(rows))**.5
        summary['group_relative_edit_norm']={group:sum(r['group_relative_edit_norm'][group] for r in rows)/len(rows)
                                            for group in rows[0]['group_relative_edit_norm']}
        summary.update(state_count=len(rows),compute=counts,elapsed_seconds=time.perf_counter()-started)
        summary['objective_kind']=rows[0]['objective_kind']
        summary['objective_used_for_selection']=False
        if rows[0]['objective_kind']=='primal-dual':
            summary.update(kl_tolerance=rows[0]['kl_tolerance'],
                feasible_fraction=sum(r['feasible'] for r in rows)/len(rows),
                maximum_kl=max(r['mean_kl'] for r in rows),
                maximum_violation=max(r['constraint_violation'] for r in rows))
        (args.output/'states.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(dict(event='finished',**summary)),flush=True)
    if world>1:dist.destroy_process_group()


if __name__=='__main__':main()
