"""Prepare a new-state known-target diagnostic, frozen shared actor and fresh private frames.

One existing TRAIN sequence is actually evaluated on CPU. This does not qualify
heldout preservation or generalization, and changes no main learning objective.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baseline.model import GPT
from rewriter.adaptive_coordinate_chisel import AdaptiveCoordinateChisel
from rewriter.prefix_cache import unpack_record
from rewriter.grouped_norm import GroupedNormObjective
from rewriter.edit_scope import editable_names,editable_weight_norm_sq,scoped_snapshot
from rewriter.gpt_precision import configure_precision,fp32_head_inputs
from rewriter.adaptive_edit_target import edit_target_terms
from scripts.fit_cpu_edit_teacher import make_target,relative_target_error,stored_readout,sha,save,exact_state_equal
from scripts.evaluate_cpu_fitted_chisel import stored_edits,logit_metrics


def transplant_pattern(selection,state_id,state_sha):
    """Copy only structural channel choices; never copy old norm/finite claims."""
    rows=[]
    for row in selection['rows']:
        rows.append(dict(rank_limit=row['rank_limit'],inverse_power=row['inverse_power'],
            selected=copy.deepcopy(row['selected']),state_id=state_id,state_sha256=state_sha,
            finite_logits_qualified=False,target_pattern_origin='state6 fixed channel pattern; state-specific target recomputed'))
    return dict(rows=rows,scope='structural pattern transplant only; no inherited norm or finite-logit claims')


def validate_train_panel(panel,metadata,state_sha,panel_sha):
    if metadata['sha256']!=panel_sha or metadata['state_sha256']!=state_sha:
        raise ValueError('panel/state hash mismatch')
    if panel['metadata']['role']!='train' or panel['metadata']['state_sha256']!=state_sha or panel['metadata']['sequence_ids']!=metadata['sequence_ids']:
        raise ValueError('matching existing TRAIN panel required')


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    for name in ('state','parent','pattern','panel','panel-metadata','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--private-seed',type=int,default=920007);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);configure_precision('fp32');started=time.perf_counter()
    state_sha=sha(a.state);parent=torch.load(a.parent,map_location='cpu',weights_only=False,mmap=True);state=torch.load(a.state,map_location='cpu',weights_only=False,mmap=True)
    panel=torch.load(a.panel,map_location='cpu',weights_only=False,mmap=True);panel_meta=json.loads(a.panel_metadata.read_text())
    validate_train_panel(panel,panel_meta,state_sha,sha(a.panel))
    selection=transplant_pattern(json.loads(a.pattern.read_text()),state['completed_steps'],state_sha)
    selection.update(parent_pattern=str(a.pattern),parent_pattern_sha256=sha(a.pattern));selection_path=a.output/'selection.json';selection_path.write_text(json.dumps(selection,indent=2)+'\n')
    snapshot=unpack_record(state['observation'],device='cpu').snapshot
    names=set(snapshot.matrices)|set(snapshot.vectors);parameters={n:v.detach().float() for n,v in state['model'].items() if n in names}
    grouped=GroupedNormObjective(parameters,zero_reference_policy='freeze');editable=editable_names(parameters,'grouped')-grouped.frozen_parameter_names
    snapshot=scoped_snapshot(snapshot,editable_weight_norm_sq(parameters,editable),.01,scope='grouped')
    target,_=make_target(parameters,snapshot,selection['rows'],editable);target_squared=float(target.norm_sq())
    if not 0<target_squared<=float(snapshot.radius)**2:raise ValueError('target exceeds fixed radius')
    relative=float(relative_target_error(edit_target_terms(target.scaled(0.),target),grouped))
    actor=AdaptiveCoordinateChisel(rank=8,trajectory_dim=400,model_dim=256,num_layers=2,latent_tokens=8,fresh_weight_core=True,factor_precision='fp64',residual_precision='fp32')
    if actor.configuration()!=parent['metadata']['architecture']:raise ValueError('shared architecture mismatch')
    actor.load_state_dict(parent['actor'],strict=True)
    private=actor.make_private(snapshot,state_key=('cpu-teaching',0,state['completed_steps'],state_sha),seed=a.private_seed)
    if not exact_state_equal(actor.state_dict(),parent['actor']):raise ValueError('shared actor changed')
    if private.get_extra_state()==parent['metadata']['private_identity']:raise ValueError('private identity not fresh')
    edited=stored_edits(parameters,target);base=state['model'];depth=len({n.split('.')[1] for n in base if n.startswith('blocks.')})
    model=GPT(base['embed.weight'].shape[0],depth,base['embed.weight'].shape[1]).float().eval().requires_grad_(False);model.load_state_dict(base,strict=True)
    sequence=panel['inputs'][:1];z=model.proj(fp32_head_inputs(model,sequence)).float();before=15*z*(z.square()+225).rsqrt()
    live=dict(model.named_parameters())
    for n,v in edited.items():live[n].copy_(v)
    z=model.proj(fp32_head_inputs(model,sequence)).float();after=15*z*(z.square()+225).rsqrt()
    measured=logit_metrics(before,after)
    evidence=dict(scope='actual CPU FP32 target-action forward on one existing TRAIN sequence; not copied GPU logits or a heldout qualification',state_sha256=state_sha,selection_sha256=sha(selection_path),panel_sha256=sha(a.panel),sequence_id=panel_meta['sequence_ids'][0],metrics=measured,gpt_forwards=2,gpt_backwards=0,query_access=False,teacher_preservation_pass=measured['different_logits']==0,elapsed_seconds=time.perf_counter()-started)
    evidence_path=a.output/'finite_cpu_train1.json';evidence_path.write_text(json.dumps(evidence,indent=2)+'\n')
    (a.output/'stored_readout.json').write_text(json.dumps(stored_readout(target,target,parameters,grouped),indent=2)+'\n')
    if not evidence['teacher_preservation_pass']:raise ValueError('observed teacher target does not exactly preserve this CPU TRAIN sequence; evidence saved')
    meta=copy.deepcopy(parent['metadata'])
    for key in ('projection_continuation','analytic_seed','resumed_from'):meta.pop(key,None)
    # Ancestor training settings are provenance, not this preparation's optimizer.
    meta['ancestor_training_settings']={k:meta.pop(k) for k in ('active_optimizer','optimizer_note','clip','shared_epsilon','private_epsilon','betas','control_source_sha256') if k in meta}
    meta.update(scope=__doc__,active_optimizer=None,optimizer_note='No optimizer constructed in seed preparation; future projection fit creates fresh moments',gpt_forwards=2,gpt_backwards=0,control_source_sha256=sha(__file__))
    meta.update(diagnostic_control='new_state_fresh_private_teacher_seed',state_sha256=state_sha,selection_sha256=sha(selection_path),finite_evidence_sha256=sha(evidence_path),private_identity=private.get_extra_state(),target_norm_squared=target_squared,relative_target_squared=relative,group_reference_squared={g:float(v) for g,v in grouped.reference_squared.items()},radius=float(snapshot.radius),radius_fraction=.01,trainable='future projection control only; shared actor copied exactly; new independently seeded private frames; no optimizer moments',parent_checkpoint=str(a.parent),parent_sha256=sha(a.parent),source_sha256={str(Path(__file__).relative_to(Path.cwd())):sha(__file__)},shared_actor_origin=dict(path=str(a.parent),sha256=sha(a.parent),learned_updates=parent['step']),private_origin=dict(initialization='fresh random',seed=a.private_seed,state_id=state['completed_steps'],inherited_optimizer_moments=False),target_provenance=dict(pattern=str(a.pattern),sha256=sha(a.pattern),weights_state=str(a.state)))
    meta['args'].update(state=str(a.state),selection=str(selection_path),finite_evidence=str(evidence_path),private_seed=a.private_seed,output=str(a.output),resume=None)
    save(a.output/'checkpoint_seed.pt',dict(metadata=meta,step=parent['step'],control_updates=0,actor=actor.state_dict(),private=private.state_dict(),diagnostic_control=True,optimizer=None,private_optimizer=None,rng=torch.get_rng_state()))
    (a.output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');(a.output/'source.py').write_text(Path(__file__).read_text())
    print(json.dumps(evidence),flush=True)


if __name__=='__main__':main()
