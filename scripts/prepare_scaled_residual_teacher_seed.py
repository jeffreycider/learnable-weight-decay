"""Known-target diagnostic seed: scaled-random body, shared analytic LS heads.

Private frames AND trajectory_input are reused from target-fitted states6/7.
This is not random initialization, independent pretraining, norm/KL learning or
preservation evidence. No GPT execution, optimizer updates or run queue changes.
"""
import argparse
import gc
import json
import math
from pathlib import Path
import resource
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scripts.diagnose_residual_head_conditioning import extract
from scripts.prepare_random_grouped_cpu_seeds import fresh_actor,residual_output_spec
from scripts.fit_cpu_grouped_heads_gradient import merge_designs
from scripts.fit_cpu_teacher_heads_lstsq import solve
from scripts.fit_cpu_edit_teacher import sha,save,make_target,relative_target_error,error_metrics,stored_readout,exact_state_equal
from scripts.evaluate_cpu_fitted_chisel import load_action
from rewriter.adaptive_edit_target import edit_target_terms
from rewriter.prefix_cache import unpack_record
from rewriter.edit_scope import editable_names,editable_weight_norm_sq,scoped_snapshot


def head_norms(actor_state):
    rows={}
    for n,v in actor_state.items():
        if n.startswith(('matrix_output.','vector_output.')) and n.endswith('.weight'):
            bias=actor_state[n[:-6]+'bias'];rows[n.rsplit('.',1)[0]]=dict(weight_l2=float(v.norm()),bias_l2=float(bias.norm()),combined_l2=float(torch.cat((v.flatten(),bias.flatten())).norm()))
    return dict(heads=rows,total_l2=math.sqrt(sum(r['combined_l2']**2 for r in rows.values())))


def norm_summary(readout):
    result={}
    for label,row in readout['stored_norm_metrics'].items():
        groups=row['group_l2_percentage_reduction']
        result[label]=dict(mean_group_shrink_percent=row['mean_group_l2_percentage_reduction'],
            contributions_to_mean_percent={role:sum(v for g,v in groups.items() if
                ('matrices' if g.endswith('.weight') else 'head_bias' if g=='head.bias' else 'other_biases')==role)/len(groups)
                for role in ('matrices','head_bias','other_biases')},group_shrink_percent=groups)
    return result


@torch.no_grad()
def verify_seed(checkpoint_path,old_path,design_record):
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False,mmap=True);meta=checkpoint['metadata'];source=meta['args']
    state=torch.load(source['state'],map_location='cpu',weights_only=False,mmap=True)
    if sha(source['state'])!=meta['state_sha256'] or sha(source['selection'])!=meta['selection_sha256']:raise ValueError('original target provenance mismatch')
    params,grouped,action,execution=load_action(state,checkpoint,meta['state_sha256'])
    if execution['guard']!=1.:raise ValueError('unexpected radius guard; no alternate target allowed')
    snapshot=unpack_record(state['observation'],device='cpu').snapshot
    editable=editable_names(params,'grouped')-grouped.frozen_parameter_names
    snapshot=scoped_snapshot(snapshot,editable_weight_norm_sq(params,editable),meta['radius_fraction'],scope='grouped')
    target,_=make_target(params,snapshot,json.loads(Path(source['selection']).read_text())['rows'],editable)
    target_squared=target.norm_sq();relative_squared=relative_target_error(edit_target_terms(target.scaled(0.),target),grouped)
    metrics=error_metrics(edit_target_terms(action,target),target_squared,relative_squared,grouped)
    actual=stored_readout(action,target,params,grouped)
    # Comparison uses the actual old saved actor/private, not merely its old log.
    old=torch.load(old_path,map_location='cpu',weights_only=False,mmap=True)
    if old['metadata']['state_sha256']!=meta['state_sha256'] or old['metadata']['selection_sha256']!=meta['selection_sha256']:
        raise ValueError('old seed target differs')
    if not exact_state_equal(old['private'],checkpoint['private']):raise ValueError('old/new private modules differ')
    # Release first loader's dense FP32 source before reconstructing old source.
    del params,grouped,action;gc.collect()
    old_params,old_grouped,old_action,old_execution=load_action(state,old,meta['state_sha256'])
    if old_execution['guard']!=1.:raise ValueError('old reference guard changed')
    old_metrics=error_metrics(edit_target_terms(old_action,target),target_squared,relative_squared,old_grouped)
    prior=stored_readout(old_action,target,old_params,old_grouped)
    record=dict(state_id=state['completed_steps'],checkpoint=str(checkpoint_path),checkpoint_sha256=sha(checkpoint_path),
        old_checkpoint=str(old_path),old_checkpoint_sha256=sha(old_path),same_private_exact=True,new_execution=execution,old_execution=old_execution,
        new_physical_target_error=metrics,old_physical_target_error=old_metrics,new_head_norms=head_norms(checkpoint['actor']),old_head_norms=head_norms(old['actor']),
        new_stored=actual,old_stored=prior,new_norm_summary=norm_summary(actual),old_norm_summary=norm_summary(prior),
        target_provenance=design_record,gpt_forwards=0,gpt_backwards=0)
    (checkpoint_path.parent/'verification.json').write_text(json.dumps(record,indent=2)+'\n')
    return {k:v for k,v in record.items() if k not in ('new_stored','old_stored')},dict(
        new_stored_target_error_squared=actual['stored_weight_error_squared'],new_stored_target_different_elements=actual['stored_different_elements'],
        old_stored_target_error_squared=prior['stored_weight_error_squared'],old_stored_target_different_elements=prior['stored_different_elements'])


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('highest')
    started=time.perf_counter();actor,_=fresh_actor(940067,residual_output_init='scaled_random',residual_seed=960067);actor.eval().requires_grad_(False)
    body={n:v.clone() for n,v in actor.state_dict().items() if torch.is_tensor(v) and not n.startswith(('matrix_output.','vector_output.'))}
    datasets=[];provenance=[]
    for uid in (6,7):
        # Existing extractor's dictionary key names arms only; this single arm
        # is explicitly scaled_random, regardless of its historical 'zero' key.
        data,record=extract(Path(f'runs/grouped-head-diagnostic/state{uid}-export-only-v1/designs.pt'),{'zero':actor})
        datasets.append(data['zero']);provenance.append(record);gc.collect()
    groups=merge_designs(datasets);del datasets
    solvers={};coefficients={}
    for group,g in groups.items():
        coeff,report=solve(g['A'],g['b']);head=actor.output_head(group);joined=coeff.reshape(head.out_features,head.in_features+1)
        head.weight.copy_(joined[:,:-1]);head.bias.copy_(joined[:,-1]);coefficients[group]=coeff
        report.pop('singular_values');solvers[group]=report
    del groups;gc.collect()
    if any(not torch.equal(v,actor.state_dict()[n]) for n,v in body.items()):raise ValueError('shared body changed during head solve')
    save(a.output/'coefficients.pt',dict(grouped_coefficients=coefficients,source_records=provenance))
    all_sources={str(Path(__file__)):sha(__file__),**{str(p):sha(p) for p in (
        'scripts/diagnose_residual_head_conditioning.py','scripts/prepare_random_grouped_cpu_seeds.py',
        'scripts/fit_cpu_teacher_heads_lstsq.py','scripts/fit_cpu_grouped_heads_gradient.py',
        'scripts/evaluate_cpu_fitted_chisel.py','scripts/fit_cpu_edit_teacher.py')}}
    rows=[];cases=[];shared_reference=None
    for uid,record in zip((6,7),provenance):
        parent=torch.load(record['private_checkpoint'],map_location='cpu',weights_only=False,mmap=True);prior=parent['metadata']
        out=a.output/f'state{uid}';out.mkdir()
        meta=dict(schema='cpu_edit_teacher_fit_v1',scope=__doc__,architecture=actor.configuration(),
            args=dict(state=record['state'],selection=record['selection'],private_seed=prior['args']['private_seed'],output=str(out),objective='balanced'),
            state_sha256=record['state_sha256'],selection_sha256=record['selection_sha256'],private_identity=parent['private']['_extra_state'],
            radius=prior['radius'],radius_fraction=prior['radius_fraction'],diagnostic_control='functional_group_head_lstsq',
            initialization=dict(shared_seed=940067,residual_outputs=residual_output_spec(actor,'scaled_random',960067),
                shared_body='fresh scaled-random residual body; other internals from fresh shared seed',
                heads='analytic shared balanced least-squares on known state6/7 edit targets',
                private='unchanged target-fitted frames AND trajectory_input; not random/independent',
                private_parent=record['private_checkpoint'],private_parent_sha256=record['private_checkpoint_sha256'],private_parent_control_step=parent['step']),
            shared_head_training_cases=[6,7],source_sha256=all_sources,optimizer_state='none; incompatible with naive training resume',
            finite_preservation_evidence='none for this seed; actual stored weight verification only',gpt_forwards=0,gpt_backwards=0)
        path=out/'checkpoint_control.pt'
        save(path,dict(metadata=meta,step=0,control_updates=0,actor=actor.state_dict(),private=parent['private'],
            diagnostic_control=True,optimizer=None,private_optimizer=None,rng=torch.get_rng_state()))
        loaded=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        if not exact_state_equal(loaded['private'],parent['private']):raise ValueError('private changed while saving')
        if shared_reference is None:shared_reference=loaded['actor']
        elif not exact_state_equal(shared_reference,loaded['actor']):raise ValueError('state seeds do not share identical actor')
        old_path=Path(f'runs/cpu-edit-teacher/shared-states6-7-actor{uid}-v1/checkpoint_control.pt')
        row,stored_summary=verify_seed(path,old_path,record);row.update(stored_summary);rows.append(row)
        panel=Path(f'data/chisel-teaching-diagnostic/teacher_state{uid}_train.pt');panelmeta=panel.with_suffix('.json')
        if json.loads(panelmeta.read_text())['state_sha256']!=meta['state_sha256']:raise ValueError('existing TRAIN panel source differs')
        cases.append(dict(state=str(Path(record['state']).resolve()),checkpoint=str(path.resolve()),panel=str(panel.resolve()),panel_metadata=str(panelmeta.resolve())))
        print(json.dumps(dict(state=uid,stored=stored_summary,new_norm=row['new_norm_summary']['predicted'],seconds=time.perf_counter()-started)),flush=True)
        del loaded,parent;gc.collect()
    (a.output/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    result=dict(scope=__doc__,cases=rows,solvers=solvers,shared_actor_exact_between_states=True,shared_body_unchanged_during_solve=True,
        heads=head_norms(actor.state_dict()),projected_balanced_sse=sum(r['residual_squared'] for r in solvers.values()),
        coefficients_sha256=sha(a.output/'coefficients.pt'),source_sha256=all_sources,elapsed_seconds=time.perf_counter()-started,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,gpt_forwards=0,gpt_backwards=0,optimizer_updates=0)
    (a.output/'complete.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
