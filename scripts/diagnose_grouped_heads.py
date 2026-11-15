"""Conditional per-functional-group head LSQ; no GPT or weight-action application.

This diagnostic holds body/private features fixed at one checkpoint. It tests
representability, not learnability, preservation, or held-out generalization.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from rewriter.adaptive_coordinate_chisel import AdaptiveCoordinateChisel
from rewriter.adaptive_edit_target import edit_target_terms
from rewriter.adaptive_preservation import group_rms_scales
from rewriter.edit_scope import editable_names,editable_weight_norm_sq,scoped_snapshot
from rewriter.grouped_norm import GroupedNormObjective
from rewriter.prefix_cache import unpack_record
from scripts.fit_cpu_edit_teacher import make_target,relative_target_error,sha
from scripts.fit_cpu_teacher_heads_lstsq import matrix_design,vector_design,balanced_sse_weight,orthonormal_error,solve,state_equal
from scripts.frozen_private_frames import FrozenPrivateFrames


def summarize(groups,target_squared,relative_squared,reference_squared):
    total=sum(v['physical_error_squared'] for v in groups.values())
    relative=sum(v['physical_error_squared']/float(reference_squared[g]) for g,v in groups.items())
    global_error=total/float(target_squared);relative_error=relative/float(relative_squared)
    return dict(physical_error_squared=total,normalized_edit_error=global_error,
                relative_normalized_edit_error=relative_error,balanced_objective=.5*(global_error+relative_error))


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,default=Path('runs/cpu-edit-teacher/heads-balanced-polished-spans-v1/complete.json'))
    parser.add_argument('--export-designs',action='store_true',help='Save frozen-feature linear maps for gradient-fitting diagnostics.')
    parser.add_argument('--export-only',action='store_true',help='Export checked features without performing any least-squares fits.')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    if args.export_only and not args.export_designs:raise ValueError('export-only requires export-designs')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('highest');started=time.perf_counter()
    parent=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True);meta=parent['metadata'];training=meta['args']
    if sha(training['state'])!=meta['state_sha256'] or sha(training['selection'])!=meta['selection_sha256']:raise ValueError('parent evidence mismatch')
    state=torch.load(training['state'],map_location='cpu',weights_only=False,mmap=True)
    snapshot=unpack_record(state['observation'],device='cpu').snapshot
    names=set(snapshot.matrices)|set(snapshot.vectors);parameters={n:v.detach().float() for n,v in state['model'].items() if n in names}
    grouped=GroupedNormObjective(parameters,zero_reference_policy='freeze');editable=editable_names(parameters,'grouped')-grouped.frozen_parameter_names
    snapshot=scoped_snapshot(snapshot,editable_weight_norm_sq(parameters,editable),meta['radius_fraction'],scope='grouped')
    target,selected=make_target(parameters,snapshot,json.loads(Path(training['selection']).read_text())['rows'],editable)
    target_squared=target.norm_sq();relative_squared=relative_target_error(edit_target_terms(target.scaled(0.),target),grouped)
    actor=AdaptiveCoordinateChisel(rank=8,trajectory_dim=400,model_dim=256,num_layers=2,latent_tokens=8,fresh_weight_core=True,factor_precision='fp64',residual_precision='fp32')
    if actor.configuration()!=meta['architecture']:raise ValueError('architecture mismatch')
    private=actor.make_private(snapshot,state_key=('cpu-teaching',0,state['completed_steps'],meta['state_sha256']),seed=training['private_seed'])
    actor.load_state_dict(parent['actor']);private.load_state_dict(parent['private']);actor.eval();actor.requires_grad_(False);private.requires_grad_(False)
    hidden={}
    def capture(key):
        def hook(module,inputs):hidden[key]=inputs[0].detach().clone()
        return hook
    hooks=[actor.matrix_output.register_forward_pre_hook(capture('matrix')),actor.vector_output.register_forward_pre_hook(capture('vector'))]
    phase=FrozenPrivateFrames(private);cores=phase.query_weight_cores(parameters);raw=phase.evaluate(actor,snapshot,weight_cores=cores)
    for hook in hooks:hook.remove()
    scales=group_rms_scales(grouped,dtype=torch.float64);r=actor.latent_tokens
    mh=hidden['matrix'].reshape(len(snapshot.matrices),2,r,-1);vh=hidden['vector'].reshape(len(snapshot.vectors),r,-1)
    parameter_groups={n:g for g,ns in grouped.groups.items() for n in ns}
    groups={g:dict(designs=[],targets=[],names=[],target_squared=0.,span_floor_squared=0.,kind='matrix' if g.endswith('.weight') else 'vector') for g in grouped.groups}
    current_matrix=torch.cat((actor.matrix_output.weight,actor.matrix_output.bias[:,None]),dim=1).reshape(-1)
    current_vector=torch.cat((actor.vector_output.weight,actor.vector_output.bias[:,None]),dim=1).reshape(-1)
    for kind,observations in [('matrix',snapshot.matrices),('vector',snapshot.vectors)]:
        for i,name in enumerate(observations):
            if name not in editable:continue
            if kind=='matrix':
                u=phase.view.matrix_rows[i].orthonormal_matrix();v=phase.view.matrix_columns[i].orthonormal_matrix()
                orthonormal_error(u);orthonormal_error(v)
                factor=target.matrices[name];wanted=((u.T@factor.U)@factor.C@(factor.V.T@v)).flatten()
                design=matrix_design(mh[i,0],mh[i,1],scales[name]*math.sqrt(u.shape[0]*v.shape[0]/(r*r))/math.sqrt(2))
                torch.testing.assert_close(design@current_matrix,(raw.matrices[name].C*scales[name]).flatten(),rtol=1e-10,atol=1e-11)
                norm=float(factor.norm_sq())
            else:
                q=phase.view.vector_frames[i].orthonormal_matrix();orthonormal_error(q)
                design=vector_design(vh[i],scales[name]*math.sqrt(q.shape[0]/r));wanted=q.T@target.vectors[name]
                torch.testing.assert_close(q@(design@current_vector),raw.vectors[name]*scales[name],rtol=1e-10,atol=1e-11)
                norm=float(target.vectors[name].square().sum())
            floor=norm-float(wanted.square().sum())
            if floor < -1e-10*max(1.,norm):raise ValueError('projected target exceeds target energy')
            group=groups[parameter_groups[name]];group['designs'].append(design);group['targets'].append(wanted);group['names'].append(name)
            group['target_squared']+=norm;group['span_floor_squared']+=max(0.,floor)
    if {n for g in groups.values() for n in g['names']}!=editable:raise ValueError('design omits editable outputs')
    for group in groups.values():group['A']=torch.cat(group.pop('designs'));group['b']=torch.cat(group.pop('targets'))
    design_sources=[str(args.checkpoint),training['state'],training['selection'],__file__,'scripts/fit_cpu_teacher_heads_lstsq.py','scripts/fit_cpu_edit_teacher.py','rewriter/grouped_norm.py']
    if args.export_designs:
        design_manifest={str(f):sha(f) for f in design_sources}
        torch.save(dict(groups={n:dict(A=g['A'],b=g['b'],weight=balanced_sse_weight(target_squared,relative_squared,grouped.reference_squared[n])) for n,g in groups.items()},source_sha256=design_manifest,checkpoint=str(args.checkpoint),objective='half global plus half group-relative projected edit error; fixed probe floor omitted'),args.output/'designs.pt')
    if args.export_only:
        for module,original in ((actor,parent['actor']),(private,parent['private'])):
            if not state_equal(module.state_dict(),original):raise ValueError('export mutated actor/private')
        phase.validate();phase.close()
        export=dict(scope='Frozen shared-transformer features for gradient head fitting; no least-squares fits or GPT evaluations',source_sha256=design_manifest,actual_actor_design_parity=True,actor_private_unchanged=True,gpt_forwards=0,gpt_backwards=0,elapsed_seconds=time.perf_counter()-started)
        (args.output/'complete.json').write_text(json.dumps(export,indent=2)+'\n')
        print(json.dumps(export));return
    shared_coeff={};shared_solver={}
    for kind in ('matrix','vector'):
        rows=[];targets=[]
        for name,g in groups.items():
            if g['kind']!=kind:continue
            weight=balanced_sse_weight(target_squared,relative_squared,grouped.reference_squared[name]).sqrt()
            rows.append(g['A']*weight);targets.append(g['b']*weight)
        shared_coeff[kind],shared_solver[kind]=solve(torch.cat(rows),torch.cat(targets))
    shared={};separate={};predicted_norm_squared=0.;coefficients={}
    for name,g in groups.items():
        A,b=g['A'],g['b'];coef,report=solve(A,b);coefficients[name]=coef
        prediction=A@coef;predicted_norm_squared+=float(prediction.square().sum())
        common={k:v for k,v in g.items() if k not in ('A','b')}
        shared_residual=float((A@shared_coeff[g['kind']]-b).square().sum())
        shared[name]=dict(common,core_residual_squared=shared_residual,physical_error_squared=shared_residual+g['span_floor_squared'])
        separate[name]=dict(common,solver=report,design_rows=A.shape[0],head_parameters=A.shape[1],nullity=A.shape[1]-report['rank'],underdetermined=A.shape[0]<A.shape[1],core_residual_squared=report['residual_squared'],physical_error_squared=report['residual_squared']+g['span_floor_squared'])
    baseline=summarize(shared,target_squared,relative_squared,grouped.reference_squared)
    expected=json.loads(args.baseline.read_text())['final']
    for key in ('normalized_edit_error','relative_normalized_edit_error'):
        if not math.isclose(baseline[key],expected[key],rel_tol=2e-7,abs_tol=1e-9):raise ValueError('recomputed baseline differs: '+key)
    final=summarize(separate,target_squared,relative_squared,grouped.reference_squared)
    if predicted_norm_squared>float(snapshot.radius)**2:raise ValueError('global radius guard would activate')
    for label,module,original in [('actor',actor,parent['actor']),('private',private,parent['private'])]:
        if not state_equal(module.state_dict(),original):raise ValueError(label+' changed')
    phase.validate();phase.close()
    shared_count=sum(v.numel() for v in shared_coeff.values());separate_count=sum(v.numel() for v in coefficients.values())
    sources=[str(args.checkpoint),str(args.baseline),training['state'],training['selection'],__file__,'scripts/fit_cpu_teacher_heads_lstsq.py','scripts/fit_cpu_edit_teacher.py','rewriter/grouped_norm.py']
    result=dict(scope=__doc__,checkpoint=str(args.checkpoint),completed_steps=state['completed_steps'],baseline=baseline,grouped=final,shared_parameter_count=shared_count,grouped_parameter_count=separate_count,parameter_count_delta=separate_count-shared_count,parameter_count_ratio=separate_count/shared_count,shared_solver=shared_solver,shared_groups=shared,groups=separate,projected_action_norm_squared=predicted_norm_squared,radius_squared=float(snapshot.radius)**2,guard_would_activate=False,original_actor_private_unchanged=True,actual_actor_design_parity=True,gpt_forwards=0,gpt_backwards=0,optimizer_updates=0,weight_actions_applied=0,source_sha256={str(f):sha(f) for f in sources},elapsed_seconds=time.perf_counter()-started)
    torch.save(dict(grouped_coefficients=coefficients,shared_coefficients=shared_coeff,source_sha256=result['source_sha256']),args.output/'coefficients.pt')
    (args.output/'complete.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('baseline','grouped','shared_parameter_count','grouped_parameter_count','elapsed_seconds')}))

if __name__=='__main__':main()
