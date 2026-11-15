"""Analytic conditional shared-head control, not a training/preservation algorithm."""
import argparse
import copy
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
from rewriter.adaptive_sensing import query_weight_cores
from rewriter.edit_scope import editable_names,editable_weight_norm_sq,scoped_snapshot
from rewriter.grouped_norm import GroupedNormObjective
from rewriter.prefix_cache import unpack_record
from scripts.fit_cpu_edit_teacher import make_target,physical_action,relative_target_error,error_metrics,stored_readout,write_span_capture,sha,save


def matrix_design(row_hidden,column_hidden,scale):
    """Map flattened [head.weight | head.bias] to physical core entries."""
    if row_hidden.shape!=column_hidden.shape:raise ValueError('paired hidden shape mismatch')
    r,d=row_hidden.shape
    row=torch.cat((row_hidden,row_hidden.new_ones(r,1)),dim=1)
    col=torch.cat((column_hidden,column_hidden.new_ones(r,1)),dim=1)
    design=row_hidden.new_zeros(r,r,r,d+1)
    for i in range(r):
        for j in range(r):
            design[i,j,j]+=row[i]*scale
            design[i,j,i]+=col[j]*scale
    return design.reshape(r*r,r*(d+1))


def vector_design(hidden,scale):
    return torch.cat((hidden,hidden.new_ones(len(hidden),1)),dim=1)*scale


def balanced_sse_weight(target_squared,relative_squared,group_reference_squared):
    """Physical SSE coefficient in the mean of two identity-normalized losses."""
    values=(target_squared,relative_squared,group_reference_squared)
    if any(not math.isfinite(float(v)) or float(v)<=0 for v in values):
        raise ValueError('positive finite target and group scales required')
    return .5/target_squared+.5/(relative_squared*group_reference_squared)


def orthonormal_error(q):
    if q.shape[0]<q.shape[1]:raise ValueError('control requires column-orthonormal frames')
    error=float((q.T@q-torch.eye(q.shape[1],dtype=q.dtype)).abs().max())
    if error>1e-10:raise ValueError('frame columns not orthonormal')
    return error


def state_equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(state_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(state_equal(x,y) for x,y in zip(a,b))
    return a==b


def solve(design,target):
    started=time.perf_counter()
    rcond=torch.finfo(design.dtype).eps*max(design.shape)
    result=torch.linalg.lstsq(design,target[:,None],rcond=rcond,driver='gelsd')
    coefficients=result.solution[:,0];prediction=design@coefficients
    singular=result.singular_values
    report=dict(shape=list(design.shape),driver='gelsd',rcond=rcond,rank=int(result.rank),
        singular_values=singular.tolist(),largest_singular=float(singular[0]),smallest_singular=float(singular[-1]),
        full_condition_number=float(singular[0]/singular[-1]) if singular[-1]>0 else None,
        retained_condition_number=float(singular[0]/singular[int(result.rank)-1]) if result.rank>0 else None,
        residual_squared=float((prediction-target).square().sum()),target_squared=float(target.square().sum()),
        prediction_squared=float(prediction.square().sum()),elapsed_seconds=time.perf_counter()-started)
    if report['prediction_squared']>report['target_squared']*(1+1e-8)+1e-12:raise ValueError('LSQ projection norm inequality failed')
    if not torch.isfinite(coefficients).all():raise ValueError('nonfinite least-squares coefficients')
    return coefficients,report


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--objective',choices=['global','balanced'],default='global');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');started=time.perf_counter()
    parent=torch.load(a.checkpoint,map_location='cpu',weights_only=False,mmap=True);meta=parent['metadata'];args=meta['args']
    if sha(args['state'])!=meta['state_sha256'] or sha(args['selection'])!=meta['selection_sha256']:raise ValueError('parent evidence mismatch')
    state=torch.load(args['state'],map_location='cpu',weights_only=False,mmap=True)
    snapshot=unpack_record(state['observation'],device='cpu').snapshot
    names=set(snapshot.matrices)|set(snapshot.vectors);parameters={n:v.detach().float() for n,v in state['model'].items() if n in names}
    grouped=GroupedNormObjective(parameters,zero_reference_policy='freeze');editable=editable_names(parameters,'grouped')-grouped.frozen_parameter_names
    snapshot=scoped_snapshot(snapshot,editable_weight_norm_sq(parameters,editable),meta['radius_fraction'],scope='grouped')
    target,selected_names=make_target(parameters,snapshot,json.loads(Path(args['selection']).read_text())['rows'],editable)
    target_squared=target.norm_sq();relative_squared=relative_target_error(edit_target_terms(target.scaled(0.),target),grouped)
    actor=AdaptiveCoordinateChisel(rank=8,trajectory_dim=400,model_dim=256,num_layers=2,latent_tokens=8,fresh_weight_core=True,factor_precision='fp64',residual_precision='fp32')
    if actor.configuration()!=meta['architecture']:raise ValueError('architecture mismatch')
    private=actor.make_private(snapshot,state_key=('cpu-teaching',0,state['completed_steps'],meta['state_sha256']),seed=args['private_seed'])
    actor.load_state_dict(parent['actor']);private.load_state_dict(parent['private']);actor.eval()
    for module in (actor,private):module.requires_grad_(False)
    hidden={}
    def capture(key):
        def hook(module,inputs):hidden[key]=inputs[0].detach().clone()
        return hook
    hooks=[actor.matrix_output.register_forward_pre_hook(capture('matrix')),actor.vector_output.register_forward_pre_hook(capture('vector'))]
    cores=query_weight_cores(parameters,private);raw=actor(snapshot,private,weight_cores=cores)
    for hook in hooks:hook.remove()
    predicted,norm,guard=physical_action(raw,grouped,snapshot)
    initial=error_metrics(edit_target_terms(predicted,target),target_squared,relative_squared,grouped)
    scales=group_rms_scales(grouped,dtype=torch.float64);r=actor.latent_tokens
    mh=hidden['matrix'].reshape(len(snapshot.matrices),2,r,-1);vh=hidden['vector'].reshape(len(snapshot.vectors),r,-1)
    matrix_rows=[];matrix_targets=[];vector_rows=[];vector_targets=[];inventory=[];max_qerror=0.
    matrix_weights=[];vector_weights=[]
    parameter_groups={n:g for g,names in grouped.groups.items() for n in names}
    def fit_weight(name):
        return (balanced_sse_weight(target_squared,relative_squared,grouped.reference_squared[parameter_groups[name]])
                if a.objective=='balanced' else target_squared.new_ones(()))
    for i,name in enumerate(snapshot.matrices):
        if name not in editable:continue
        u=private.matrix_rows[i].orthonormal_matrix();v=private.matrix_columns[i].orthonormal_matrix()
        max_qerror=max(max_qerror,orthonormal_error(u),orthonormal_error(v))
        factor=target.matrices[name];wanted=(u.T@factor.U)@factor.C@(factor.V.T@v)
        scale=scales[name]*math.sqrt(u.shape[0]*v.shape[0]/(r*r))/math.sqrt(2)
        design=matrix_design(mh[i,0],mh[i,1],scale)
        current=torch.cat((actor.matrix_output.weight,actor.matrix_output.bias[:,None]),dim=1).reshape(-1)
        torch.testing.assert_close((design@current).reshape(r,r),raw.matrices[name].C*scales[name],rtol=1e-10,atol=1e-11)
        matrix_rows.append(design);matrix_targets.append(wanted.flatten());inventory.append(name)
        matrix_weights.append(fit_weight(name).expand(r*r))
    for i,name in enumerate(snapshot.vectors):
        if name not in editable:continue
        q=private.vector_frames[i].orthonormal_matrix();max_qerror=max(max_qerror,orthonormal_error(q))
        design=vector_design(vh[i],scales[name]*math.sqrt(q.shape[0]/r));wanted=q.T@target.vectors[name]
        current=torch.cat((actor.vector_output.weight,actor.vector_output.bias[:,None]),dim=1).reshape(-1)
        torch.testing.assert_close(q@(design@current),raw.vectors[name]*scales[name],rtol=1e-10,atol=1e-11)
        vector_rows.append(design);vector_targets.append(wanted);inventory.append(name)
        vector_weights.append(fit_weight(name).expand(len(wanted)))
    if set(inventory)!=editable:raise ValueError('design omits editable outputs')
    A=torch.cat(matrix_rows);b=torch.cat(matrix_targets);V=torch.cat(vector_rows);vb=torch.cat(vector_targets)
    print(json.dumps(dict(event='solver_start',matrix_shape=list(A.shape),vector_shape=list(V.shape))),flush=True)
    mw=torch.cat(matrix_weights).sqrt();vw=torch.cat(vector_weights).sqrt()
    matrix_coeff,matrix_report=solve(A*mw[:,None],b*mw)
    vector_coeff,vector_report=solve(V*vw[:,None],vb*vw)
    matrix_coeff=matrix_coeff.reshape(r,-1);actor.matrix_output.weight.copy_(matrix_coeff[:,:-1]);actor.matrix_output.bias.copy_(matrix_coeff[:,-1])
    actor.vector_output.weight.copy_(vector_coeff[:-1][None]);actor.vector_output.bias.copy_(vector_coeff[-1:])
    after_raw=actor(snapshot,private,weight_cores=query_weight_cores(parameters,private));action,norm,guard=physical_action(after_raw,grouped,snapshot)
    if float(guard)!=1.:raise ValueError('guard must be inactive for linear head fit')
    if a.objective=='global' and float(action.norm_sq())>float(target_squared)*(1+1e-8):
        raise ValueError('global projection norm qualification failed')
    allcores=torch.cat([(after_raw.matrices[n].C*scales[n]).flatten() for n in snapshot.matrices if n in editable])
    torch.testing.assert_close(allcores,A@matrix_coeff.flatten(),rtol=1e-9,atol=1e-10)
    vector_projection=torch.cat([private.vector_frames[i].orthonormal_matrix().T@(after_raw.vectors[n]*scales[n]) for i,n in enumerate(snapshot.vectors) if n in editable])
    torch.testing.assert_close(vector_projection,V@vector_coeff,rtol=1e-9,atol=1e-10)
    for name,value in actor.state_dict().items():
        if not name.startswith(('matrix_output.','vector_output.')) and not state_equal(value,parent['actor'][name]):raise ValueError('frozen actor state changed: '+name)
    for name,value in private.state_dict().items():
        if not state_equal(value,parent['private'][name]):raise ValueError('private state changed: '+name)
    terms=edit_target_terms(action,target)
    errors=error_metrics(terms,target_squared,relative_squared,grouped);capture=write_span_capture(private,target)
    relative_capture=sum(sum(capture['captured_squared_by_parameter'].get(n,0.) for n in ns)/float(grouped.reference_squared[g]) for g,ns in grouped.groups.items() if float(grouped.reference_squared[g])>0)/float(relative_squared)
    objective_error=(.5*(errors['normalized_edit_error']+errors['relative_normalized_edit_error'])
                     if a.objective=='balanced' else errors['normalized_edit_error'])
    # Zero heads are feasible; a conditional minimizer cannot be worse than
    # identity in the selected normalized objective. This does not imply KL.
    if objective_error>1.+1e-7:raise ValueError('head solve worse than zero in selected objective')
    result=dict(scope=__doc__,objective=a.objective,objective_error=objective_error,
        solver_metric='weighted core norm' if a.objective=='balanced' else 'physical core norm',
        initial=initial,final=errors,matrix_solver=matrix_report,vector_solver=vector_report,
        physical_norm_squared=float(action.norm_sq()),target_norm_squared=float(target_squared),radius=float(snapshot.radius),guard=float(guard),
        editable_outputs=len(inventory),max_column_orthonormal_error=max_qerror,span_capture=capture,
        relative_span_error_floor=1-relative_capture,
        selected_error_squared=float(sum(v for n,v in terms.items() if n in selected_names)),
        zero_target_error_squared=float(sum(v for n,v in terms.items() if n not in selected_names)),
        embedding_error_squared=float(terms['embed.weight']),
        group_error_squared={g:float(sum((terms[n] for n in ns if n in terms),target_squared.new_zeros(()))) for g,ns in grouped.groups.items()},
        frozen_body_private_exactly_equal=True,actual_actor_design_parity=True,elapsed_seconds=time.perf_counter()-started,
        gpt_forwards=0,gpt_backwards=0,optimizer_updates=0,learned_preservation_claim=False)
    copied=copy.deepcopy(meta);copied.update(diagnostic_control=f'conditional_shared_head_{a.objective}_lstsq',parent_checkpoint=str(a.checkpoint),parent_sha256=sha(a.checkpoint),control_source_sha256=sha(__file__))
    copied['args']=dict(copied['args'],objective=a.objective,output=str(a.output));copied['trainable']='analytic head-only control; body/private frozen; no optimizer state';copied['loss']=a.objective+' physical target MSE, conditional shared-head least squares'
    save(a.output/'checkpoint_control.pt',dict(metadata=copied,step=parent['step'],actor=actor.state_dict(),private=private.state_dict(),diagnostic_control=True,optimizer=None,private_optimizer=None,lbfgs_optimizer=None,rng=parent['rng']))
    (a.output/'complete.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.output/'stored_readout.json').write_text(json.dumps(stored_readout(action,target,parameters,grouped,{n:v for n,v in state['model'].items() if not n.endswith('.angular_freq')}),indent=2)+'\n')
    print(json.dumps(dict(event='complete',final=errors,rank=matrix_report['rank'],seconds=result['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
